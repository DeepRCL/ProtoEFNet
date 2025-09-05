import numpy as np
from glob import glob
import logging
import sys
import os
import random
import shutil
import yaml
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import pickle
import _pickle as cPickle
import scipy.io as sio
import bz2
from typing import Dict, Any
from distutils.util import strtobool
from PIL import Image
from collections import OrderedDict
import itertools
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# TODO delete?
def cfg_parser(cfg_file: str) -> dict:
    """
    This functions reads an input config file and returns a dictionary of configurations.
    args:
        cfg_file (string): path to cfg file
    returns:
        cfg (dict)
    """
    cfg = yaml.load(open(cfg_file, "r"), Loader=yaml.FullLoader)
    cfg["cfg_file"] = cfg_file
    return cfg


def update_nested_dict(original, update):
    for key, value in update.items():
        if isinstance(value, dict):
            original[key] = update_nested_dict(original.get(key, {}), value)
        else:
            original[key] = value
    return original


def updated_config() -> Dict[str, Any]:
    # creating an initial parser to read the config.yml file.
    # useful for changing config parameters in bash when running the script
    initial_parser = argparse.ArgumentParser()
    # initial_parser.add_argument('--config_path', default="src/configs/XProtoNet_end2end.yml",
    initial_parser.add_argument(
        "--config_path",
        default="src/configs/Video_XProtoNet_e2e.yml",
        help="Path to a config",
    )
    # initial_parser.add_argument('--save_dir', default="logs/as_tom/XProtoNet_e2e/test_run_00",
    initial_parser.add_argument(
        "--save_dir",
        default="logs/ef_pilot/Video_XProtoNet_e2e/test_run_00",
        help="Path to directory for saving training results",
    )
    initial_parser.add_argument("--eval_only", default=False, help="Evaluate trained model when true")
    initial_parser.add_argument(
        "--eval_data_type",
        default="val",
        help="Data split for evaluation. either val, val_push or test",
    )
    initial_parser.add_argument(
        "--push_only",
        default=False,
        help="Push prototypes if it is true. Useful for pushing a model checkpoint.",
    )
    initial_parser.add_argument(
        "--explain_locally",
        default=False,
        help="Locally explains cases from eval_data_type split",
    )
    # TODO shouldn't we use the train images to explain a prototype globally?
    initial_parser.add_argument(
        "--explain_globally",
        default=False,
        help="Globally explains the learnt prototypes from the eval_data_type split",
    )
    initial_parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="DEBUG",
        help="Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    initial_parser.add_argument(
        "-m",
        "--comment",
        type=str,
        default="",
        help="A single line comment for the experiment",
    )
    initial_parser.add_argument(
        "--SWEEP_ID",
        type=str,
        default="",
        help="the sweep ID",
    )
    initial_parser.add_argument(
        "--SWEEP_COUNT",
        type=int,
        default=1,
        help="the number of sweep runs",
    )
    args, unknown = initial_parser.parse_known_args()

    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config["config_path"] = args.config_path
    config["save_dir"] = args.save_dir
    config["eval_only"] = args.eval_only
    config["eval_data_type"] = args.eval_data_type
    config["push_only"] = args.push_only
    config["explain_locally"] = args.explain_locally
    config["explain_globally"] = args.explain_globally
    config["log_level"] = args.log_level
    config["comment"] = args.comment
    config["SWEEP_ID"] = args.SWEEP_ID
    config["SWEEP_COUNT"] = args.SWEEP_COUNT

    # Define a custom argument type for a list of strings
    def list_of_strings(arg):
        return arg.split(',')

    def get_type_v(v):
        """
        for boolean configs, return a lambda type for argparser so string input can be converted to boolean
        """
        if type(v) == bool:
            return lambda x: bool(strtobool(x))
        elif type(v) == list:
            return lambda x: list_of_strings(x)
        else:
            return type(v)

    # creating a final parser with arguments relevant to the config.yml file
    parser = argparse.ArgumentParser()
    for k, v in config.items():
        if type(v) is not dict:
            parser.add_argument(f"--{k}", type=get_type_v(v), default=None)
        else:
            for k2, v2 in v.items():
                if type(v2) is not dict:
                    parser.add_argument(f"--{k}.{k2}", type=get_type_v(v2), default=None)
                else:
                    for k3, v3 in v2.items():
                        if type(v3) is not dict:
                            parser.add_argument(f"--{k}.{k2}.{k3}", type=get_type_v(v3), default=None)
                        else:
                            for k4, v4 in v3.items():
                                parser.add_argument(
                                    f"--{k}.{k2}.{k3}.{k4}",
                                    type=get_type_v(v4),
                                    default=None,
                                )
    args, unknown = parser.parse_known_args()

    # Update the configuration with the python input arguments
    for k, v in config.items():
        if type(v) is not dict:
            if args.__dict__[k] is not None:
                config[k] = args.__dict__[k]
        else:
            for k2, v2 in v.items():
                if type(v2) is not dict:
                    if args.__dict__[f"{k}.{k2}"] is not None:
                        config[k][k2] = args.__dict__[f"{k}.{k2}"]
                else:
                    for k3, v3 in v2.items():
                        if type(v3) is not dict:
                            if args.__dict__[f"{k}.{k2}.{k3}"] is not None:
                                config[k][k2][k3] = args.__dict__[f"{k}.{k2}.{k3}"]
                        else:
                            for k4, v4 in v3.items():
                                if args.__dict__[f"{k}.{k2}.{k3}.{k4}"] is not None:
                                    config[k][k2][k3][k4] = args.__dict__[f"{k}.{k2}.{k3}.{k4}"]
    return config


def set_seed(seed):
    """
    Set up random seed number
    """
    # # Setup random seed
    # if dist_helper.is_ddp:
    #     seed += dist.get_rank()
    # else:
    #     pass
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_save_loc(config):
    save_dir = os.path.join(config["save_dir"])
    print("save_dir: ", save_dir) #TODO: debugging
    #################### Updating the save_dir to avoid overwriting on existing trained models ##################
    # if the save_dir directory exists, find the most recent identifier and increment it
    if os.path.exists(save_dir):
        if os.path.exists(config["model"]["checkpoint_path"]):
            save_dir = os.path.dirname(config["model"]["checkpoint_path"])
            print(
                f"###### Checkpoint '{os.path.basename(config['model']['checkpoint_path'])}'"
                f" provided in path '{save_dir}' ####### \n"
            )
        else:
            print(f"Existing save_dir: {save_dir}\n" f"incrementing the folder number")
            print(sorted(glob(f"{save_dir[:-3]}*")))
            print(sorted(glob(f"{save_dir[:-3]}*"))[-1][-2:])
            run_id = int(sorted(glob(f"{save_dir[:-3]}*"))[-1][-2:]) + 1
            save_dir = f"{save_dir[:-3]}_{run_id:02}"
            print(f"New location to save the log: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "img"), exist_ok=True)
    config["save_dir"] = save_dir


def save_configs(config):
    save_dir = os.path.join(config["save_dir"])

    # ############# Document configs ###############
    config_dir = os.path.join(save_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    if config["eval_only"]:
        config_path = os.path.join(config_dir, f"eval_{config['eval_data_type']}_config.yml")
    elif config["push_only"]:
        config_path = os.path.join(config_dir, "push_config.yml")
    elif config["explain_locally"]:
        config_path = os.path.join(config_dir, "explain_locally_config.yml")
    elif config["explain_globally"]:
        config_path = os.path.join(config_dir, "explain_globally_config.yml")
    else:
        config_path = os.path.join(config_dir, "train_config.yml")
    with open(config_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def dict_print(a_dict):
    for k, v in a_dict.items():
        logging.info(f"{k}: {v}")


def print_run_details(config, input_shape):
    print(f"input shape = {input_shape}")


######### Logging
def set_logger(logdir, log_level, filename, comment):
    """
    Set up global logger.
    """
    log_file = os.path.join(logdir, log_level.lower() + f"_{filename}.log")
    logger_format = comment + "| %(asctime)s %(message)s"
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format=logger_format,
        datefmt="%m-%d %H:%M:%S",
        handlers=[fh, logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)  # remove excessive matplotlib messages
    logging.getLogger("matplotlib").setLevel(logging.INFO)  # remove excessive matplotlib messages
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info("EXPERIMENT BEGIN: " + comment)
    logging.info("logging into %s", log_file)


def backup_code(logdir):
    code_path = os.path.join(logdir, "code")
    dirs_to_save = ["src"]
    os.makedirs(code_path, exist_ok=True)
    # os.system("cp ./*py " + code_path)
    [shutil.copytree(os.path.join("./", this_dir), os.path.join(code_path, this_dir), dirs_exist_ok=True) for this_dir in dirs_to_save]


def print_cuda_statistics():
    import sys
    from subprocess import call
    import torch

    logger = logging.getLogger("Cuda Statistics")
    logger.info("__Python VERSION:  {}".format(sys.version))
    logger.info("__pyTorch VERSION:  {}".format(torch.__version__))
    logger.info("__CUDA VERSION")
    # call(["nvcc", "--version"])
    logger.info("__CUDNN VERSION:  {}".format(torch.backends.cudnn.version()))
    logger.info("__Number CUDA Devices:  {}".format(torch.cuda.device_count()))
    logger.info("__Devices")
    call(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
        ]
    )
    logger.info("Active CUDA Device: GPU {}".format(torch.cuda.current_device()))
    logger.info("Available devices  {}".format(torch.cuda.device_count()))
    logger.info("Current cuda device  {}".format(torch.cuda.current_device()))


######### ProtoPNet helpers
def makedir(path):
    """
    if path does not exist in the file system, create it
    """
    if not os.path.exists(path):
        os.makedirs(path)


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


######## Visualization
def load_image(filepath):
    pil_image = Image.open(filepath)
    return pil_image


def plot_image(ax, pil_image):
    ax.imshow(pil_image)


def plot_bbox(ax, bbox, linewidth=3):  # TODO add label if available
    """
    Plots bounding box around regions of interest given the bbox labels. This was used before in Hitachi work for insulator damages with damage labels
    :param ax: matplotlib figure axis
    :param bbox: bbox in format of (x, y, width, height)
    :param linewidth: width of the line
    :return:
    """
    # select color based on the label type!
    color = "red"

    # Create a Rectangle patch
    x, y, width, height = bbox
    rect = patches.Rectangle(
        xy=(x, y),
        width=width,
        height=height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        label="label",
    )  # set this up if used
    ax.add_patch(rect)

    # marker on top left of the box
    ax.plot(x, y, marker="x", color="white", markersize=8)


def visualize_img_with_bbox(ax, pil, df_case, linewidth=3):
    plot_image(ax, pil)

    for i, row in df_case.iterrows():
        plot_bbox(ax, row.bbox, linewidth=linewidth)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


# TODO check if it's the same as the one in push.py
# def plot_source_with_bbox(ax, filepath, img_size, df_case, img_array=None):
#     # load image pil
#     img_pil = load_image(filepath)
#
#     # transform image and bboxes
#     transform_list = [A.Resize(img_size, img_size, interpolation=0)]
#     transform_album = A.Compose(transform_list, bbox_params=A.BboxParams(format='coco', label_fields=['conditions']))
#     transformed = transform_album(image=np.asarray(img_pil),
#                                   bboxes=df_case.bbox.to_list(),
#                                   conditions=df_case.conditions.to_list())
#     transformed_image = transformed['image']
#     transformed_df_case = pd.DataFrame({
#         'conditions': transformed['conditions'],
#         'bbox': transformed['bboxes'],
#     })
#
#     # visualize on the ax
#     if img_array is None:
#         img_pil_transformed = Image.fromarray(np.uint8(transformed_image))
#         linewidth = 1
#     else:
#         img_pil_transformed = Image.fromarray(np.uint8(img_array))
#         linewidth = 1
#     visualize_img_with_bbox(ax, img_pil_transformed, transformed_df_case, linewidth=linewidth)


def plot_source(ax, filepath, img_size, img_array=None):
    raise "not implemented"
    # # load image pil
    # img_pil = load_image(filepath)
    #
    # # visualize on the ax
    # img_pil_transformed = Image.fromarray(np.uint8(img_array))
    # visualize_img_with_bbox(ax, img_pil_transformed)


######## Pickle loading and saving
def load_pickle(pickle_path, log=print):
    with open(pickle_path, "rb") as handle:
        pickle_data = pickle.load(handle)
        log(f"data successfully loaded from {pickle_path}")
    return pickle_data

def decompress_pickle(path):
    """
    Load any compressed pickle file
    :param path: file path with extension .pbz2
    :return:
    """
    data = bz2.BZ2File(path, 'rb')
    data = cPickle.load(data)
    return data

def save_pickle(pickle_data, pickle_path, log=print):
    with open(pickle_path, "wb") as handle:
        pickle.dump(pickle_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"data successfully saved in {pickle_path}")

def load_data(path):
    """
    :param path:    the path to file to load
    """
    if path.endswith('.pbz2'):
        data_dict = decompress_pickle(path)
        cine = data_dict['resized'].transpose(2,0,1)
    elif path.endswith('.mat'):
        matfile = sio.loadmat(path, verify_compressed_data_integrity=False)
        cine = matfile['Patient']['DicomImage'][0][0]   # (H,W,T)
        cine = cine.transpose(2,0,1)    # (T,H,W)
    elif path.endswith('.npz'):
        cine = np.load(path)['arr_0']
    else:
        raise (f"loading file with format {path.split('.')[-1]} is not supported")

    return cine

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm,  cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #plt.text(j, i, format(cm[i, j], fmt),
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.colorbar()

def plot_embeddings(prototypes, labels, savepath, embed_type='umap', dim = '2D', k=5, train_points=None, test_points=None, train_labels=None, test_labels=None, train_similarities=None, test_similarities=None, test_type="TEST"):
    """ 
    Args:
        prototype_vectors ([type]): [description] torch
        labels ([type]): [description] is prototype labels torch
        savepath ([type]): [description]
        embed_type (str, optional): [description]. Defaults to 'umap'.
        dim (str, optional): Whether to plot the embeddings in 2D ('2D') or 3D ('3D')
        k (int, optional): [description]. Defaults to 5. is the number of closest embeddings to each prototype to plot
        train_points ([type], optional): [description]. Defaults to None. torch
        test_points ([type], optional): [description]. Defaults to None. torch
        train_labels ([type], optional): [description]. Defaults to None. numpy
        test_labels ([type], optional): [description]. Defaults to None. numpy
        train_similarities ([type], optional): [description]. Defaults to None. shape (N, P) torch
        test_similarities ([type], optional): [description]. Defaults to None. shape (N, P) torch
        all numpy
    """

    assert embed_type in ['tsne', 'PCA', "umap"], f"Embedding type {embed_type} is not implemented"
    num_protos = prototypes.shape[0]
    labels = labels.cpu().numpy()

    if train_points is not None:
        train_points = train_points.cpu().numpy()
        if len(train_points.shape) > 2:
            train_labels = np.repeat(train_labels, train_points.shape[1]).reshape(train_points.shape[0], train_points.shape[1]) # shape (N, P)
            
            #swap = np.swapaxes(train_points, 1, 2)
            #train_points = swap.reshape(-1, swap.shape[2])
            train_points = train_points.reshape(-1, train_points.shape[2]) # shape (N*P, 256)
            
        #all_points = np.vstack([torch.squeeze(prototypes).cpu().numpy(), train_points])
        all_points = np.vstack([prototypes.cpu().detach().numpy(), train_points])

    else:
        all_points = prototypes.cpu().numpy()
    
    if test_points is not None:
        test_points = test_points.cpu().numpy()
        if len(test_points.shape) > 2:
            test_labels = np.repeat(test_labels, test_points.shape[1]).reshape(test_points.shape[0], test_points.shape[1]) # shape (N, P)
            
            #swap = np.swapaxes(test_points, 1, 2)
            #test_points = swap.reshape(-1, swap.shape[2])
            test_points = test_points.reshape(-1, test_points.shape[2]) # shape (N*P, 256)

    if dim == '2D':
        n_components = 2
    elif dim == '3D':
        n_components = 3
    else:
        raise ValueError(f'Dim is not implemented for: {dim}')

    ################################# Reduce dimension #######################################
    if embed_type == 'tsne': # doesnt work on same coord for train and test
        trans = TSNE(n_components=n_components, init='pca', learning_rate = 'auto').fit(all_points)
        embed = trans.embedding_
        test_embed = trans.transform(test_points)
        #embed = TSNE(n_components=n_components, init='pca', learning_rate = 'auto').fit_transform(all_points)
    elif embed_type == 'PCA':
        # normalise the data before PCA
        all_points = normalize(all_points)
        if test_points is not None:
            test_points = normalize(test_points)
        trans = PCA(n_components=n_components).fit(all_points)
        embed = trans.transform(all_points)
        test_embed = trans.transform(test_points)
    elif embed_type == 'umap':
        #embed = umap.UMAP(n_components=n_components).fit_transform(all_points)
        trans = umap.UMAP(n_components=n_components).fit(all_points)
        embed = trans.embedding_
        test_embed = trans.transform(test_points)

    # # if labels of prototypes are in one hot format convert them
    # labels = labels.cpu().numpy()
    # if len(labels.shape) != 1:
    #     labels = np.argmax(labels, axis=1)

    ############ Calculate the closest k train and test samples for each prototype ############
    if train_points is not None:
        train_sims , train_inds = torch.topk(train_similarities, k=k, dim=0)#, largest=False) # shape (k, P) train_inds are the indices of the closest points (row of similarities)
        # transform the train embeddings back to the original shape for correct indexing
        train_embeds = embed[num_protos:] # N * P, 2
        # transform train_embeds to the shape of the original train_points: N, P, 2 without changing the data
        train_embeds = train_embeds.reshape(train_labels.shape[0], train_labels.shape[1], n_components)
        # get the closest points to each prototype
        train_closest_points = train_embeds[train_inds, np.arange(train_inds.shape[1]), :] # shape (k, P, 2)
        train_closest_labels = train_labels[train_inds, np.arange(train_inds.shape[1])] # shape (k, P) 
        
        # reshape train_closest_points to (k*P, 2) for plotting without changing the data
        train_closest_points = train_closest_points.reshape(-1, n_components)
        train_closest_labels = train_closest_labels.reshape(-1) # shape (k*P)


    
    if test_points is not None:
        test_sims , test_inds = torch.topk(test_similarities, k=k, dim=0)#, largest=False) # shape (N, k) train_inds are the indices of the closest points to each prototype (row of similarities)
        # transform the test embeddings back to the original shape for correct indexing
        test_embed = test_embed.reshape(test_labels.shape[0], test_labels.shape[1], n_components) # N, P, 2
        # get the closest points to each prototype
        test_closest_points = test_embed[test_inds, np.arange(test_inds.shape[1]), :] # shape (k, P, 2)
        test_closest_labels = test_labels[test_inds, np.arange(test_inds.shape[1])] # shape (k, P) 
        
        # reshape test_closest_points to (k*P, 2) for plotting without changing the data
        test_closest_points = test_closest_points.reshape(-1, n_components) 
        test_closest_labels = test_closest_labels.reshape(-1) # shape (k*P)

    #################################### make actual figure ###################################
    # visualise only the 10 closest points to each prototype using similarity
    
    fig = plt.figure()
    if dim == '3D': 
        ax = fig.add_subplot(projection='3d')
        embeds_protos = [embed[:num_protos, 0], embed[:num_protos, 1], embed[:num_protos,2]]
    else:
        ax = fig.add_subplot()
        embeds_protos = [embed[:num_protos, 0], embed[:num_protos, 1]]

    if train_points is not None:
        if dim == '3D':
            embeds_points = [train_closest_points[:, 0], train_closest_points[:, 1], train_closest_points[:,2]]
        else:
            embeds_points = [train_closest_points[:, 0], train_closest_points[:, 1]]
        
        scatt = ax.scatter(*embeds_points, c=train_closest_labels, marker='*', alpha = 0.3)

    scatt = ax.scatter(*embeds_protos, c=labels, edgecolors = 'k', alpha =0.5)

    #legend1 = ax.legend(*scatt.legend_elements(), title="EF", loc = 'upper left')
    #legend1 = ax.legend(loc=1, mode='expand', numpoints=1, ncol=9, fancybox = True,
    #       fontsize='small', *scatt.legend_elements())
    #ax.add_artist(legend1)
    plt.title(f'{embed_type} of Train Data with Prototypes (top {k} closest patches)')
    plt.colorbar(scatt, ax=ax)
    plt.savefig(os.path.join(savepath, f'{embed_type}_train_prototypes_top{k}.png'))
    plt.close('all')

    # plot tests next to prototypes
    fig = plt.figure()
    if dim == '3D': 
        ax = fig.add_subplot(projection='3d')
        embeds_protos = [embed[:num_protos, 0], embed[:num_protos, 1], embed[:num_protos,2]]
    else:
        ax = fig.add_subplot()
        embeds_protos = [embed[:num_protos, 0], embed[:num_protos, 1]]

    if test_points is not None:
        if dim == '3D':
            embeds_points = [test_closest_points[:, 0], test_closest_points[:, 1], test_closest_points[:,2]]
        else:
            embeds_points = [test_closest_points[:, 0], test_closest_points[:, 1]]
        
        scatt = ax.scatter(*embeds_points, c=test_closest_labels, marker='*', alpha = 0.3)

    scatt = ax.scatter(*embeds_protos, c=labels, edgecolors = 'k', alpha =0.5)

    # legend1 = ax.legend(*scatt.legend_elements(), title="EF", loc = 'upper left')
    # ax.add_artist(legend1)
    plt.colorbar(scatt, ax=ax)

    plt.title(f'{embed_type} Plot of {test_type} Samples with Prototypes')
    plt.savefig(os.path.join(savepath, f'{embed_type}_{test_type}_prototypes_top{k}.png'))
    plt.close('all')