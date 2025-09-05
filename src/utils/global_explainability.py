import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from src.utils.utils import makedir, plot_source, load_pickle, save_pickle
from tqdm import tqdm

# from MAB-beta ranking of model with absoluted Occurrence map and color jitter data augmentation
# Oct27_AbsOM_ColJitter | Epoch 85
prototypes_MAB_ranking = [
    16,
    3,
    5,
    1,
    10,
    12,
    7,
    4,
    19,
    9,
    0,
    17,
    13,
    11,
    15,
    14,
    6,
    18,
    8,
    2,
]


def explain_global(
    mode,  # val or test
    dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
    dataset,  # pytorch dataset for train_push group
    model,  # pytorch network with prototype_vectors
    preprocess_input_function=None,  # normalize if needed
    model_directory=None,  # if not None, explainability results will be saved here
    epoch_number=0,
    log=print,
):
    if model_directory is not None:
        root_dir_for_saving = os.path.join(model_directory, f"epoch_{epoch_number}")
    else:
        root_dir_for_saving = f"global_explain_model_epoch_{epoch_number}"

    model.eval()
    log(f"\t global explanation of model in {root_dir_for_saving}")

    ##### Loading the prototype information (src img, heatmap, etc)
    prototypes_info_path = os.path.join(model_directory, f"img/epoch-{epoch_number}_pushed/prototypes_info.pickle")
    if os.path.exists(prototypes_info_path):
        prototype_data_dict = load_pickle(prototypes_info_path, log)
    else:
        raise f"path {prototypes_info_path} does not exist. Project the prototypes with Push function first"

    ##### Loading/creating the validation/training data and model products
    data_dict_path = f"logs/IDID/{mode}_data.pickle"
    makedir(os.path.dirname(data_dict_path))
    model_products_path = f"{root_dir_for_saving}/{mode}/model_products.pickle"
    makedir(os.path.dirname(model_products_path))
    if os.path.exists(model_products_path):
        data_dict = load_pickle(data_dict_path, log)
        log(f"img    and labels and filenames of {mode}-dataset is loaded")

        model_products_dict = load_pickle(model_products_path, log)
        log(
            f"model products: prototypical layer input, similarity scores (distances), "
            f"ROI maps, predictions, and FC Layer weights for model {root_dir_for_saving} "
            f"for {mode}-dataset is loaded"
        )
    else:
        log(f"model products not saved. running the epoch on {mode}-dataset to save the results.")
        protoL_input_ = []
        proto_dist_ = []
        occurrence_map_ = []
        inputs = []
        ys_gt = []
        ys_pred = []
        filenames = []
        fc_layer_weights = model.last_layer.weight.detach().cpu().numpy()

        data_iter = iter(dataloader)
        iterator = tqdm(range(len(dataloader)), dynamic_ncols=True)
        for i in iterator:
            data_sample = next(data_iter)
            search_batch_input = data_sample["img"]
            search_y = data_sample["label"]

            inputs.extend(search_batch_input.detach().cpu().numpy())
            ys_gt.extend(search_y.detach().cpu().numpy())
            filenames.extend(data_sample["filename"])

            if preprocess_input_function is not None:
                search_batch = preprocess_input_function(search_batch_input)
            else:
                search_batch = search_batch_input

            with torch.no_grad():
                search_batch = search_batch.cuda()
                (
                    protoL_input_torch,
                    proto_dist_torch,
                    occurrence_map_torch,
                    logits,
                ) = model.push_forward(search_batch)
                y_pred = torch.sigmoid(logits)

            protoL_input_.extend(protoL_input_torch.detach().cpu().numpy())
            proto_dist_.extend(proto_dist_torch.detach().cpu().numpy())
            occurrence_map_.extend(occurrence_map_torch.detach().cpu().numpy())
            ys_pred.extend(y_pred.detach().cpu().numpy())

        protoL_input_ = np.asarray(protoL_input_)
        proto_dist_ = np.asarray(proto_dist_)
        occurrence_map_ = np.asarray(occurrence_map_)
        inputs = np.asarray(inputs)
        ys_gt = np.asarray(ys_gt)
        ys_pred = np.asarray(ys_pred)

        ### Sanity Check
        from sklearn.metrics import multilabel_confusion_matrix, f1_score

        ys_pred_orig = ys_pred > 0.5
        f1_orig = f1_score(ys_gt, ys_pred_orig, average=None, zero_division=0)
        cm = multilabel_confusion_matrix(ys_gt, ys_pred_orig)
        log(f"f1 score is {f1_orig}")
        log(f"confusion matrix is \n{cm}")

        data_dict = {
            "inputs": inputs,
            "ys_gt": ys_gt,
            "filenames": filenames,
        }
        model_products_dict = {
            "fc_layer_weights": fc_layer_weights,
            "protoL_input_": protoL_input_,
            "proto_dist_": proto_dist_,
            "occurrence_map_": occurrence_map_,
            "ys_pred": ys_pred,
        }

        save_pickle(data_dict, data_dict_path)
        save_pickle(model_products_dict, model_products_path)

    n_prototypes = model.num_prototypes
    num_classes = model.num_classes
    fc_layer_weights = model_products_dict["fc_layer_weights"]
    similarities_ = 1 - model_products_dict["proto_dist_"]
    n_prototypes_per_class = n_prototypes // num_classes

    #########################################
    ##### Process the prototype info
    # get the source image
    prototypes_src_imgs = np.transpose(
        prototype_data_dict["prototypes_src_imgs"], (0, 2, 3, 1)
    )  # shape (n_prototypes, 224,224, 3)
    img_size = prototypes_src_imgs.shape[1]

    # resize the occurrence map  shape (224,224,n_prototypes)
    prototypes_occurrence_maps = np.transpose(prototype_data_dict["prototypes_occurrence_maps"].squeeze(), (1, 2, 0))
    prototypes_upsampled_occurrence_maps = cv2.resize(
        prototypes_occurrence_maps,
        dsize=(img_size, img_size),
        interpolation=cv2.INTER_CUBIC,
    )
    # normalize the occurrence map
    prototypes_rescaled_occurrence_maps = prototypes_upsampled_occurrence_maps - np.amin(
        prototypes_upsampled_occurrence_maps, axis=(0, 1)
    )
    prototypes_rescaled_occurrence_maps = prototypes_rescaled_occurrence_maps / np.amax(
        prototypes_rescaled_occurrence_maps, axis=(0, 1)
    )
    prototypes_rescaled_occurrence_maps = np.transpose(prototypes_rescaled_occurrence_maps, (2, 0, 1))

    # prototype src image masked with normalized occurrence map
    mask = np.expand_dims(prototypes_rescaled_occurrence_maps, axis=3)
    masked_prototpe_src_img = prototypes_src_imgs * mask

    # image with normalized occurrence map overlay
    prototypes_heatmaps = [
        cv2.applyColorMap(np.uint8(255 * prototypes_rescaled_occurrence_maps[p]), cv2.COLORMAP_TURBO)
        for p in range(n_prototypes)
    ]
    prototypes_heatmaps = np.float32(prototypes_heatmaps) / 255
    prototypes_heatmaps = prototypes_heatmaps[..., ::-1]
    prototypes_overlayed_imgs = 0.5 * prototypes_src_imgs + 0.3 * prototypes_heatmaps

    iterator = tqdm(range(n_prototypes), dynamic_ncols=True)
    for p in iterator:
        if p < 10:
            prototype_class = "flash"
        else:
            prototype_class = "broken"

        #### PLOTTING
        fig, axs = plt.subplots(10, 4, figsize=(12, 30))

        prototype_filename = prototype_data_dict["prototypes_filenames"][p]
        # add bbox information if available #TODO change to be similar to push.py
        plot_source(axs[3, 0], prototype_filename, img_size)  # prototype_data_dict['prototypes_bb_pandas'][p])
        axs[3, 0].title.set_text(f"MAB_Rank {prototypes_MAB_ranking.index(p)} | {prototype_filename}")
        axs[4, 0].imshow(prototypes_overlayed_imgs[p])
        axs[5, 0].imshow(masked_prototpe_src_img[p])

        #########################################
        # find the top 10 closest validation images
        similarities = similarities_[:, p]
        sorted_similarities_indices = np.argsort(similarities)
        top_10_indices = sorted_similarities_indices[..., ::-1][:10]
        for i, indx in enumerate(top_10_indices):
            #########################################
            ##### get the img's information
            filename = data_dict["filenames"][indx]
            src_img = data_dict["inputs"][indx]  # shape = (3, 224, 224)
            gt = data_dict["ys_gt"][indx]  # shape = (num_classes)
            df_case = dataset.get_filename_df_case(
                filename, ["glaze_Flashover damage", "shell_Broken"]
            )  # bbox information
            ##### get the model products for the img
            occurrence_map = model_products_dict["occurrence_map_"][indx]
            protoL_input = model_products_dict["protoL_input_"][indx]  # shape = (n_prototypes, 128, 1, 1)
            similarity = similarities[indx]
            pred = model_products_dict["ys_pred"][indx]  # shape = (num_classes)

            #########################################
            ##### prepare the plot
            # get the source image
            src_img = np.transpose(src_img, (1, 2, 0))
            src_img_size = src_img.shape[0]

            # resize the occurrence map  shape (224,224,n_prototypes)
            occurrence_map = np.transpose(occurrence_map.squeeze(), (1, 2, 0))
            upsampled_occurrence_map = cv2.resize(
                occurrence_map,
                dsize=(src_img_size, src_img_size),
                interpolation=cv2.INTER_CUBIC,
            )
            # normalize the occurrence map
            rescaled_occurrence_map = upsampled_occurrence_map - np.amin(upsampled_occurrence_map, axis=(0, 1))
            rescaled_occurrence_map = rescaled_occurrence_map / np.amax(rescaled_occurrence_map, axis=(0, 1))

            # image masked with normalized occurrence map
            mask = rescaled_occurrence_map[:, :, p, np.newaxis]
            img_masked_prototpe_j = src_img * mask

            # image with normalized occurrence map overlay
            heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_occurrence_map[:, :, p]), cv2.COLORMAP_TURBO)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]
            overlayed_original_img_j = 0.5 * src_img + 0.3 * heatmap

            # 1. plot the source image of prototype with its label annotations
            # add bbox information if available
            plot_source(axs[i, 1], filename, src_img_size)  # df_case)  #TODO change to be similar to push.py
            axs[i, 1].title.set_text(
                f"pred = {[f'{pred[c]:.2f}' for c in range(pred.shape[0])]} \n"
                f"gt = {[f'{gt[c]:.0f}' for c in range(gt.shape[0])]}"
            )
            # 2. plot the image masked out with normalized softmaxed occurrence map
            axs[i, 2].imshow(overlayed_original_img_j)
            # 3. plot the image with occurrence map overlaid
            axs[i, 3].imshow(img_masked_prototpe_j)
            axs[i, 3].title.set_text(f"Cosine Similarity to p_{p:d}={similarity:.2f}")

        # add super title for the figure
        fig.suptitle(
            f"{prototype_data_dict['prototypes_filenames'][p]}"
            # f"| similarity to p_{p:02d}={similarities[p]:.2f} | pred = {[f'{pred[c]:.3f}' for c in range(pred.shape[0])]} "
            # f"| gt = {[f'{gt[c]:.0f}' for c in range(gt.shape[0])]}"
            ,
            fontsize=15,
        )
        # some visual configs for the figure
        fig.tight_layout()
        makedir(os.path.join(root_dir_for_saving, mode, "global"))
        plt.savefig(
            os.path.join(
                root_dir_for_saving,
                mode,
                "global",
                f'{p:02d}_{prototype_class}_{prototype_data_dict["prototypes_filenames"][p]}.png',
            )
        )

        plt.close()
