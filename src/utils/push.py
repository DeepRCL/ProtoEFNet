import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

from src.utils.utils import makedir, save_pickle
from tqdm import tqdm

"""
Search the training set for image patches that are semantically closest to
each learned prototype, then updates the prototypes to those image patches.

To do this, it computes the image patch embeddings (IPBs) and saves those
closest to the prototypes. It also saves the prototype-to-IPB distances and
predicted occurrence maps.

If abstain_class==True, it assumes num_classes actually equals to K+1, where
K is the number of real classes and the last class is the extra "abstain" 
class for uncertainty estimation.
"""


def push_prototypes(
    dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
    # dataset,   # pytorch dataset for train_push group
    model,  # pytorch network with prototype_vectors
    class_specific=True,
    preprocess_input_function=None,  # normalize if needed
    abstain_class=True,  # enable abstain class as the K+1-th class
    prototype_layer_stride=1,
    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved here
    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    proto_bound_boxes_filename_prefix=None,
    save_prototype_class_identity=True,  # which class the prototype image comes from
    log=print,
    prototype_activation_function_in_numpy=None,
    replace_prototypes=True,
):
    model.eval()
    start = time.time()

    prototype_shape = model.prototype_shape
    n_prototypes = model.num_prototypes
    num_classes = model.num_classes

    log(f"##### push at epoch {epoch_number} with abstain class {abstain_class} #####")

    # saves the maximum similarity seen so far for each prototype (useful when dataset size is too large to save all)
    global_min_proto_dist = np.full(n_prototypes, np.inf)  # TODO remove?
    # saves the patch representation that gives the current smallest distance. Replaces the prototype vectors at the end
    global_min_fmap_patches = np.zeros([n_prototypes, prototype_shape[1]])  # same shape as protoL_input_torch

    """
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    """
    # if save_prototype_class_identity:
    #     proto_rf_boxes = np.full(shape=[n_prototypes, 5+model.num_classes],
    #                                 fill_value=-1)
    #     proto_bound_boxes = np.full(shape=[n_prototypes, 5+model.num_classes],
    #                                         fill_value=-1)
    # else:
    #     proto_rf_boxes = np.full(shape=[n_prototypes, 5],
    #                                 fill_value=-1)
    #     proto_bound_boxes = np.full(shape=[n_prototypes, 5],
    #                                         fill_value=-1)

    # creating the folder (with epoch number) to save the prototypes' info and visualizations
    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, "epoch-" + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    protoL_input_ = []  # saves the input to prototypical layer (the ROI features), shape (N, P, 128)
    proto_dist_ = []  # saves the distances to prototypes (distance = 1-CosineSimilarities). shape (N, P)
    occurrence_map_ = []  # saves the occurence maps of cases seen. shape (N, P, 1, H, W)
    inputs = []  # saves the input images (may not be able to hold all if RAM is not enough). shape (N, 3, H, W)
    ys_gt = []  # saves the gt label of cases seen. shape (N)
    ys_pred = []  # saves the prediction probabilities of cases seen. shape (N, 4)
    filenames = []  # saves the filenames of cases seen. shape (N)

    data_iter = iter(dataloader)
    iterator = tqdm(range(len(dataloader)), dynamic_ncols=True)
    for push_iter in iterator:
        data_sample = next(data_iter)
        search_batch_input = data_sample["cine"]
        search_y = data_sample["target_AS"]

        inputs.extend(search_batch_input.detach().cpu().numpy())  # TODO shape
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
            y_pred = logits.softmax(dim=1)

        protoL_input_.extend(protoL_input_torch.detach().cpu().numpy())
        proto_dist_.extend(proto_dist_torch.detach().cpu().numpy())
        occurrence_map_.extend(occurrence_map_torch.detach().cpu().numpy())
        ys_pred.extend(y_pred.detach().cpu().numpy())

    protoL_input_ = np.asarray(protoL_input_)  # shape (N, P, 128)
    proto_dist_ = np.asarray(proto_dist_)  # shape (N, P)
    occurrence_map_ = np.asarray(occurrence_map_)  # shape (N, P, 1, H, W)
    inputs = np.asarray(inputs)  # shape (N, 3, H, W)
    ys_gt = np.asarray(ys_gt)  # shape (N)
    ys_pred = np.asarray(ys_pred)  # shape (N, 4)

    # group cases based on their class label (gt)
    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(ys_gt):
            # Multi-Class setting
            if len(img_y.shape) == 0:
                img_label = img_y.item()
                class_to_img_index_dict[img_label].append(img_index)
                # if the abstention class is included,
                # we want the "abstain" class to be able to push from any image class
                if abstain_class:
                    class_to_img_index_dict[num_classes - 1].append(img_index)
            # Multi-Label setting (does not support abstain_class yet)
            elif len(img_y.shape) == 1:
                img_label = [i for i, e in enumerate(img_y) if e != 0]
                for indx in img_label:
                    class_to_img_index_dict[indx].append(img_index)
            else:
                raise "invalid ground truth labels. can only handle multi-class (shapeless) or multi-label (shape=(n))"

    # keep track of prototypes information
    prototypes_filenames = []
    # prototypes_bb_pandas = []  # to save bbox. Can add later if we had bbox info!
    prototypes_src_imgs = []
    prototypes_gts = []
    prototypes_preds = []
    prototypes_occurrence_maps = []
    prototypes_similarity_to_src_ROIs = []

    for j in range(n_prototypes):
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(model.prototype_class_identity[j]).item()
            # if there is not images of the target_class we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:, j]

        min_proto_dist_j = np.amin(proto_dist_j)
        if min_proto_dist_j < global_min_proto_dist[j]:
            # retrieve the case index of the feature patch that is the closest to the prototype
            argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape))
            if class_specific:
                """
                change the argmin index from the index among
                images of the target class to the index in the entire search group
                """
                argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][argmin_proto_dist_j[0]]
            img_index = argmin_proto_dist_j[0]

            # retrieve the feature of the corresponding patch that was compared with the prototype (prototpe input)
            min_fmap_patch_j = protoL_input_[img_index, j]

            global_min_proto_dist[j] = min_proto_dist_j
            global_min_fmap_patches[j] = min_fmap_patch_j

            prototypes_similarity_to_src_ROIs.append(1 - min_proto_dist_j)

            # get the filename
            filename_j = filenames[img_index]
            prototypes_filenames.append(filename_j)
            # get bbox information from dataset, to be used later when bbox is available
            # df_case = dataset.get_filename_df_case(filename_j, ['glaze_Flashover damage', 'shell_Broken'])
            # prototypes_bb_pandas.append(df_case)
            # get the prediction
            pred = ys_pred[img_index]
            prototypes_preds.append(pred)
            prototypes_gts.append(ys_gt[img_index])

            # get the source image and rescale it
            original_img_j = inputs[img_index]
            # rescaled_original_img_j = original_img_j - np.amin(original_img_j)
            # rescaled_original_img_j = rescaled_original_img_j / np.amax(rescaled_original_img_j)
            m = 0.099
            std = 0.171
            rescaled_original_img_j = original_img_j * std + m
            prototypes_src_imgs.append(rescaled_original_img_j.copy())
            rescaled_original_img_j = np.transpose(rescaled_original_img_j, (1, 2, 0))
            original_img_size = rescaled_original_img_j.shape[0]

            # resize the occurrence map
            occurrence_map_j = occurrence_map_[img_index, j]
            prototypes_occurrence_maps.append(occurrence_map_j)
            upsampled_occurrence_map_j = cv2.resize(
                occurrence_map_j.squeeze(),
                dsize=(original_img_size, original_img_size),
                interpolation=cv2.INTER_CUBIC,
            )
            # normalize the occurrence map
            rescaled_occurrence_map_j = upsampled_occurrence_map_j - np.amin(upsampled_occurrence_map_j)
            rescaled_occurrence_map_j = rescaled_occurrence_map_j / np.amax(rescaled_occurrence_map_j)

            # image masked with normalized occurrence map
            mask = rescaled_occurrence_map_j[:, :, np.newaxis]
            prototype_img_j = rescaled_original_img_j * mask

            # image with normalized occurrence map overlay
            heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_occurrence_map_j), cv2.COLORMAP_TURBO)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]
            overlayed_original_img_j = 0.5 * rescaled_original_img_j + 0.3 * heatmap

            #### PLOTTING
            fig, axs = plt.subplots(1, 4, figsize=(20, 6))

            # 1. plot the source image of prototype with its label annotations
            axs[0].imshow(rescaled_original_img_j)  # TODO denomarlize maybe using our normalization?
            axs[0].title.set_text("source")
            # 2. plot the image masked out with normalized softmaxed occurrence map
            # 3. plot the image with occurrence map overlaid
            img = {"masked": prototype_img_j, "overlay": overlayed_original_img_j}
            i = 1
            for key, item in img.items():
                # plot the image
                axs[i].imshow(item)
                axs[i].title.set_text(key)
                i += 1
            # 4. the mask (non-rescaled) heatmap alone
            im = axs[i].imshow(upsampled_occurrence_map_j)
            axs[i].title.set_text("mask")
            fig.colorbar(im, ax=axs[i], shrink=0.75)

            # add super title for the figure
            fig.suptitle(
                f"p_{j:02d}   |  {filename_j}  |  img_pred = {[f'{pred[i]:.2f}' for i in range(pred.shape[0])]}",
                fontsize=15,
            )
            # some visual configs for the figure
            fig.tight_layout()
            plt.savefig(os.path.join(proto_epoch_dir, f"{j:02d}_{filename_j}.png"))

            plt.close()

    prototypes_filenames = np.asarray(prototypes_filenames)
    # prototypes_bb_pandas = np.asarray(prototypes_bb_pandas)
    prototypes_src_imgs = np.asarray(prototypes_src_imgs)
    prototypes_gts = np.asarray(prototypes_gts)
    prototypes_preds = np.asarray(prototypes_preds)
    prototypes_occurrence_maps = np.asarray(prototypes_occurrence_maps)
    prototypes_similarity_to_src_ROIs = np.asarray(prototypes_similarity_to_src_ROIs)

    prototype_data_dict = {
        "prototypes_filenames": prototypes_filenames,
        "prototypes_src_imgs": prototypes_src_imgs,
        "prototypes_gts": prototypes_gts,
        "prototypes_preds": prototypes_preds,
        "prototypes_occurrence_maps": prototypes_occurrence_maps,
        "prototypes_similarity_to_src_ROIs": prototypes_similarity_to_src_ROIs,
        # "prototypes_bb_pandas": prototypes_bb_pandas
    }
    save_pickle(prototype_data_dict, f"{proto_epoch_dir}/prototypes_info.pickle")

    if class_specific:
        del class_to_img_index_dict

    # USELESS stuff?
    # if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
    #     np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
    #             proto_rf_boxes)
    #     np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
    #             proto_bound_boxes)

    if replace_prototypes:
        log("\tExecuting push ...")
        prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
        model.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    end = time.time()
    log("\tpush time: \t{0}".format(end - start))
