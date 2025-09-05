import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

from src.utils.utils import makedir, save_pickle
from src.utils.video_utils import saveVideo
from tqdm import tqdm
from moviepy.video.io.bindings import mplfig_to_npimage

def prototype_plot_frame(
    unnorm_img, 
    upsampled_occ_map,  
    prototypes_src_imgs_masked, 
    prototypes_overlayed_imgs, 
    occ_map_min,
    occ_map_max,
    proto_id,
    fn,
    pred,
    gt,
    fig_path=None,
    imshow_interp_method="none",
):
    """
    Plots one frame of the prototype plot using upsampled occ_map
    Ho x Wo x 3 un-normalized [0,1] image
    Ho x Wo upsampled occurrence map
    Ho x Wo [0,1] rescaled occurrence map (normalized with respect to 3D volume if video
    """
    fig, axs = plt.subplots(1, 4, figsize=(12, 5))

    images = {
        "base": unnorm_img,  # base image
        "masked": prototypes_src_imgs_masked,  # image with [0,1] mask of occurrence map
        "overlay": prototypes_overlayed_imgs,
    }  # image with colormap overlay of occurrence map

    for i, (key, v) in enumerate(images.items()):
        axs[i].imshow(v, interpolation=imshow_interp_method)
        axs[i].title.set_text(key)
    # 4. raw occurrence map (non-rescaled)
    i += 1
    im = axs[i].imshow(
        upsampled_occ_map,
        interpolation=imshow_interp_method,
        vmin=occ_map_min,
        vmax=occ_map_max,
    )
    axs[i].title.set_text("mask")
    fig.colorbar(im, ax=axs[i], shrink=0.75)

    # add super title for the figure
    fig.suptitle(
        f"p_{proto_id:02d} | gt={gt} | {fn}\n"
        f"img_pred = {[f'{pred[i]:.2f}' for i in range(pred.shape[0])]}",
        fontsize=12,
    )
    # some visual configs for the figure
    fig.tight_layout()
    if fig_path is None:
        frame = mplfig_to_npimage(fig)  
        plt.close()
        return frame
    else:
        plt.savefig(fig_path)
        plt.close()

def prototype_plot(
    unnorm_img,
    upsampled_occurrence_map,
    prototypes_src_imgs_masked,
    prototypes_overlayed_imgs,
    proto_id,
    fn,
    pred,
    gt,
    proto_dir,
    interp="none",
):
    """
    Plot a visualization pertaining to one prototype and its associated image.

    Parameters
    ----------
    unnorm_img : Ho x Wo x 3 (or To x Ho x Wo x 3 for videos) ndarray
        unnormalized image (not in [0, 1]) where Ho and Wo denote original size
    upsampled_occurrence_map : Ho x Wo ndarray for images (or T x Ho x Wo for videos)
        occurence mask denoting model-predicted occurrence, upsampled to original size
    prototypes_src_imgs_masked: Ho x Wo x 3 (or To x Ho x Wo x 3 for videos) ndarray
        unnormalized image (not in [0, 1]) with the occurance map applied on it as a mask
    prototypes_overlayed_imgs: Ho x Wo x 3 (or To x Ho x Wo x 3 for videos) ndarray
        unnormalized image (not in [0, 1]) with the occurance map applied on it as a heatmap overlay
    proto_id : integer
        ID number of the prototype
    fn : string
        filename of the image
    pred : K ndarray
        array indicating logits or confidences of model
    gt: int
        integer representing ground truth class
    proto_dir : os.path
        path to the save directory of prototypes
    interp: string, optional
        interpolation method for the display purpose only. The default is 'none'.
    """
    D = len(unnorm_img.shape)
    if D == 3:
        # plot and save the image
        fig_path = os.path.join(proto_dir, f"{proto_id:02d}_{fn}.png")
        prototype_plot_frame(
            unnorm_img, 
            upsampled_occurrence_map,  
            prototypes_src_imgs_masked,  
            prototypes_overlayed_imgs,
            np.amin(upsampled_occurrence_map),
            np.amax(upsampled_occurrence_map) + 1e-7,
            proto_id,
            fn,
            pred,
            gt,
            fig_path=fig_path,
            imshow_interp_method=interp,
        )
    elif D == 4:
        # plot each frame, then, when all images have been plotted create a video and delete the frames
        frames = []
        for t in range(unnorm_img.shape[0]):
            frames.append(prototype_plot_frame(
                unnorm_img[t],  
                upsampled_occurrence_map[t],  
                prototypes_src_imgs_masked[t],  
                prototypes_overlayed_imgs[t],  
                np.amin(upsampled_occurrence_map),
                np.amax(upsampled_occurrence_map) + 1e-7,
                proto_id,
                fn,
                pred,
                gt,
                fig_path=None,
                imshow_interp_method=interp,
            ))
        frames = np.asarray(frames)  
        format = "gif"
        saveVideo(frames, save_path=proto_dir, filename=f"{proto_id:02d}_{fn}", format=format, fps=10, overwrite=True)
       

def push_prototypes(
    dataloader, 
    model,  # pytorch network with feature encoder and prototype vectors
    device,
    class_specific=False,  # enable pushing protos from only the alotted class
    abstain_class=True,  # indicates K+1-th class is of the "abstain" type
    preprocess_input_function=None,  # normalize if needed 
    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved in this dir
    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
    log=print,
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    proto_bound_boxes_filename_prefix=None,
    replace_prototypes=True,
    delta=5,
):
    """
    Search the training set for image patches that are semantically closest to
    each learned prototype, then updates the prototypes to those image patches.

    To do this, it computes the image patch embeddings (IPBs) and saves those
    closest to the prototypes. It also saves the prototype-to-IPB distances and
    predicted occurrence maps.

    If abstain_class==True, it assumes num_classes actually equals to K+1, where
    K is the number of real classes and 1 is the extra "abstain" class for
    uncertainty estimation.
    """

    model.eval()
    log(f"############## push at epoch {epoch_number} #################")

    start = time.time()

    # creating the folder (with epoch number) to save the prototypes' info and visualizations
    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, "epoch-" + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    # find the number of prototypes, and number of classes for this push
    prototype_shape = model.prototype_shape  # shape (P, D, (1), 1, 1)
    P = model.num_prototypes
    proto_class_ID = model.proto_classes.detach().clone() # initialised as same intervals
    protonames = list(range(P))

    # TODO: delete these and adjust accordingly
    proto_class_identity = np.argmax(model.prototype_class_identity.cpu().numpy(), axis=1)  # shape (P)
    proto_class_specific = np.full(P, class_specific)
    num_classes = model.num_classes
    if abstain_class:
        K = num_classes - 1
        assert K >= 2, "Abstention-push must have >= 2 classes not including abstain"
        # for the uncertainty prototypes, class_specific is False
        # for now assume that each class (inc. unc.) has P_per_class == P/num_classes
        P_per_class = P // num_classes
        proto_class_specific[K * P_per_class : P] = False
    else:
        K = num_classes

    # keep track of the input embedding closest to each prototype, GLOBAL MINIMUM
    proto_dist_ = np.full(P, np.inf)  # saves the distances to prototypes (distance = 1-CosineSimilarities). shape (P)
    # save some information dynamically for each prototype
    # which are updated whenever a closer match to prototype is found
    occurrence_map_ = [None for _ in range(P)] 
    # saves the input to prototypical layer (conv feature * occurrence map), shape (P, D)
    protoL_input_ = [None for _ in range(P)]
    # saves the input images with embeddings closest to each prototype. shape (P, 3, (To), Ho, Wo)
    image_ = [None for _ in range(P)]
    # saves the gt label. shape (P)
    gt_ = [None for _ in range(P)]
    # saves the prediction logits of cases seen. shape (P, K)
    pred_ = [None for _ in range(P)]
    # saves the filenames of cases closest to each prototype. shape (P)
    filename_ = [None for _ in range(P)]
    ed_frames_ = [None for _ in range(P)]
    es_frames_ = [None for _ in range(P)]

    data_iter = iter(dataloader)
    iterator = tqdm(range(len(dataloader)), dynamic_ncols=True)
    for push_iter in iterator:
        data_sample = next(data_iter)

        x = data_sample["cine"]  # shape (B, 3, (To), Ho, Wo)
        if preprocess_input_function is not None:
            x = preprocess_input_function(x)

        # get the network outputs for this instance, features_extracted, 1 - similarity, occurrence_map, logits
        with torch.no_grad():
            x = x.to(device)
            (
                protoL_input_torch,
                proto_dist_torch,
                occurrence_map_torch,
                logits,
            ) = model.push_forward(x)

        # record down batch data as numpy arrays
        protoL_input = protoL_input_torch.detach().cpu().numpy()  # shape (B, P, D), embeddings, extracted feature
        proto_dist = proto_dist_torch.detach().cpu().numpy()  # shape (B, P)
        occurrence_map = occurrence_map_torch.detach().cpu().numpy()  # shape (B, P, 1, (T), H, W)
        pred = logits.detach().cpu().numpy()  # shape (B, num_classes)
        gt = data_sample["target_EF"].detach().cpu().numpy()  # shape (B)
        image = x.detach().cpu().numpy()  # shape (B, 3, (To), Ho, Wo)
        filename = data_sample["filename"]  # shape (B)
        ed_frames = data_sample["ed_frame"].detach().cpu().numpy()
        es_frames = data_sample["es_frame"].detach().cpu().numpy()

        # for each prototype, find the minimum distance and their indices
        proto_class_identity = model.proto_classes.data.cpu().numpy() 
        for j in range(P):
            proto_dist_j = proto_dist[:, j]  # (B)

            # Mask distances where the ground truth label is not within delta (5) of the prototype's class 
            valid_mask = np.abs(gt - proto_class_identity[j]) <= delta  # Boolean mask

            if not np.any(valid_mask):  # Skip if no valid samples in batch
                continue

            '''if proto_class_specific[j]: 
                # compare with only the images of the prototype's class
                proto_dist_j = np.ma.masked_array(proto_dist_j, gt != proto_class_identity[j])
                if proto_dist_j.mask.all():
                    # if none of the classes this batch are the class of interest, move on
                    continue'''
            # Apply mask to filter distances
            proto_dist_j_masked = np.where(valid_mask, proto_dist_j, np.inf)  # Set invalid distances to infinity

            # get min distance to prototype j in current batch
            proto_dist_j_min = np.amin(proto_dist_j_masked)  # scalar

            # if the distance this batch is smaller than prev.best, save it
            if proto_dist_j_min <= proto_dist_[j]: # closes in batch smaller than global closest.
                a = np.argmin(proto_dist_j_masked) # get index of min distance
                # save the information of the min to global minimum.
                proto_dist_[j] = proto_dist_j_min
                protoL_input_[j] = protoL_input[a, j] # shape (D), latent patch, latent embedding of the prototype
                occurrence_map_[j] = occurrence_map[a, j]
                pred_[j] = pred[a]
                image_[j] = image[a]
                gt_[j] = gt[a]
                filename_[j] = filename[a]
                ed_frames_[j] = ed_frames[a]
                es_frames_[j] = es_frames[a]


    prototypes_similarity_to_src_ROIs = 1 - np.array(proto_dist_)  # invert distance to similarity, this is global_mindist from protoEF shape (P) 1 - cosine similarity
    prototypes_occurrence_maps = np.array(occurrence_map_)  # shape (P, 1, (T), H, W)
    prototypes_embeddings = np.array(protoL_input_)  # shape (P, D) protoL_input_ is the input to the prototypical layer
    prototypes_src_imgs = np.array(image_)  # shape (P, 3, (To), Ho, Wo)
    prototypes_gts = np.array(gt_)  # shape (P)
    prototypes_preds = np.array(pred_)  # shape (P, K)
    prototypes_filenames = np.array(filename_)  # shape (P)
    prototypes_ed_frames = np.array(ed_frames_)
    prototypes_es_frames = np.array(es_frames_)


    # save the prototype information in a pickle file
    prototype_data_dict = {
        "prototypes_filenames": prototypes_filenames,
        "prototypes_src_imgs": prototypes_src_imgs,
        "prototypes_gts": prototypes_gts,
        "prototypes_preds": prototypes_preds,
        "prototypes_occurrence_maps": prototypes_occurrence_maps,
        "prototypes_embeddings": prototypes_embeddings,
        "prototypes_similarity_to_src_ROIs": prototypes_similarity_to_src_ROIs,
        "prototypes_ed_frames": prototypes_ed_frames,
        "prototypes_es_frames": prototypes_es_frames,
       }
    save_pickle(prototype_data_dict, f"{proto_epoch_dir}/prototypes_info.pickle")

    #########################################
    # perform visualization for each prototype
    log("\tVisualizing prototypes ...")
    #########################################
    ##### Process the prototype info ########
    # get the source image/video
    from src.utils.explainability_utils import get_src, get_normalized_upsample_occurence_maps, get_heatmap
    prototypes_src_imgs, upsampler = get_src(prototypes_src_imgs) 

    # resize, upsample, and normalize the occurrence map.  
    rescaled_occurrence_maps, upsampled_occurrence_maps = get_normalized_upsample_occurence_maps(
        prototypes_occurrence_maps, upsampler)  

    # # prototype src image masked with normalized occurrence map
    mask = np.expand_dims(rescaled_occurrence_maps, axis=-1)
    prototypes_src_imgs_masked = prototypes_src_imgs * mask 

    # prototype src image with normalized occurrence map overlay
    prots_heatmaps = get_heatmap(rescaled_occurrence_maps)  
    prototypes_overlayed_imgs = np.clip(prototypes_src_imgs + 0.3 * prots_heatmaps, 0, 1)  

    for j in range(P):
        if image_[j] is not None:
            prototype_plot(unnorm_img=prototypes_src_imgs[j], upsampled_occurrence_map=upsampled_occurrence_maps[j],
                           prototypes_src_imgs_masked=prototypes_src_imgs_masked[j],
                           prototypes_overlayed_imgs=prototypes_overlayed_imgs[j], proto_id=j, fn=filename_[j],
                           pred=pred_[j], gt=gt_[j], proto_dir=proto_epoch_dir, interp="bilinear")
    

    if replace_prototypes:
        protoL_input_ = np.array(protoL_input_)
        log("\tExecuting push ...")
        prototype_update = np.reshape(protoL_input_, tuple(prototype_shape))
        model.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).to(device))
        # replace model.proto_classes with the updated proto_class_ID
        model.proto_classes.data.copy_(torch.tensor(prototypes_gts, dtype=torch.float32).to(device))
        print(model.proto_classes)
        log("\tPrototypes updated.")

    end = time.time()
    log("\tpush time: \t{0}".format(end - start))
