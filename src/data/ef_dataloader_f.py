
# %%
import math
import os
import sys
from os.path import join
from random import randint, random
import warnings
import logging, pickle

import numpy as np
import pandas as pd
from scipy.io import loadmat
from skimage.transform import resize
from pathlib import Path
import cv2
import skimage
import torch
import collections
from torch.nn import Upsample
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, RandomHorizontalFlip, GaussianBlur

from torchvision.transforms._transforms_video import RandomResizedCropVideo
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.video_transforms import RandomRotateVideo
from random import uniform
from random import seed
from collections import Counter

# only for debugging
import matplotlib.pyplot as plt

# filter out pytorch user warnings for upsampling behaviour
warnings.filterwarnings("ignore", category=UserWarning)

################################################################################################
################### uncomment for Hooman/Extract_features_for_Armins_project ###################
EXTRACT_FEATURES_FOR_STUDYEPISODES = False
################################################################################################

# %%
def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)

def get_ef_dataloader(config, split, mode):
    """
    Uses the configuration dictionary to instantiate AS dataloaders

    Parameters
    ----------
    config : data configuration in dictionary format with keys in below
    split : string, 'train'/'val'/'test' for which section to obtain
    mode : string, 'train'/'val'/'push'/'test' for setting augmentation/metadata ops

    Returns
    -------
    Training, validation or test dataloader with data arranged according to
    pre-determined splits
    Note: config dictionary should have the following:
        "name": "ef_base", 
        "dataset_root": DATA_ROOT, 
        "data_info_file": CSV_NAME, 
        "sample_size": None,
        "sampler": "random",
        "label_scheme_name": "all",
        "augmentation": True,
        "img_size": 112,
        "frames": 32,
        "batch_size": 2,
        "num_workers": 0,
        
    """

    num_workers = config["num_workers"]
    bsize = config["batch_size"]
    dset = ProtoEfDataset(
        **config,
        split=split)

    if mode == "TRAIN":
        if config["sampler"] == "EF":
            logging.info("Using EF sampler")
            sampler_ef = dset.class_sampler_EF()
            loader = DataLoader(dset, batch_size=bsize, sampler=sampler_ef, num_workers=num_workers)
        else:  # random sampling
            logging.info("Using random sampler")
            loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=num_workers)
    else:
        loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=num_workers) # For now, test and val have a b_size of 1
    return loader

class ProtoEfDataset(Dataset):
    """
    Dataset class for EchoNet-Dynamic dataset
    The dataset can be found at: https://echonet.github.io/dynamic/

    Attributes
    ----------
    frames: int, maximum number of frames per video=
    train_idx: numpy.ndarray, list indices indicating CSV rows belonging to the train set
    val_idx: numpy.ndarray, list indices indicating CSV rows belonging to the validation set
    test_idx: numpy.ndarray, list indices indicating CSV rows belonging to the test set
    es_frames: torch.tensor, tensor containing ES frame indices
    ed_frames: torch.tensor, tensor containing ED frame indices
    patient_data_dirs: list['str'], list containing the directory to each sample video
    sample_weights: np.ndarray, numpy array containing weights for each sample based on EF frequency
    sample_intervals: np.ndarray, the bins for EF used to find sample_weights
    num_samples: int, number of samples in the dataset
    trans: torchvision.transforms.Compose, torchvision transformation for each data sample
    augmentation: bool, indicates whether LV zoom-in augmentation is used during training
    upsample: torch.nn.Upsample, performs 2D upsampling (needed for zomm augmentation)

    taken from echognn
    """
    def __init__(self,
                 dataset_path: str,
                 csv_file: str,
                 frames: int = 32, 
                 max_clips: int = 1, 
                 test_clips= "fixed",
                 img_size: int = 112,
                 mean: float = 0.1289,
                 std: float = 0.1911,
                 label_strings: str = 'EF_Category',
                 label_scheme_name: str = "ef_2class",
                 split = "TRAIN",
                 label_divs: float = 1.0,
                 augmentation: bool = False,
                 transform_rotate_degrees: float = 10.0,
                 use_seg_labels=False,
                 patch_width: int = 7,
                 sample_period: int = 1,
                 test_type = "3clip",
                 **kwargs):
        """
        :param dataset_path: str, path to dataset directory
        :param frames: int, maximum number of frames per video
        :param img_size: int, the size to reshape frames to
        :param mean: float, mean used in data standardization
        :param std: float, std used in data standardization
        :param label_strings: list, string indicating which column in dataset CSV is for the labels
        :param label_divs: list, value to divide labels by (for example, EF can be normalized between 0-1)
        :param augmentation: bool, indicates whether data augmentation is used during training
        """

        super().__init__()

        # Default classification classes
        if label_strings is None:
            label_strings = ['EF', 'EF_Category', "EF_binary"] # EF category has 4 bins in the range of [0−30],(30,40],(40,55],(55,100]
            label_divs = [1, 1, 1] # [100, 1, 1]
        else:
            raise ValueError("Invalid label_strings")
        # CSV file containing file names and labels
        filelist_df = pd.read_csv(os.path.join(dataset_path, csv_file))

        # Load the start indexes for the test clips
        start_idx_tests_path = os.path.join(dataset_path, "test_start_indexes.pkl")
        #open pkl file as a dictionary
        with open(start_idx_tests_path, 'rb') as f:
            self.start_idx_tests = pickle.load(f)

        # Take train/test/val
        if split in ("TRAIN", "VAL", "TEST"):
            filelist_df = filelist_df[filelist_df["Split"] == split]
        self.split = split
        self.use_seg_labels = use_seg_labels
        # Extract ES and ED frame indices
        es_frames = list(filelist_df['ESFrame'])
        self.es_frames = [None if np.isnan(es_frame) else int(es_frame) for es_frame in es_frames]
        ed_frames = list(filelist_df['EDFrame'])
        self.ed_frames = [None if np.isnan(ed_frame) else int(ed_frame) for ed_frame in ed_frames]

        # Extract video file names
        self.filenames = np.array(filelist_df['FileName'].tolist())

        # All file paths
        self.patient_data_dirs = [os.path.join(dataset_path,
                                               'Videos',
                                               file_name + '.avi')
                                  for file_name
                                  in self.filenames.tolist()]

        # Get the normalised EF labels, and EF Category labels
        self.labels = list()
        for patient, _ in enumerate(self.patient_data_dirs):
            for i, (label_str, label_div) in enumerate(zip(label_strings, label_divs)):
                if i == 0:
                    self.labels.append({label_str: filelist_df[label_str].tolist()[patient] / label_div})
                else:
                    self.labels[-1].update({label_str: filelist_df[label_str].tolist()[patient] / label_div})        
        # Extract LV segmentation masks
        if self.use_seg_labels:
            self.filenames, self.labels = self._extract_lv_trace(
                dataset_path, self.filenames, self.labels
            )


        # Extract the number of available data samples
        self.num_samples = len(self.patient_data_dirs)

        # Transform operation operation
        self.trans1 = Compose([ToTensor(),
                               Resize((img_size, img_size)),])

        self.trans2 = Normalize((mean), (std))
        self.mean = mean
        self.std = std

        # Interpolation needed if augmentation is required
        if augmentation:
            self.aug = Compose(
                [
                    RandomRotateVideo(degrees=transform_rotate_degrees), 
                ]
            )
        self.blurrer = GaussianBlur(kernel_size=(3, 3), sigma=(2, 2))
        self.dialation_kernel = 5

        self.max_frames = frames #Maximum number of frames per video clip
        self.max_clips = max_clips
        self.augmentation = augmentation
        self.label_strings = label_strings
        self.label_scheme_name = label_scheme_name
        self.patch_width = patch_width
        self.sample_period = sample_period
        self.test_type = test_type
        #self.label_keys = label_keys

    def __getitem__(self, idx: int) -> dict:
        """
        Fetch a sample from the dataset

        :param idx: int, index to extract from the dataset
        :return: dict of cine video and metadata including the target labels.
        """
        # List of labels for the sample
        sample_labels = dict.fromkeys(self.label_strings, None) # instantiate a dictionary with the keys of label_strings and values of None. It is populated with the ground truth labels of the sample
        # Get the labels
        for label_str in sample_labels.keys():
            sample_labels[label_str] = self.labels[idx][label_str]

        ########### 1- load the original video ###########
        cine_vid = self._loadvideo(self.patient_data_dirs[idx])
        orig_size = cine_vid.shape[0]

        ########### 2- Sample frames using self.period ###########
        if self.sample_period > 1:
            # Sample every self.period-th frame (this is done after loading the entire video)
            cine_vid = cine_vid[:, :, ::self.sample_period]  # Slicing to select frames at self.period intervals

        ########### 3- Transform original video ###########
        cine_vid = self.trans1(cine_vid) 

        ########## 4- Extract proper video frames by either padding or clipping ###########
        # Mask indicating which frames are padding
        mask = torch.ones((1, self.max_frames), dtype=torch.bool)
        
        ########## 5- Perform Augmentation during training ###########
        
        (
            cine_vid,
            mask,
            lv_mask,
            ed_frame,
            ed_valid,
            es_frame,
            es_valid,
        ) = self._pad_vid(cine_vid, mask, idx, orig_size)

        if (self.split == "TRAIN") and self.augmentation: # must aug first so that the mask is also augmented
            if self.use_seg_labels:
                # Temporarily replicate lv_mask across the temporal dimension of cine_vid for augmentation
                lv_mask_replicated = lv_mask.unsqueeze(0).expand(cine_vid.size(0), -1, -1)

                # Combine video and mask for joint augmentation
                combined = torch.cat((cine_vid.unsqueeze(1), lv_mask_replicated.unsqueeze(1)), dim=1)

                # Apply augmentations
                combined = self.aug(combined)

                # Split video and mask after augmentation
                cine_vid = combined[:, 0, :, :] 
                lv_mask_augmented = combined[:, 1, :, :] 

                # Reduce the augmented lv_mask back to a single 2D mask using aggregation (e.g., mean or max)
                lv_mask = lv_mask_augmented.mean(dim=0)
            else:
                cine_vid = self.aug(cine_vid.unsqueeze(0))[0]

        # transform into 3D and unsqueeze
        cine_vid = self.trans2(cine_vid)
        if self.split == "TRAIN" or self.split == "VAL":
            cine_vid = cine_vid.expand(3, -1, -1, -1) 
        else: 
            cine_vid = cine_vid.unsqueeze(1).expand(-1, 3, -1, -1, -1) # shape (max_clips, 3, max_frames, H, W)
        ########## 6- Interpolate (smooth) the segmentation map ###########
        if self.use_seg_labels:
            ###### 1. Downsample to the size of occurance map using bilinear interpolation
            lv_mask = F.interpolate(
                lv_mask.unsqueeze(0).unsqueeze(1),
                size=(self.patch_width,self.patch_width)
            )
            ###### 2. Dilate the mask using max-pooling)
            lv_mask = 1.0 - lv_mask  # Invert the mask
            lv_mask = F.max_pool2d(lv_mask, kernel_size=self.dialation_kernel, stride=1, padding=self.dialation_kernel // 2)
            lv_mask = 1.0 - lv_mask  # Invert the mask back

            ###### 3. Relax the boundaries using Gaussian blur
            lv_mask = self.blurrer(lv_mask)

            ###### 4. Use another Gaussian blur or average pooling to smooth the region
            lv_mask = lv_mask.squeeze(0)

        if self.label_scheme_name == "ef_2class":
            sample_label_category = int(sample_labels[ self.label_strings[2]])
        elif self.label_scheme_name == "ef_4classclinical":
            sample_label_category = int(sample_labels[ self.label_strings[1]])
        else:
            sample_label_category = 0

        ret = {
            "data_dir": self.patient_data_dirs[idx],
            "filename": self.filenames[idx],
            "cine": cine_vid,
            "lv_mask": lv_mask,
            "mask": mask,
            "target_EF": torch.tensor(sample_labels[self.label_strings[0]], dtype=torch.float32),
            "target_EF_Category": sample_label_category,
            "ed_frame": torch.tensor(ed_frame),
            "es_frame": torch.tensor(es_frame),
            "ed_valid": torch.tensor(ed_valid),
            "es_valid": torch.tensor(es_valid),
            
        }

        return ret

    def __len__(self) -> int:
        """
        Returns number of available samples

        :return: Number of graphs
        """

        return self.num_samples

    @staticmethod
    def _loadvideo(filename: str):
        """
        Video loader code from https://github.com/echonet/dynamic/tree/master/echonet with some modifications

        :param filename: str, path to video to load
        :return: numpy array of dimension H*W*T
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_height, frame_width, frame_count), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            v[:, :, count] = frame
        
        capture.release()

        return v
    
    def _extract_lv_trace(self, dataset_path, file_names, labels):
        '''A helper function to extract the LV trace from the dataset which will be used later for segmentation
        dataset_path: path to the dataset
        file_names: list of filenames
        labels: list of labels: EF and EF_Category in this case

        returns:
        file_names: list of filenames that have more than 2 frames. Note: in echonet dynamics train, all of the data points have more than 2 frames.
        labels: list of labels of the files that have more than 2 frames

        Mutates:
        self.total_frames: a dictionary with keys of filenames and values of list of frames in the original video
        self.trace: a dictionary with keys of filenames and values of dictionary with keys of frames (like total_frames) and values of list of coordinates of the LV trace in that frame
        self.ed_frames: list of ED frames in the files that have more than 2 frames. In this case should be the same as the original ED frames
        self.es_frames: list of ES frames in the files that have more than 2 frames. In this case should be the same as the original ES frames
        '''
        self.total_frames = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        with open(os.path.join(dataset_path, "VolumeTracings.csv")) as f:
            header = f.readline().strip().split(",")
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            for line in f:
                filename, x1, y1, x2, y2, frame = line.strip().split(",")
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                frame = int(frame)
                if frame not in self.trace[filename]: # if the frame is not already in the total_frames
                    self.total_frames[filename].append(frame)
                self.trace[filename][frame].append((x1, y1, x2, y2))
        for filename in self.total_frames:
            for frame in self.total_frames[filename]:
                self.trace[filename][frame] = np.array(self.trace[filename][frame])

        keep = [len(self.total_frames[f + ".avi"]) >= 2 for f in file_names] # + ".avi"
        file_names = [f for (f, k) in zip(file_names, keep) if k]
        labels = [f for (f, k) in zip(labels, keep) if k]
        self.ed_frames = [f for (f, k) in zip(self.ed_frames, keep) if k] 
        self.es_frames = [f for (f, k) in zip(self.es_frames, keep) if k]

        return file_names, labels
    
    def _pad_vid(self, vid, mask, patient_idx, orig_size=None):
        '''A helper function to pad or clips the video to the max frames and return the ED and ES frame indices in the clip together the segmentation mask of LV

        vid: tensor of shape (T, H, W). The video is loaded, trasformed to tensor and resized and normalised: in GEMTrans: ToTensor, Resize, Normalize. It has one channel
        mask: tensor of shape (T,)
        patient_idx: index of the patient in the dataset
        orig_size: original size of the video frames

        returns:
        vid: tensor of shape (max_frames, H, W) if single clip of max_frame else (#clips, max_frames, H, W). The video is padded with 0's if the number of frames is less than max frames. Otherwise, for training it is randomly clipped to max frames. For testing and validation, it is divided into clips of max frames.
        mask: tensor of shape (max_frames,). The mask is padded with False if the number of frames is less than max frames.
        lv_mask_collated: tensor of shape (H, W). The LV mask for the ED and ES frames collated (union) together.
        ed_frame_idx: index of the ED frame in the clip
        ed_valid: boolean indicating if the ED frame is in the clip
        es_frame_idx: index of the ES frame in the clip
        es_valid: boolean indicating if the ES frame is in the clip
        '''

        file_name = os.path.basename(self.patient_data_dirs[patient_idx])

        # Combine the LV mask for ED and ES frames
        lv_mask_collated = torch.zeros(1)
        if self.use_seg_labels:
            for i in range(2):
                t = self.trace[file_name][self.total_frames[file_name][i]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))
                r, c = skimage.draw.polygon(
                    np.rint(y).astype(np.int32),
                    np.rint(x).astype(np.int32),
                    (orig_size, orig_size), 
                )
                # this is the opposite of MAssoud's since I want the inside 0 outsize 1
                lv_mask = np.ones((orig_size, orig_size), bool) 
                lv_mask[r, c] = 0 # foreground is 0, background is 1
                lv_mask_collated = (
                    lv_mask if i == 0 else np.bitwise_and(lv_mask_collated, lv_mask)
                )
            lv_mask_collated = torch.from_numpy(lv_mask_collated.astype(np.float32))
        # If the number of frames is less than max frames, pad with 0's
        if vid.shape[0] <= self.max_frames:
            mask[0, vid.shape[0] :] = False
            vid = torch.cat(
                (
                    vid,
                    torch.zeros(
                        self.max_frames - vid.shape[0], vid.shape[1], vid.shape[2]
                    ),
                ),
                dim=0,
            )

            ed_frame_idx, ed_valid, es_frame_idx, es_valid = self._frame_idx_in_clip(
                patient_idx, np.arange(self.max_frames)
            )

            #if not self.train:
            if self.split == "VAL" or self.split == "TEST":
                #mask = mask.unsqueeze(0) # shape (1, max_frames)
                vid = vid.unsqueeze(0) # shape (1, max_frames, H, W)
        else:
            #if self.train:
            if self.split == "TRAIN" or self.split == "VAL":
                starting_idx = np.random.randint(0, vid.shape[0] - self.max_frames)

                (
                    ed_frame_idx,
                    ed_valid,
                    es_frame_idx,
                    es_valid,
                ) = self._frame_idx_in_clip(
                    patient_idx, np.arange(starting_idx, starting_idx + self.max_frames)
                )

                vid = vid[starting_idx : starting_idx + self.max_frames]
            else: # TEST with the whole video
                if self.test_type == 'all':
                    # During validation and testing use all available clips
                    ed_valid = []
                    ed_frame_idx = []
                    es_valid = []
                    es_frame_idx = []
                    num_clips = min(
                        math.ceil(vid.shape[0] / self.max_frames), self.max_clips
                    )

                    curated_clips = None
                    for clip_idx in range(num_clips - 1):
                        curated_clips = (
                            vid[0 : self.max_frames].unsqueeze(0) # shape (1, max_frames, H, W)
                            if curated_clips is None
                                
                            else torch.cat(
                                (
                                    curated_clips,
                                    vid[
                                        self.max_frames
                                        * clip_idx : self.max_frames
                                        * (clip_idx + 1)
                                    ].unsqueeze(0), # shape (1, max_frames, H, W)
                                ),
                                dim=0,
                            )
                        )
                        # get the indx and details of existance of ed and es frames in the given clip.
                        (
                            clip_ed_idx,
                            clip_ed_valid,
                            clip_es_idx,
                            clip_es_valid,
                        ) = self._frame_idx_in_clip(
                            patient_idx,
                            np.arange(
                                self.max_frames * clip_idx, self.max_frames * (clip_idx + 1)
                            ),
                        )

                        ed_valid.append(clip_ed_valid)
                        ed_frame_idx.append(clip_ed_idx)
                        es_valid.append(clip_es_valid)
                        es_frame_idx.append(clip_es_idx)

                    # The last clip is allowed to overlap with the previous one
                    curated_clips = (
                        vid[-self.max_frames :].unsqueeze(0)
                        if curated_clips is None
                        else torch.cat(
                            (curated_clips, vid[-self.max_frames :].unsqueeze(0)), dim=0
                        )
                    )

                    (
                        clip_ed_idx,
                        clip_ed_valid,
                        clip_es_idx,
                        clip_es_valid,
                    ) = self._frame_idx_in_clip(
                        patient_idx,
                        np.arange(vid.shape[0] - self.max_frames, vid.shape[0]),
                    )

                    ed_valid.append(clip_ed_valid)
                    ed_frame_idx.append(clip_ed_idx)
                    es_valid.append(clip_es_valid)
                    es_frame_idx.append(clip_es_idx)

                    vid = curated_clips # shape (num_clips, max_frames, H, W)
                    mask = torch.cat([mask.unsqueeze(0)] * num_clips, dim=0) # shape (num_clips, max_frames) with value true for the frames in the clip and false for the padded frames

                elif self.test_type == "3clip": # TEST with 3 clips
                    # During validation and testing use all available clips
                    ed_valid = []
                    ed_frame_idx = []
                    es_valid = []
                    es_frame_idx = []
                    
                    curated_clips = None
                    start_indices = self.start_idx_tests[file_name]

                    num_clips = len(start_indices)
                    total_frames = vid.shape[0]  # Total number of frames in the video

                    adjusted_start_indices = []
                    for i, start_idx in enumerate(start_indices):
                        if start_idx + self.max_frames > total_frames:  # If clip would exceed total frames
                            start_idx = max(0, total_frames - self.max_frames)  # Shift back to ensure full-length
                        adjusted_start_indices.append(start_idx)

                    for i, start_idx in enumerate(adjusted_start_indices):
                        clip_idx = i  # Using loop index instead of np.where for simplicity
                        
                        curated_clips = (
                            vid[start_idx : (start_idx+ self.max_frames)].unsqueeze(0) # shape (1, max_frames, H, W)
                            if curated_clips is None
                                
                            else torch.cat(
                                (
                                    curated_clips,
                                    vid[start_idx : (start_idx+ self.max_frames)].unsqueeze(0), # shape (1, max_frames, H, W)
                                ),
                                dim=0,
                            )
                        )
                        # get the indx and details of existance of ed and es frames in the given clip.
                        (
                            clip_ed_idx,
                            clip_ed_valid,
                            clip_es_idx,
                            clip_es_valid,
                        ) = self._frame_idx_in_clip(
                            patient_idx,
                            np.arange(
                                start_idx, self.max_frames * (clip_idx + 1)
                            ),
                        )

                        ed_valid.append(clip_ed_valid)
                        ed_frame_idx.append(clip_ed_idx)
                        es_valid.append(clip_es_valid)
                        es_frame_idx.append(clip_es_idx)

                    vid = curated_clips # shape (num_clips, max_frames, H, W)
                    mask = torch.cat([mask.unsqueeze(0)] * 3, dim=0) # shape (num_clips, max_frames) with value true for the frames in the clip and false for the padded frames
                
                elif self.test_type == "single":
                    # Ensure we have enough frames to sample from
                    if vid.shape[0] > self.max_frames:
                        starting_idx = np.random.randint(0, vid.shape[0] - self.max_frames + 1)
                    else:
                        starting_idx = 0  # If not enough frames, start from the beginning

                    # Extract the randomly selected clip
                    vid = vid[starting_idx : starting_idx + self.max_frames]

                    # Get frame index details for ED and ES
                    (
                        clip_ed_idx,
                        clip_ed_valid,
                        clip_es_idx,
                        clip_es_valid,
                    ) = self._frame_idx_in_clip(
                        patient_idx,
                        np.arange(starting_idx, starting_idx + self.max_frames),
                    )

                    # Convert indices and mask accordingly
                    ed_valid = [clip_ed_valid]
                    ed_frame_idx = [clip_ed_idx]
                    es_valid = [clip_es_valid]
                    es_frame_idx = [clip_es_idx]

                    # Update the vid and mask to only contain the selected clip
                    vid = vid.unsqueeze(0)

                elif self.test_type == "2clip":
                    # Ensure we have enough frames to sample two clips
                    if vid.shape[0] >= 2 * self.max_frames:
                        # Select the first clip from the beginning half and the second from the latter half
                        first_start = np.random.randint(0, (vid.shape[0] // 2) - self.max_frames + 1)
                        second_start = np.random.randint(vid.shape[0] // 2, vid.shape[0] - self.max_frames + 1)
                    elif vid.shape[0] > self.max_frames:
                        # Not enough for two non-overlapping clips, so select two overlapping ones
                        first_start = 0
                        second_start = vid.shape[0] - self.max_frames
                    else:
                        # If not enough frames, just duplicate the same clip
                        first_start = 0
                        second_start = 0

                    # Extract the two clips
                    first_clip = vid[first_start : first_start + self.max_frames]
                    second_clip = vid[second_start : second_start + self.max_frames]

                    # Get frame index details for ED and ES for both clips
                    (
                        first_ed_idx,
                        first_ed_valid,
                        first_es_idx,
                        first_es_valid,
                    ) = self._frame_idx_in_clip(
                        patient_idx, np.arange(first_start, first_start + self.max_frames)
                    )

                    (
                        second_ed_idx,
                        second_ed_valid,
                        second_es_idx,
                        second_es_valid,
                    ) = self._frame_idx_in_clip(
                        patient_idx, np.arange(second_start, second_start + self.max_frames)
                    )

                    # Convert indices and mask accordingly
                    ed_valid = [first_ed_valid, second_ed_valid]
                    ed_frame_idx = [first_ed_idx, second_ed_idx]
                    es_valid = [first_es_valid, second_es_valid]
                    es_frame_idx = [first_es_idx, second_es_idx]

                    # Stack both clips along a new batch dimension
                    vid = torch.stack([first_clip, second_clip], dim=0)
                    mask = torch.cat([mask.unsqueeze(0)] * 2, dim=0)

        return (
            vid,
            mask,
            lv_mask_collated,
            ed_frame_idx,
            ed_valid,
            es_frame_idx,
            es_valid,
        )

    def _frame_idx_in_clip(self, data_idx, clip_idx):
        '''A helper function to identify the location of the ED and ES frames in the clip
        data_idx: index of the data sample
        clip_idx: indices of the frames of the original video in the clip: a list of frame indices.
        note: clip indices are reference to the original video frames.

        returns:
        ed_frame: index of the ED frame in the clip
        ed_valid: boolean indicating if the ED frame is in the clip
        es_frame: index of the ES frame in the clip
        es_valid: boolean indicating if the ES frame is in the clip
        '''
        ed_frame, ed_valid, es_frame, es_valid = 0, False, 0, False
        if self.ed_frames[data_idx] in clip_idx: # if ed frame is among the frames in the clip
            ed_frame = np.where(clip_idx == self.ed_frames[data_idx])[0].item() # frame index in the clip
            ed_valid = True

        if self.es_frames[data_idx] in clip_idx:
            es_frame = np.where(clip_idx == self.es_frames[data_idx])[0].item()
            es_valid = True

        return ed_frame, ed_valid, es_frame, es_valid
        
    def class_sampler_EF(self):
        """
        returns samplers (WeightedRandomSamplers) based on frequency of the ef class occurring
        """
        
        if self.label_scheme_name == "ef_2class":
            weight_ef = [1.73057622, 6.00660661]
            ef_labels = [int(label['EF_binary']) for label in self.labels]
        
        elif self.label_scheme_name == "ef_4classclinical":
            weight_ef = [21.78867102, 20.62061856,  7.53654861,  1.93330756] # this is only for 4 classes
            ef_labels = [int(label['EF_Category']) for label in self.labels]

        else:
            # General case: dynamically compute labels for n-class classification
            n = int(self.label_scheme_name.split("_")[1].replace("class", ""))  # Extract n from "ef_nclass"

            # Extract EF values
            ef_values = np.array([label["EF"] for label in self.labels])

            # Compute bin edges for n classes (evenly spaced between min and max EF values)
            bin_edges = np.linspace(ef_values.min(), ef_values.max(), num=n+1)

            # Assign each EF value to a bin index
            ef_labels = np.digitize(ef_values, bins=bin_edges, right=True) 

            # Compute class frequencies
            class_counts = Counter(ef_labels)
            total_samples = len(ef_labels)

            # Compute inverse frequency weights
            weight_ef = {cls: total_samples / count for cls, count in class_counts.items()}
   
        # Assign sample weights based on labels
        samples_weight_ef = [weight_ef[label] for label in ef_labels]
        sampler_ef = WeightedRandomSampler(samples_weight_ef, len(samples_weight_ef)) # len ef is len of dataset
        return sampler_ef