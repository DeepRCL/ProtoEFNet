"""
Main
-Process the yml config file
-Create an agent instance
-Run the agent
"""
from src.agents import * #extra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import sys, os
from pathlib import Path
import matplotlib.pyplot as plt
import umap
from src.utils.utils import (
    updated_config,
    dict_print,
    create_save_loc,
    set_logger,
    backup_code,
    set_seed,
    plot_embeddings
)
import wandb
import logging

if __name__ == "__main__":
    # ############# handling the bash input arguments and yaml configuration file ###############
    config = updated_config()
    #os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

    # create saving location and document config files
    create_save_loc(config)  # config['save_dir'] gets updated here!
    save_dir = config["save_dir"]

    # ############# handling the logistics of (seed), and (logging) ###############
    set_seed(config["train"]["seed"])
    set_logger(save_dir, config["log_level"], "train", config["comment"])
    #backup_code(save_dir)

    # printing the configuration
    dict_print(config)

    # ############# Wandb setup ###############
    wandb.init(
        project="EFexplanableAI",
        config=config,
        name=None if config["run_name"] == "" else config["run_name"],
        entity="yeganeh1377",  # your wandb username or team name
        mode=config["wandb_mode"],  # one of "online", "offline" or "disabled" 
        notes=config["save_dir"],  # to know where the model is saved!
    )
    # Update config based on wandb sweep selected configs

    ############# Load features Metadata and Similarities ###############
    train_features = torch.load(os.path.join(config["save_dir"], "e30_clips_TRAIN.pt"))
    test_features = torch.load(os.path.join(config["save_dir"], "e30_clips_TEST.pt"))
    val_features = torch.load(os.path.join(config["save_dir"], "e30_clips_VAL.pt"))
    
    train_metadata = pd.read_csv(os.path.join(config["save_dir"], "e30_clips_TRAIN.csv"))
    test_metadata = pd.read_csv(os.path.join(config["save_dir"], "e30_clips_TEST.csv"))
    val_metadata = pd.read_csv(os.path.join(config["save_dir"], "e30_clips_VAL.csv"))

    train_similarities = torch.load(os.path.join(config["save_dir"], "e30_similarities_TRAIN.pt")) # shape (N, P)
    test_similarities = torch.load(os.path.join(config["save_dir"], "e30_similarities_TEST.pt"))
    val_similarities = torch.load(os.path.join(config["save_dir"], "e30_similarities_VAL.pt"))
    
    prototypes = torch.load(os.path.join(config["save_dir"], "e30_prototypes.pt"))
    prototype_labels = torch.load(os.path.join(config["save_dir"], "e30_prototype_labels.pt"))

    train_labels = train_metadata["target_EF"].to_numpy()
    test_labels = test_metadata["target_EF"].to_numpy()
    val_labels = val_metadata["target_EF"].to_numpy()

    logging.info("Loaded all features and metadata \n")
    logging.info(f"Train Features:{train_features.shape} | Test Features:{test_features.shape}| Val Features:{val_features.shape} | " #TODO: double check the sequence of losses
                f"Train Similarities:{train_similarities.shape} | Test Similarities:{test_similarities.shape} | Val Similarities:{val_similarities.shape} |"
                f"Train labels:{train_labels.shape} | Test labels:{test_labels.shape} | Val labels:{val_labels.shape} |"
                f"Prototypes:{prototypes.shape} | Prototype Labels:{prototype_labels.shape}")
    
    # ############# UMAP ###############
    plot_embeddings(prototypes, prototype_labels, config["save_dir"], embed_type='PCA', dim = '2D', k=500, train_points=train_features, test_points=test_features, train_labels=train_labels, test_labels=test_labels, train_similarities=train_similarities, test_similarities=test_similarities, test_type='TEST')
    plot_embeddings(prototypes, prototype_labels, config["save_dir"], embed_type='PCA', dim = '2D', k=500, train_points=train_features, test_points=val_features, train_labels=train_labels, test_labels=val_labels, train_similarities=train_similarities, test_similarities=val_similarities, test_type='VAL')

    print("")

