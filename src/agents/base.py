"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import os
import pandas as pd
import numpy as np
import wandb
import logging

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from typing import Dict
from torchsummary import summary

from src.models import model_builder
from src.utils.utils import print_cuda_statistics
from src.data.as_dataloader_f import get_as_dataloader
from src.data.ef_dataloader_f import get_ef_dataloader
from src.data.view_dataloader_f import get_dataloader

cudnn.benchmark = True  # IF input size is same all the time, it's faster this way

DATALOADERS = {
    "view_quality": get_dataloader,
    "as_tom": get_as_dataloader,
    "ef": get_ef_dataloader,
}


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        # Load configurations
        self.config = config
        self.run_name = config["run_name"]
        self.model_config = config["model"]
        self.train_config = config["train"]
        self.data_config = config["data"]

        # #################### define models ####################
        img_size = self.data_config["img_size"]
        self.model_config.update(
            {
                "img_size": img_size,
            }
        )
        self.model = model_builder.build(self.model_config)
        self.print_model_summary()

        # ##############  set cuda flag, seed, and gpu devices #############
        self.setup_cuda()
        print(torch.cuda.memory_allocated(0))

        # ############# define dataset and dataloader ##########
        self.data_config.update(
            {
                "batch_size": self.train_config["batch_size"],
                "num_workers": self.train_config["num_workers"],
            }
        )
        self.data_loaders: Dict[str, DataLoader] = {}
        for x in ["TRAIN", "VAL", "TEST"]:
            if x in self.data_config["modes"]:
                self.data_loaders.update({
                    x: DATALOADERS[self.data_config["name"]](self.data_config, split=x, mode=x)
                })
              
        # ############# Wandb setup ###############
        if config.get("wandb_mode") != "disabled":
        
            wandb.watch(self.model, log="all")  # logs gradients and parameters
            # tensorboard
            # from torch.utils.tensorboard import SummaryWriter
            # from tensorboardX import SummaryWriter
            # self.writer = SummaryWriter(log_dir=os.path.join('tensorboard', self.config['run_name']))

            # define our custom x axis metric
            wandb.define_metric("batch_train/step")
            wandb.define_metric("batch_val/step")
            wandb.define_metric("batch_val_push/step")
            wandb.define_metric("epoch")
            # set all other metrics to use the corresponding step
            wandb.define_metric("batch_train/*", step_metric="batch_train/step")
            wandb.define_metric("batch_val/*", step_metric="batch_val/step")
            wandb.define_metric("batch_val_push/*", step_metric="batch_val_push/step")
            wandb.define_metric("epoch/*", step_metric="epoch")
            wandb.define_metric("lr", step_metric="epoch")
            # define a metric we are interested in the minimum of
            wandb.define_metric("epoch/train/loss_all", summary="min")
            wandb.define_metric("epoch/val/loss_all", summary="min")
            wandb.define_metric("epoch/val_push/loss_all", summary="min")
            # define a metric we are interested in the maximum of
            wandb.define_metric("epoch/train/f1_mean", summary="max")
            wandb.define_metric("epoch/train/accuracy", summary="max")
            wandb.define_metric("epoch/train/AUC_mean", summary="max")
            wandb.define_metric("epoch/val/f1_mean", summary="max")
            wandb.define_metric("epoch/val/accuracy", summary="max")
            wandb.define_metric("epoch/val/AUC_mean", summary="max")
            wandb.define_metric("epoch/val_push/f1_mean", summary="max")
            wandb.define_metric("epoch/val_push/accuracy", summary="max")
            wandb.define_metric("epoch/val_push/AUC_mean", summary="max")

    def get_criterion(self):  # TODO use builder in future?
        """
        creates the pytorch criterion loss function by calling the corresponding loss class
        """
        raise NotImplementedError

    def get_optimizer(self):  # TODO use builder in future?
        """
        creates the pytorch optimizer
        """
        raise NotImplementedError

    def get_lr_scheduler(self):  # TODO use builder in future?
        raise NotImplementedError

    def setup_cuda(self):
        self.cuda = torch.cuda.is_available()
        #print("is available:", self.cuda)
        if not self.cuda:
            self.device = torch.device("cpu")
            logging.info("Program will run on *****CPU*****\n")
        else:
            self.device = torch.device("cuda")
            print_cuda_statistics()
            self.model = self.model.to(self.device)
            logging.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()

    def load_checkpoint(self, file_name): 
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:
            if file_name is not None:
                checkpoint = torch.load(file_name, map_location="cpu")

                self.current_epoch = checkpoint["epoch"]
                self.current_iteration = checkpoint["iteration"]
                self.model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])

                logging.info(
                    "Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(
                        file_name, checkpoint["epoch"], checkpoint["iteration"]
                    )
                )

                # print(self.model.load_state_dict(torch.load(file_name)))
        except OSError as e:
            logging.info(f"Error {e}")
            logging.info("No checkpoint exists from '{}'. Skipping...".format(file_name))
            logging.info("**First time to train**")

    def get_state(self):
        return {
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def save_checkpoint(self, is_best=0):
        """
        Checkpoint saver
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        if not self.train_config["save"]:  # TODO review for multiGPU system
            return
        state = self.get_state()
        if (self.train_config["save_step"] is not None) and self.current_epoch % self.train_config["save_step"] == 0:
            # Save the state
            torch.save(
                state,
                os.path.join(self.config["save_dir"], f"epoch_{self.current_epoch}.pth"),
            )
        if is_best:
            torch.save(state, os.path.join(self.config["save_dir"], f"model_best.pth"))
        # save last
        torch.save(state, os.path.join(self.config["save_dir"], f"last.pth"))

    def save_best_checkpoint(self):
        if self.train_config["save"]:
            state = self.get_state()
            torch.save(state, os.path.join(self.config["save_dir"], f"best.pth"))

    def save_last_checkpoint(self):
        state = self.get_state()
        torch.save(state, os.path.join(self.config["save_dir"], f"last.pth"))

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            logging.info("You have entered CTRL+C.. Wait to finalize")

    def push(self):
        """
        pushing prototypes
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def create_pred_log_df(self, data_sample, logit, logit_names=None):
        """
        creates a pandas dataframe of predictions and labels. suitable for logging in .csv or on wandb
        :return:
        a pandas df to collect information for post processing to evaluate model's performance
        """
        pred_log_data = {
            "file_name": [item for item in data_sample["filename"]],
            "target_EF_Category": [item.int().numpy() for item in data_sample["target_EF_Category"]],
            "target_EF": [item.float().numpy() for item in data_sample["target_EF"]],
            "ed_frame": [item.int().numpy() for item in data_sample["ed_frame"]],
            "es_frame": [item.int().numpy() for item in data_sample["es_frame"]],
        }

        if logit_names == None:
            pred_log_data.update({f"preds": [value.float().numpy() for value in logit]})
            #pred_log_data.update({f"preds": value for value in logit.t()})

        else:
            pred_log_data.update({f"logit_{as_label}": value for as_label, value in zip(logit_names, logit.t())})
        return pd.DataFrame(pred_log_data)

    def run_epoch(self, epoch, mode="TRAIN"):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        # self.writer.close()  # uncomment when using tensorboard
        pass

    def print_model_summary(self):
        img_size = self.data_config["img_size"]
        summary(self.model, (3, img_size, img_size), device="cpu")