"""
Image-based ProtoPNet agent, uses the architecture of ProtoPNet (Neurips 2019)
"""
import os
import numpy as np
import pandas as pd
import time
import logging

import torch
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from copy import deepcopy
from tqdm import tqdm
import wandb

from src.agents.base import BaseAgent
from src.data.as_dataloader_f import get_as_dataloader
from src.data.view_dataloader_f import get_dataloader
from src.data.ef_dataloader_f import get_ef_dataloader
from src.loss.loss import ClusterPatch, SeparationPatch, L_norm, CeLoss
from src.utils import push_ProtoPNet
from ..utils.utils import makedir
from src.utils.preprocess import preprocess_input_function
# from src.utils.as_tom_data_utils import class_labels #TODO: generalise this so that one class_label for each dataset
from src.utils.ef_data_utils import class_labels

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
    balanced_accuracy_score,
    f1_score,
)

DATALOADERS = {
    "view_quality": get_dataloader,
    "as_tom": get_as_dataloader,
    "ef": get_ef_dataloader,
}

cudnn.benchmark = True  # IF input size is same all the time, it's faster this way


class ProtoPNet_Base(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # #################### define models ####################

        # ############# define dataset and dataloader ##########'
        if "TRAIN_push" in self.data_config["modes"]:
            self.data_loaders.update({"TRAIN_push": DATALOADERS[self.data_config["name"]](self.data_config, split="TRAIN", mode="push")})
            # self.data_loaders.update({'train_push': DataLoader(self.datasets['train_push'],
            #                                                    shuffle=False,
            #                                                    drop_last=False,
            #                                                    batch_size=self.train_config['batch_size'],
            #                                                    num_workers=self.train_config['num_workers'],
            #                                                    pin_memory=True)})

        # #################### define loss  ###################
        self.get_criterion()

        # #################### define optimizer  ###################
        # 3 step optimization,  warm-up, joint, last-layer only
        self.get_optimizer()
        # Build the scheduler for the joint optimizer only
        self.scheduler = self.get_lr_scheduler()

        # #################### define Evaluators  ###################  #TODO make systematic after initial runs
        # add other required configs
        # self.eval_config.update({'frame_size': self.data_config['transform']['image_size'],
        #                          'batch_size': self.train_config['batch_size'],
        #                          'use_coordinate_graph': self.use_coordinate_graph})
        # self.evaluators = evaluator_builder.build(self.eval_config)

        # # #################### define Checkpointer  ################### #TODO use builder?
        # # self.checkpointer = checkpointer_builder.build(
        # #     self.save_dir, self.model, self.optimizer,
        # #     self.scheduler, self.eval_config['standard'], best_mode='min')
        #

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # ########## resuming model training if config.resume is provided ############## #TODO use builder?
        # checkpoint_path = self.model_config.get('checkpoint_path', '')
        # self.misc = self.checkpointer.load(
        #     mode, checkpoint_path, use_latest=False)
        self.load_checkpoint(self.model_config["checkpoint_path"])

    def get_criterion(self):
        """
        creates the pytorch criterion loss function by calling the corresponding loss class
        """
        config = deepcopy(self.train_config["criterion"])

        # classification cost
        self.criterion = CeLoss(**config["CeLoss"])

        # prototypical layer cost
        num_classes = self.model.num_classes
        self.Cluster = ClusterPatch(num_classes=num_classes, **config["ClusterPatch"])
        self.Separation = SeparationPatch(num_classes=num_classes, **config["SeparationPatch"])

        # regularizations for classification layer
        self.Lnorm_fc = L_norm(**config["Lnorm_FC"], mask=1 - torch.t(self.model.prototype_class_identity))

    def get_optimizer(self):
        """
        creates the pytorch optimizer
        """
        config = deepcopy(self.train_config["optimizer"])

        joint_optimizer_specs = [
            {
                "params": self.model.features.parameters(),
                "lr": config["joint_lrs"]["features"],
                "weight_decay": 1e-3,
            },
            # bias are now also being regularized
            {
                "params": self.model.add_on_layers.parameters(),
                "lr": config["joint_lrs"]["add_on_layers"],
                "weight_decay": 1e-3,
            },
            {
                "params": self.model.prototype_vectors,
                "lr": config["joint_lrs"]["prototype_vectors"],
            },
        ]
        self.joint_optimizer = torch.optim.Adam(joint_optimizer_specs)

        warm_optimizer_specs = [
            {
                "params": self.model.add_on_layers.parameters(),
                "lr": config["warm_lrs"]["add_on_layers"],
                "weight_decay": 1e-3,
            },
            {
                "params": self.model.prototype_vectors,
                "lr": config["warm_lrs"]["prototype_vectors"],
            },
        ]
        self.warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        last_layer_optimizer_specs = [
            {
                "params": self.model.last_layer.parameters(),
                "lr": config["last_layer_lr"],
            }
        ]
        self.last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    def get_lr_scheduler(self):
        config = deepcopy(self.train_config["lr_schedule"])
        scheduler_name = config.pop("name")

        scheduler = optim.lr_scheduler.__dict__[scheduler_name](self.joint_optimizer, **config)
        return scheduler

    def push(self, replace_prototypes=True):
        """
        pushing prototypes
        :param replace_prototypes: to replace prototypes with the closest features or not
        """
        epoch = f"{self.current_epoch}_pushed"
        push_ProtoPNet.push_prototypes(
            self.data_loaders["TRAIN_push"],  # pytorch dataloader (must be unnormalized in [0,1])
            model=self.model,  # pytorch network with prototype_vectors
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            root_dir_for_saving_prototypes=os.path.join(
                self.config["save_dir"], "img"
            ),  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix="prototype-img",
            prototype_self_act_filename_prefix="prototype-self-act",
            proto_bound_boxes_filename_prefix="bb",
            replace_prototypes=replace_prototypes,
        )

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.train_config["num_train_epochs"]):
            self.current_epoch = epoch

            if epoch < self.train_config["num_warm_epochs"]:
                self.warm_only()
                accu, mean_f1, auc = self.run_epoch(epoch, self.warm_optimizer, mode="TRAIN")
            else:
                self.joint()
                accu, mean_f1, auc = self.run_epoch(epoch, self.joint_optimizer, mode="TRAIN")
                self.scheduler.step()

            accu, mean_f1, auc = self.run_epoch(epoch, mode="VAL")
            self.save_model_w_condition(
                model_dir=self.config["save_dir"],
                model_name=f"{epoch}nopush",
                metric_dict={"f1": mean_f1},
                threshold=0.65,
            )

            if (epoch >= self.train_config["push_start"]) and (epoch % self.train_config["push_rate"] == 0):
                self.push()
                accu, mean_f1, auc = self.run_epoch(epoch, mode="VAL_push")
                self.save_model_w_condition(
                    model_dir=self.config["save_dir"],
                    model_name=f"{epoch}push",
                    metric_dict={"f1": mean_f1},
                    threshold=0.65,
                )

                if self.model_config["prototype_activation_function"] != "linear":
                    self.last_only()
                    for i in range(2):
                        logging.info("iteration: \t{0}".format(i))
                        accu, mean_f1, auc = self.run_epoch(epoch, self.last_layer_optimizer, mode="TRAIN")
                        accu, mean_f1, auc = self.run_epoch(epoch, mode="VAL")
                        self.save_model_w_condition(
                            model_dir=self.config["save_dir"],
                            model_name=f"{epoch}_{i}push",
                            metric_dict={"f1": mean_f1},
                            threshold=0.65,
                        )

                is_best = mean_f1 > self.best_metric
                if is_best:
                    self.best_metric = mean_f1
                    logging.info(f"achieved best model with mean_f1 of {mean_f1}")
                self.save_checkpoint(is_best=is_best)

    def evaluate(self, mode="VAL"):
        accu, mean_f1, auc = self.run_epoch(self.current_epoch, mode=mode)
        return accu, mean_f1, auc

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:  # TODO REVIEW
            if (file_name is not None) and (os.path.exists(file_name)):
                checkpoint = torch.load(file_name)

                self.current_epoch = checkpoint["epoch"]
                self.current_iteration = checkpoint["iteration"]
                self.model.load_state_dict(checkpoint["state_dict"])
                self.warm_optimizer.load_state_dict(checkpoint["warm_optimizer"])
                self.joint_optimizer.load_state_dict(checkpoint["joint_optimizer"])
                self.last_layer_optimizer.load_state_dict(checkpoint["last_layer_optimizer"])

                logging.info(
                    "Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(
                        file_name, checkpoint["epoch"], checkpoint["iteration"]
                    )
                )

        except OSError as e:
            logging.error(f"Error {e}")
            logging.error("No checkpoint exists from '{}'. Skipping...".format(file_name))
            logging.error("**First time to train**")

    def save_model_w_condition(self, model_dir, model_name, metric_dict, threshold):
        name, metric = metric_dict.popitem()
        if metric > threshold:
            state = self.get_state()
            logging.info(f"\t {name} above {threshold:.2%}")
            torch.save(
                state,
                f=os.path.join(model_dir, (model_name + f"_{name}-{metric:.4f}.pth")),
            )

    def get_state(self):
        return {
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "state_dict": self.model.state_dict(),
            "warm_optimizer": self.warm_optimizer.state_dict(),
            "joint_optimizer": self.joint_optimizer.state_dict(),
            "last_layer_optimizer": self.last_layer_optimizer.state_dict(),
        }

    def warm_only(self):
        logging.info("\t#####################################################################")
        logging.info("\twarm")
        logging.info("\t#####################################################################")
        for p in self.model.features.parameters():
            p.requires_grad = False
        for p in self.model.add_on_layers.parameters():
            p.requires_grad = True
        self.model.prototype_vectors.requires_grad = True
        for p in self.model.last_layer.parameters():
            p.requires_grad = True

    def joint(self):
        logging.info("\t#####################################################################")
        logging.info("\tjoint")
        logging.info("\t#####################################################################")
        for p in self.model.features.parameters():
            p.requires_grad = True
        for p in self.model.add_on_layers.parameters():
            p.requires_grad = True
        self.model.prototype_vectors.requires_grad = True
        for p in self.model.last_layer.parameters():
            p.requires_grad = True

    def last_only(self):
        logging.info("\t#######################")
        logging.info("\tlast layer")
        logging.info("\t#######################")
        for p in self.model.features.parameters():
            p.requires_grad = False
        for p in self.model.add_on_layers.parameters():
            p.requires_grad = False
        self.model.prototype_vectors.requires_grad = False
        for p in self.model.last_layer.parameters():
            p.requires_grad = True

    def run_epoch(self, epoch, optimizer=None, mode="TRAIN"):
        logging.info(f"Epoch: {epoch} starting {mode}")
        if mode == "TRAIN":
            self.model.train()
        else:
            self.model.eval()

        if "_push" in mode:
            dataloader_mode = mode.split("_")[0]
        else:
            dataloader_mode = mode
        data_loader = self.data_loaders[dataloader_mode]
        epoch_steps = len(data_loader)

        label_names = class_labels[self.data_config["label_scheme_name"]]

        n_batches = 0
        total_loss = np.zeros(4)

        y_pred_all = torch.FloatTensor()
        y_pred_class_all = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        epoch_pred_log_df = pd.DataFrame()

        start = time.time()

        with torch.set_grad_enabled(mode == "TRAIN"):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            for i in iterator:
                batch_log_dict = {}
                step = epoch * epoch_steps + i
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)
                target = data_sample["target_AS"].to(self.device)

                logit, min_distances = self.model(input)

                ############ Compute Loss ###############
                # CE loss for multilabel data
                cross_entropy = self.criterion.compute(logit, target)
                # cluster cost
                cluster_cost = self.Cluster.compute(min_distances, target)
                # separation cost
                separation_cost = self.Separation.compute(min_distances, target)
                # FC layer L1 regularization
                fc_lnorm = self.Lnorm_fc.compute(self.model.last_layer.weight)

                loss = cross_entropy + cluster_cost + separation_cost + fc_lnorm

                ####### evaluation statistics ##########
                y_pred_prob = logit.softmax(dim=1).cpu()
                y_pred_max_prob, y_pred_class = y_pred_prob.max(dim=1)
                y_pred_all = torch.concat([y_pred_all, y_pred_prob.detach()])
                y_pred_class_all = torch.concat([y_pred_class_all, y_pred_class.detach()])
                y_true = target.detach().cpu()
                y_true_all = torch.concat([y_true_all, y_true])

                # f1 score
                f1_batch = f1_score(
                    y_true.numpy(),
                    y_pred_class.numpy(),
                    average=None,
                    labels=range(len(label_names)),
                    zero_division=0,
                )
                # confusion matrix
                # cm = confusion_matrix(y_true.numpy(), y_pred_class.numpy(), labels=range(len(label_names)))
                # Accuracy
                accu_batch = balanced_accuracy_score(y_true.numpy(), y_pred_class.numpy())

                # ########################## Logging batch information on console ###############################
                # cm_flattened = [list(cm[j].flatten()) for j in range(cm.shape[0])]
                iterator.set_description(
                    f"Epoch: {epoch} | {mode} | "
                    f"total Loss: {loss.item():.4f} | "
                    f"CE loss {cross_entropy.item():.2f} | "
                    f"Cls {cluster_cost.item():.2f} | "
                    f"Sep {separation_cost.item():.2f} | "
                    f"fc_l1 {fc_lnorm.item():.4f} | "
                    f"Acc: {accu_batch:.2%} | f1: {f1_batch.mean():.2f} |",
                    refresh=True,
                )

                if mode == "TRAIN":
                    loss.backward()
                    if (i + 1) % self.train_config["accumulation_steps"] == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    self.current_iteration += 1

                total_loss += np.asarray(
                    [
                        cross_entropy.item(),
                        cluster_cost.item(),
                        separation_cost.item(),
                        fc_lnorm.item(),
                    ]
                )
                n_batches += 1

                # ########################## Logging batch information on Wandb ###############################
                if self.config["wandb_mode"] != "disabled":
                    batch_log_dict.update(
                        {
                            # mode is 'val', 'val_push', or 'train
                            f"batch_{mode}/step": step,
                            # ######################## Loss Values #######################
                            f"batch_{mode}/loss_all": loss.item(),
                            f"batch_{mode}/loss_Clst": cluster_cost.item(),
                            f"batch_{mode}/loss_Sep": separation_cost.item(),
                            f"batch_{mode}/loss_fcL1Norm": fc_lnorm.item(),
                            # ######################## Eval metrics #######################
                            f"batch_{mode}/f1_mean": f1_batch.mean(),
                            f"batch_{mode}/accuracy": accu_batch,
                        }
                    )
                    batch_log_dict.update(
                        {f"batch_{mode}/f1_{as_label}": value for as_label, value in zip(label_names, f1_batch)}
                    )
                    wandb.log(batch_log_dict)

                # save model preds in CSV
                if mode == "VAL_push" or mode == "TEST":
                    ###### creating the prediction log table for saving the performance for each case
                    epoch_pred_log_df = pd.concat(
                        [
                            epoch_pred_log_df,
                            self.create_pred_log_df(
                                data_sample,
                                logit.detach().cpu(),
                                logit_names=label_names,
                            ),
                        ],
                        axis=0,
                    )

        end = time.time()

        ######################################################################################
        # ###################################### Calculating Metrics #########################
        ######################################################################################
        y_pred_class_all = y_pred_class_all.numpy()
        y_pred_all = y_pred_all.numpy()
        y_true_all = y_true_all.numpy()

        accu = balanced_accuracy_score(y_true_all, y_pred_class_all)
        f1 = f1_score(
            y_true_all,
            y_pred_class_all,
            average=None,
            labels=range(len(label_names)),
            zero_division=0,
        )
        f1_mean = f1.mean()

        try:
            AUC = roc_auc_score(
                y_true_all,
                y_pred_all,
                average="weighted",
                multi_class="ovr",
                labels=range(len(label_names)),
            )
        except ValueError:
            logging.error("AUC calculation failed, setting it to 0")
            AUC = 0

        total_loss /= n_batches

        #################################################################################
        # #################################### Consol Logs ##############################
        #################################################################################
        if mode == "TEST":
            logging.info(f"predicted labels for {mode} dataset are :\n {y_pred_class_all}")

        logging.info(
            f"Epoch:{epoch}_{mode} | "
            f"Time:{end - start:.0f} | "
            f"Total_Loss:{total_loss.sum() :.3f} | "
            f"[ce, clst, sep, fc_lnorm]={[f'{total_loss[j]:.3f}' for j in range(total_loss.shape[0])]} | "
            f"Acc: {accu:.2%} | "
            f"f1: {[f'{f1[j]:.0%}' for j in range(f1.shape[0])]} | "
            f"f1_avg: {f1_mean:.2f} | "
            f"AUC: {AUC}"
        )
        logging.info(
            f"\tConfusion matrix: \n {confusion_matrix(y_true_all, y_pred_class_all, labels=range(len(label_names)))}"
        )
        logging.info(classification_report(y_true_all, y_pred_class_all, zero_division=0, target_names=label_names))

        #################################################################################
        ################################### CSV Log #####################################
        #################################################################################
        if mode == "VAL_push" or mode == "TEST":
            path_to_csv = os.path.join(self.config["save_dir"], f"{self.data_config['name']}_csv_{mode}")
            makedir(path_to_csv)
            epoch_pred_log_df.reset_index(drop=True).to_csv(
                os.path.join(path_to_csv, f"e{epoch:02d}_f1_{f1_mean:.0%}.csv")
            )

        #################################################################################
        # ###################### Logging epoch information on Wandb #####################
        #################################################################################
        if self.config["wandb_mode"] != "disabled":
            epoch_log_dict = {
                # mode is 'val', 'val_push', or 'train
                f"epoch": epoch,
                # ######################## Loss Values #######################
                f"epoch/{mode}/loss_all": total_loss.sum(),
                # ######################## Eval metrics #######################
                f"epoch/{mode}/f1_mean": f1_mean,
                f"epoch/{mode}/accuracy": accu,
                f"epoch/{mode}/AUC_mean": AUC,
            }
            epoch_log_dict.update({f"epoch/{mode}/f1_{as_label}": value for as_label, value in zip(label_names, f1)})
            loss_names = ["loss_CE", "loss_Clst", "loss_Sep", "loss_fcL1Norm"]
            epoch_log_dict.update(
                {f"epoch/{mode}/{loss_name}": value for loss_name, value in zip(loss_names, total_loss)}
            )
            wandb.log(epoch_log_dict)

        return accu, f1_mean, AUC
