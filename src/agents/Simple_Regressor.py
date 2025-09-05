"""
Simple Image-based Classifier agent
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time
import wandb
import logging

import torch
import torch.optim as optim
from torch.backends import cudnn
from torchsummary import summary

from copy import deepcopy
from tqdm import tqdm

from src.agents.base import BaseAgent
from src.loss.loss import MSE
#from src.utils.view_quality_data_utils import class_labels
from src.utils.ef_data_utils import class_labels
from src.utils.utils import makedir
import warnings

from sklearn.metrics import (
mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error, f1_score
)

cudnn.benchmark = True  # IF input size is same all the time, it's faster this way
warnings.filterwarnings("ignore")

class Simple_Regressor(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # #################### define models ####################

        # ############# define dataset and dataloader ##########

        # #################### define loss  ###################
        # self.criterion = criterion_builder.build(config=self.train_config['criterion'])
        self.get_criterion()

        # #################### define optimizer  ###################
        # self.optimizer = optimizer_builder.build(self.train_config['optimizer'], self.model
        self.optimizer = self.get_optimizer()
        # Build the scheduler
        self.scheduler = self.get_lr_scheduler()

        # # #################### define Checkpointer  ################### 
        self.load_checkpoint(self.model_config["checkpoint_path"])

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_mean_mae = 0

       
    def get_criterion(self):
        """
        creates the pytorch criterion loss function by calling the corresponding loss class
        """
        config = deepcopy(self.train_config["criterion"])

        # classification cost
        self.criterion = MSE(**config["MSELoss"])

    def get_optimizer(self):
        """
        creates the pytorch optimizer
        """
        config = deepcopy(self.train_config["optimizer"])
        optimizer_name = config.pop("name")
        optimizer = optim.__dict__[optimizer_name](self.model.parameters(), **config)  # Adam
        return optimizer

    def get_lr_scheduler(self):
        config = deepcopy(self.train_config["lr_schedule"])
        scheduler_name = config.pop("name")
        config_lr = config[scheduler_name]
        scheduler = optim.lr_scheduler.__dict__[scheduler_name](self.optimizer, **config_lr)
        return scheduler

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.train_config["num_train_epochs"] + 1):
            self.current_epoch = epoch
            for mode in ["TRAIN", "VAL"]:
                logging.info(f'{"=" * 60}\nStarting {mode}')
                mean_mae, mean_rmse, mean_r2 = self.run_epoch(epoch, mode)

            if self.train_config["lr_schedule"]["name"] == "ReduceLROnPlateau":
                self.scheduler.step(mean_mae)
            elif self.train_config["lr_schedule"]["name"] != "OneCycleLR":
                self.scheduler.step()

            is_best = mean_mae < self.best_mean_mae
            if is_best:
                self.best_mean_mae = mean_mae
                logging.info(f"achieved best model with mean_mae of {mean_mae:.2%}")
            self.save_checkpoint(is_best=is_best)

    def evaluate(self, mode="VAL"):
        mae, r2, mse, rmse = self.run_eval(self.current_epoch, mode=mode)
        return mae, rmse, r2
    
    def run_epoch(self, epoch, mode="TRAIN"):
        logging.info(f"Epoch: {epoch} starting {mode}")
        if mode == "TRAIN":
            self.model.train()
        else:
            self.model.eval()

        data_loader = self.data_loaders[mode]
        epoch_steps = len(data_loader)

        label_names = class_labels[self.data_config["label_scheme_name"]]

        n_batches = 0
        total_loss = 0.

        y_pred_all = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        epoch_pred_log_df = pd.DataFrame()

        start = time.time()

        with torch.set_grad_enabled(mode = True if mode == "TRAIN" else False):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            mae_batch = 0.
            r2_batch = 0.
            rmse_batch = 0.
            for i in iterator:
                batch_log_dict = {}
                step = epoch * epoch_steps + i
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)
                target = data_sample["target_EF"].to(self.device).float()

                y_pred = self.model(input) 

                # compute loss
                loss = self.criterion.compute(pred=y_pred.view(-1), target=target)

                ####### evaluation statistics ##########
                #y_pred_class_all = torch.concat([y_pred_class_all, y_pred])
                y_pred_all = torch.concat([y_pred_all, y_pred.detach().cpu()])
                y_true = target.detach().cpu()
                y_true_all = torch.concat([y_true_all, y_true])
                
                
                print("GT:\n", y_true.numpy())
                print("Pred:\n", y_pred.detach().cpu().numpy())
                # mae
                mae_batch = mean_absolute_error(y_true.numpy(), y_pred.detach().cpu().numpy()) # TODO: consider using raw value param for better understanding

                # r2
                r2_batch = r2_score(y_true.numpy(), y_pred.detach().cpu().numpy())
                # rmse
                rmse_batch = root_mean_squared_error(y_true.numpy(), y_pred.detach().cpu().numpy())


                # ########################## Logging batch information on console ###############################

                iterator.set_description(
                    f"Epoch: {epoch} | {mode} | "
                    f"total Loss: {loss.item():.4f} | "
                    f"mae: {mae_batch:.2f} | rmse: {rmse_batch:.2f}| r2: {r2_batch:.2f} |",
                    refresh=True,
                )

                # ########################## Logging batch information on Wandb ###############################
                if self.config["wandb_mode"] != "disabled":
                    batch_log_dict.update(
                        {
                            # mode is 'val', 'val_push', or 'train
                            f"batch_{mode}/step": step,
                            # ######################## Loss Values #######################
                            f"batch_{mode}/loss_all": loss.item(),
                            # f'batch_{mode}/loss_Fl': focal_loss.item(),
                            # ######################## Eval metrics #######################
                            f"batch_{mode}/mae_mean": mae_batch,
                            f"batch_{mode}/rmse_mean": rmse_batch,
                            f"batch_{mode}/r2_mean": r2_batch,
                        }
                    )
                    #batch_log_dict.update(
                    #    {f"batch_{mode}/_{label}": value for label, value in zip(label_names, f1_batch)}
                    #)
                    # logging all information
                    wandb.log(batch_log_dict)


                if mode == "TRAIN":
                    loss.backward()

                    if (i + 1) % self.train_config["accumulation_steps"] == 0: # TODO: what is this?
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    self.current_iteration += 1

                    

                total_loss += loss.item()
                n_batches += 1

                if mode != "TRAIN" and (epoch % 5 == 0):
                    # ##### creating the prediction log table for saving the performance for each case
                    epoch_pred_log_df = pd.concat( #TODO
                        [
                            epoch_pred_log_df,
                            self.create_pred_log_df(
                                data_sample,
                                y_pred.detach().cpu(),
                            ),
                        ],
                        axis=0,
                    )

        end = time.time()

        ######################################################################################
        # ###################################### Calculating Metrics #########################
        ######################################################################################
        y_pred_all = y_pred_all.numpy()
        y_true_all = y_true_all.numpy()

        r2 = r2_score(y_true_all, y_pred_all)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        rmse = root_mean_squared_error(y_true_all, y_pred_all)

        total_loss /= n_batches
        #################################################################################
        # #################################### Consol Logs ##############################
        #################################################################################
        if mode == "test":
            logging.info(f"predicted labels for {mode} dataset are :\n {y_pred_all}")

        logging.info(
            f"Epoch:{epoch}_{mode} | Time:{end - start:.0f} | Total_Loss:{total_loss :.3f} | "
            f"MAE: {mae:.2f} | r2: {r2:.2f} | rmse: {rmse:.2f}"
        )
        
        #################################################################################
        ################################### CSV Log #####################################
        #################################################################################
        if mode != "TRAIN" and (epoch % 5 == 0): # TODO: why 5
            path_to_csv = os.path.join(self.config["save_dir"], f"csv_{mode}")
            makedir(path_to_csv)
            # epoch_pred_log_df.reset_index(drop=True).to_csv(os.path.join(path_to_csv, f'e{epoch:02d}_Auc{AUC.mean():.0%}.csv'))
            epoch_pred_log_df.reset_index(drop=True).to_csv(
                os.path.join(path_to_csv, f"e{epoch:02d}_mae_{mae:.2f}_rmse_{rmse:.2f}_r2_{r2:.2f}.csv")
            )

        # ########################## Logging epoch information on Wandb ###############################
        if self.config["wandb_mode"] != "disabled":
            epoch_log_dict = {
                # mode is 'val', 'val_push', or 'train
                f"epoch": epoch,
                # ######################## Loss Values #######################
                f"epoch/{mode}/loss_all": total_loss,
                # ######################## Eval metrics #######################
                f"epoch/{mode}/mae_mean": mae,
                f"epoch/{mode}/rmse_mean": rmse,
                f"epoch/{mode}/r2": r2,
            }
            self.log_lr(epoch_log_dict)
            # logging all information
            wandb.log(epoch_log_dict)

        return mae, rmse, r2
    
    def run_eval(self,  epoch, mode):
        '''
        Function instead of run_epoch, in order to evaluate the model after training. Supports multiple clips for test.
        '''
        logging.info(f"Epoch: {epoch} starting {mode}")
        self.model.eval()
        dataloader_mode = mode
        data_loader = self.data_loaders[dataloader_mode]
        epoch_steps = len(data_loader)
        
        label_names = "EF_pred"

        n_batches = 0
        total_loss = np.zeros(10)  # ce, mse, mae, cluster, psd, ortho, om_l2, om_trns, fc_l1

        #y_pred_class_all = torch.FloatTensor()
        y_pred_all = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        epoch_pred_log_df = pd.DataFrame()

        start = time.time()

        with torch.set_grad_enabled(mode = False): 
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)
            # accu_batch = 0
            for i in iterator:
                batch_log_dict = {}
                step = epoch * epoch_steps + i
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)
                target = data_sample["target_EF"].to(self.device)
                
                logits = torch.zeros(input.shape[0], input.shape[1]).to(self.device)
                #occurrence_maps = torch.zeros(input.shape[0], input.shape[1], self.model.prototype_shape[0], input.shape[2], input.shape[3]).to(self.device)
                for clip in range(input.shape[1]):
                    vid = input[:, clip].squeeze(1) # shape: (batch, 1, T, H, W) -> (batch, T, H, W)
                    logit = self.model(vid)
                    logits[:, clip] = logit.view(-1)
                logit = logits.mean(dim=1).view(-1) # shape (batch, 1) -> (batch)
        

                ####### evaluation statistics ##########
                y_pred_all = torch.concat([y_pred_all, logit.cpu().detach()])
                y_true = target.detach().cpu()
                y_true_all = torch.concat([y_true_all, y_true])
                #print("y_true", y_true, "y_pred", logit.view(-1).cpu().detach() ) # TODO: debugging only
                ## Reg metrics
                # MAE
                mae_batch = mean_absolute_error(y_true.numpy(), logit.cpu().detach().numpy())
                # R2
                r2_batch = r2_score(y_true.numpy(), logit.cpu().detach().numpy())
                # MSE
                mse_batch = mean_squared_error(y_true.numpy(), logit.cpu().detach().numpy())
                # RMSE
                rmse_batch = root_mean_squared_error(y_true.numpy(), logit.cpu().detach().numpy())

                n_batches += 1


                # ########################## Logging batch information on console ###############################
                # cm_flattened = [list(cm[j].flatten()) for j in range(cm.shape[0])]
                iterator.set_description(
                    f"Epoch: {epoch} | {mode} | "
                    f"MAE: {mae_batch:.2f} | "
                    f"R2: {r2_batch:.2f} | "
                    f"MSE: {mse_batch:.2f} | " 
                    f"RMSE: {rmse_batch:.2f} | ",
                    refresh=True,
                )

                # ########################## Logging batch information on Wandb ###############################
                if self.config["wandb_mode"] != "disabled":
                    batch_log_dict.update(
                        {
                            # mode is 'val', 'val_push', or 'train
                            f"batch_{mode}/step": step,
                            
                            # ######################## Eval metrics #######################
                            
                            f"batch_{mode}/MAE": mae_batch,
                            f"batch_{mode}/R2": r2_batch,
                            f"batch_{mode}/MSE": mse_batch,
                            f"batch_{mode}/RMSE": rmse_batch,
                        }
                    )
                    
                    # logging all information
                    wandb.log(batch_log_dict)

                # save model y_pred_all in CSV
                
                
                # ##### creating the prediction log table for saving the performance for each case
                epoch_pred_log_df = pd.concat(
                    [
                        epoch_pred_log_df,
                        self.create_pred_log_df(
                            data_sample,
                            logit.detach().cpu(),
                            logit_names=None,
                        ),
                    ],
                    axis=0,
                )

        end = time.time()

        ######################################################################################
        # ###################################### Calculating Metrics #########################
        ######################################################################################

        # y_pred_class_all = y_pred_class_all.numpy()
        y_pred_all = y_pred_all.numpy()
        y_true_all = y_true_all.numpy()


        mae_total = mean_absolute_error(y_true_all, y_pred_all)
        # per group mae
        mae_50 = mean_absolute_error(y_true_all[y_true_all>=50], y_pred_all[y_true_all>=50])
        mae_40 = mean_absolute_error(y_true_all[(y_true_all>=40) & (y_true_all<50)], y_pred_all[(y_true_all>=40) & (y_true_all<50)])
        mae_30 = mean_absolute_error(y_true_all[y_true_all<40], y_pred_all[y_true_all<40])
        mae = np.array([mae_30, mae_40, mae_50]) # shape (3,)

        # get f1 score of below 40% EF
        f1 = f1_score(y_true_all < 40, y_pred_all < 40)

        r2 = r2_score(y_true_all, y_pred_all)
        mse = mean_squared_error(y_true_all, y_pred_all)
        rmse = root_mean_squared_error(y_true_all, y_pred_all)

        #################################################################################
        # #################################### Scatter plot #############################
        #################################################################################
        # scatter plot of the true vs predicted values with a line of best fit
        plt.figure()
        plt.scatter(y_true_all, y_pred_all, alpha=0.5, color='blue')
        plt.plot([y_true_all.min(), y_true_all.max()], [y_true_all.min(), y_true_all.max()], color='black')
        plt.plot(np.unique(y_true_all), np.poly1d(np.polyfit(y_true_all, y_pred_all, 1))(np.unique(y_true_all)), color='red')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'True vs Predicted Values')
        plt.savefig(os.path.join(self.config["save_dir"], f"e_{epoch}_scatter_{mode}.png"))
        plt.show()
        #################################################################################
        # #################################### Consol Logs ##############################
        #################################################################################
        if mode == "TEST":
            #logging.info(f"predicted labels for {mode} dataset are :\n {y_pred_class_all}")
            logging.info(f"predicted labels for {mode} dataset are :\n {y_pred_all}")
        print(mae)
        logging.info(
            f"Epoch:{epoch}_{mode} | Time:{end - start:.0f} | " #TODO: double check the sequence of losses
            f"MAE: {mae_total:.2f} | R2: {r2:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | f1<40: {f1:.2f}| mae: {[f'{mae[j]:.2f}' for j in range(mae.shape[0])]}"
        )
        
        #################################################################################
        ################################### CSV Log #####################################
        #################################################################################
        if mode == "VAL_push" or mode == "TEST":
            path_to_csv = os.path.join(self.config["save_dir"], f"{self.data_config['name']}_csv_{mode}")
            makedir(path_to_csv)
            # epoch_pred_log_df.reset_index(drop=True).to_csv(os.path.join(path_to_csv, f'e{epoch:02d}_Auc{AUC.mean():.0%}.csv'))
            epoch_pred_log_df.reset_index(drop=True).to_csv(
                os.path.join(path_to_csv, f"e{epoch:02d}_r2_{r2:.2f}.csv")
            )
            epoch_log_dic = {
                "R2": r2,
                "MSE": mse,
                "RMSE": rmse,
                "f1<40": f1,
                "MAE": mae_total,
                "mae_30": mae[0],
                "mae_40": mae[1],
                "mae_50": mae[2],
            }
            # save dic as csv
            pd.DataFrame.from_dict(epoch_log_dic, orient='index').to_csv(
                os.path.join(path_to_csv, f"e{epoch:02d}_metrics.csv"))
                
        # ########################## Logging epoch information on Wandb ###############################
        if self.config["wandb_mode"] != "disabled":
            epoch_log_dict = {
                # mode is 'val', 'val_push', or 'train
                f"epoch": epoch,
                # ######################## Eval metrics #######################
                f"epoch/{mode}/f1<40": f1,
                f"epoch/{mode}/MAE": mae_total,
                f"epoch/{mode}/R2": r2,
                f"epoch/{mode}/MSE": mse,
                f"epoch/{mode}/RMSE": rmse,
            }
            self.log_lr(epoch_log_dict)

            wandb.log(epoch_log_dict)
        return mae, r2, mse, rmse

    def log_lr(self, epoch_log_dict):  # TODO CHECK
        epoch_log_dict.update({"lr": self.optimizer.param_groups[0]["lr"]})

    def print_model_summary(self):
        img_size = self.data_config["img_size"]
        frames = self.data_config["frames"]
        # summary(self.model, torch.rand((self.train_config['batch_size'], 3, img_size, img_size)))
        summary(self.model, (3, frames, img_size, img_size), device="cpu")  # TODO Check
        # print(self.model)

    def load_checkpoint(self, file_name):
    
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:  # TODO REVIEW
            if (file_name is not None) and (os.path.exists(file_name)):
                checkpoint = torch.load(file_name, map_location=self.device)
                #print(checkpoint.keys())
                self.current_epoch = checkpoint["epoch"]
                self.current_iteration = checkpoint["iteration"]
                self.model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])

                logging.info(
                    "Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {}) to device{}\n".format(
                        file_name, checkpoint["epoch"], checkpoint["iteration"], self.device
                    )
                )


        except OSError as e:
            logging.error(f"Error {e}")
            logging.error("No checkpoint exists from '{}'. Skipping...".format(file_name))
            logging.error("**First time to train**")

    def run_extract_features(self):
        # save the classifier layer
        torch.save(self.model.backbone.fc.state_dict(), os.path.join(self.config["save_dir"], "fc_layer.pth"))
        self.extract_features(mode="TRAIN")
        self.extract_features(mode="VAL")
        self.extract_features(mode="TEST")
    
    def extract_features(self, mode="TRAIN"):
        logging.info(f"Starting {mode}")
        if mode == "TRAIN":
            self.model.eval() # doesnt really matter as grad is disabled
        else:
            self.model.eval()

        ###### Load the data ######
        data_loader = self.data_loaders[mode]

        epoch_pred_log_df = pd.DataFrame()
        features_extracted_list = torch.tensor([]) # Container to store features

        with torch.set_grad_enabled(False):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            for i in iterator:
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)
                pred, features_extracted = self.model.feature_extractor(input)

                features_extracted = features_extracted.view(features_extracted.size(0), -1)  # Flatten the features
                # Concatenate extracted features
                features_extracted_list = torch.cat((features_extracted_list, features_extracted.detach().cpu()))

                # save model preds in CSV
                # ##### creating the prediction log table for saving the performance for each case
                epoch_pred_log_df = pd.concat(
                    [
                        epoch_pred_log_df,
                        self.create_pred_log_df(
                            data_sample,
                            pred.detach().cpu(),
                        ),
                    ],
                    axis=0,
                )

                iterator.set_description(f"Mode: {mode} | ", refresh=True,)

        #################################################################################
        ################################### CSV Log #####################################
        #################################################################################
        torch.save(features_extracted_list, os.path.join(self.config["save_dir"], f"features_{mode}.pt"))
        epoch_pred_log_df.reset_index(drop=True).to_csv(os.path.join(self.config["save_dir"], f"features_{mode}.csv"))