"""
Agent for video-based network, our model, trained end-to-end, inherits the image-based agent.
"""
import os
import numpy as np
import pandas as pd
import time
import wandb
import logging
import matplotlib.pyplot as plt

import torch
from torch.backends import cudnn
from torchsummary import summary
from ..utils.metrics import SparsityMetric

from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error, 
    r2_score, 
    mean_squared_error, 
    root_mean_squared_error
)

from src.agents.protoEF_e2e import protoEF_e2e
from src.utils.ef_data_utils import class_labels
from src.utils.utils import makedir, plot_confusion_matrix, plot_embeddings

cudnn.benchmark = True  # IF input size is same all the time, it's faster this way


class Video_protoEF_e2e(protoEF_e2e):
    def __init__(self, config):
        super().__init__(config)

    def run_epoch(self, epoch, optimizer=None, mode="TRAIN"):
        logging.info(f"Epoch: {epoch} starting {mode}")
        if mode == "TRAIN":
            self.model.train()
        else:
            self.model.eval()

        if "_push" in mode:
            # if val_push, use val for dataloder
            dataloader_mode = mode.split("_")[0]
        else:
            dataloader_mode = mode
        data_loader = self.data_loaders[dataloader_mode]
        epoch_steps = len(data_loader)

        label_scheme = self.data_config["label_scheme_name"]

        # Check if the label scheme exists, otherwise dynamically create it
        if label_scheme not in class_labels and label_scheme.startswith("ef_"):
            try:
                n = int(label_scheme.split("_")[1].replace("class", ""))  # Extract n from "ef_nclass"
                class_labels[label_scheme] = [f"Class {i+1}" for i in range(n)]  # Dynamically generate labels
            except ValueError:
                raise ValueError(f"Invalid label scheme name: {label_scheme}")

        label_names = class_labels[label_scheme]
        
        #logit_names = label_names + ["abstain"] if self.config["abstain_class"] else label_names
        num_class_prototypes = 40

        n_batches = 0
        total_loss = np.zeros(10)  # ce, mse, mae, cluster, psd, ortho, om_l2, om_trns, fc_l1

        #y_pred_class_all = torch.FloatTensor()
        y_pred_all = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        epoch_pred_log_df = pd.DataFrame()

        start = time.time()

        # Diversity Metric
        count_array = np.zeros(self.model.prototype_shape[0])
        simscore_cumsum = torch.zeros(self.model.prototype_shape[0])

        # Reset sparsity metric
        getattr(self, f"{mode.lower()}_sparsity_80").reset()

        with torch.set_grad_enabled(mode = True if mode == "TRAIN" else False):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)
            # accu_batch = 0
            for i in iterator:
                batch_log_dict = {}
                step = epoch * epoch_steps + i
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)
                target = data_sample["target_EF"].to(self.device) 
                LV_masks = data_sample["lv_mask"].to(self.device)
                
                logit, similarities, occurrence_map, beta = self.model(input)
                logit = logit.view(-1)
                ############ Compute Loss ###############
                # CrossEntropy loss for Multiclass data
                ce_loss = self.CeLoss.compute(logits=logit, target=target)
                # reg loss
                mse_loss = self.MSELoss.compute(logit, target)
                # mae loss
                mae_loss = self.MAELoss.compute(logit, target) 
                # cluster cost
                cluster_cost = self.Cluster.compute(similarities, target, self.model)
                # psd cost
                psd_cost = self.PSD.compute(1.0 - similarities, target)
                # proto decorrelation cost
                decor_cost = self.Decorrelation.compute(self.model)
                # to encourage diversity on learned prototypes
                orthogonality_loss = self.Orthogonality.compute(self.model.prototype_vectors)
                # occurrence map L1 regularization without LV mask
                lv_mask_broadcasted = LV_masks.unsqueeze(1).unsqueeze(3).expand(-1, self.model.prototype_shape[0], -1, int(self.data_config["frames"]/4), -1, -1)
                occurrence_map_lnorm = self.Lnorm_occurrence.compute(occurrence_map, lv_mask_broadcasted, dim=(-3, -2, -1)) #norm across T, H, W
                # occurrence map transformation regularization
                occurrence_map_trans = self.Trans_occurrence.compute(input, occurrence_map, self.model)
                # FC layer L1 regularization
                fc_lnorm = self.Lnorm_fc.compute(self.model.last_layer.weight)
                
                loss = (
                    ce_loss
                    + mse_loss
                    + mae_loss
                    + cluster_cost
                    + psd_cost
                    + decor_cost
                    + orthogonality_loss
                    + occurrence_map_lnorm
                    + occurrence_map_trans
                    + fc_lnorm
                )

                ####### evaluation statistics ##########
                if self.config["abstain_class"]:
                    # take only logits from the non-abstention class
                    y_pred_prob = logit[:, : self.model.num_classes - 1].softmax(dim=1).cpu()
               
                y_pred_all = torch.concat([y_pred_all, logit.cpu().detach()])
                y_true = target.detach().cpu()
                y_true_all = torch.concat([y_true_all, y_true])
                ## Reg metrics
                # MAE
                mae_batch = mean_absolute_error(y_true.numpy(), logit.cpu().detach().numpy())
                # R2
                r2_batch = r2_score(y_true.numpy(), logit.cpu().detach().numpy())
                # MSE
                mse_batch = mean_squared_error(y_true.numpy(), logit.cpu().detach().numpy())
                # RMSE
                rmse_batch = root_mean_squared_error(y_true.numpy(), logit.cpu().detach().numpy())

                if mode == "TRAIN":
                    loss.backward()
                    if (i + 1) % self.train_config["accumulation_steps"] == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    self.current_iteration += 1

                total_loss += np.asarray(
                    [
                        ce_loss.item(),
                        mse_loss.item(),
                        mae_loss.item(),
                        cluster_cost.item(),
                        psd_cost.item(),
                        decor_cost.item(),
                        orthogonality_loss.item(),  # prototypical layer
                        occurrence_map_lnorm.item(),
                        occurrence_map_trans.item(),  # ROI layer
                        fc_lnorm.item(),  # FC layer
                    ]
                )
                n_batches += 1

                if beta is not None:
                    sparsity_batch = getattr(self, f"{mode.lower()}_sparsity_80")(beta).item()
                else:
                    sparsity_batch = getattr(self, f"{mode.lower()}_sparsity_80")(similarities).item()

                # Determine the top 5 most similar prototypes to data
                # sort similarities in descending order
                if beta is not None:
                    sorted_similarities, sorted_indices = torch.sort(beta[:, :num_class_prototypes].detach().cpu(),
                                                                 descending=True)
                else:
                    sorted_similarities, sorted_indices = torch.sort(similarities[:, :num_class_prototypes].detach().cpu(),
                                                                 descending=True)
                # Add the top 5 most similar prototypes to the count array
                np.add.at(count_array[:num_class_prototypes], sorted_indices[:, :5], 1)

                if self.config["abstain_class"]:
                    # sort similarities in descending order
                    sorted_similarities, sorted_indices = torch.sort(
                        similarities[:, num_class_prototypes:].detach().cpu(), descending=True
                    )
                    # Add the type 5 most similar prototypes to the count array
                    np.add.at(count_array[num_class_prototypes:], sorted_indices[:, :2], 1)

                simscore_cumsum += similarities.sum(dim=0).detach().cpu()

                # ########################## Logging batch information on console ###############################
                iterator.set_description(
                    f"Epoch: {epoch} | {mode} | "
                    f"total Loss: {loss.item():.4f} | "
                    f"CE loss {ce_loss.item():.2f} | "
                    f"MSE loss {mse_loss.item():.2f} | "
                    f"MAE loss {mae_loss.item():.2f} | "
                    f"Cls {cluster_cost.item():.2f} | "
                    f"PSD {psd_cost.item():.2f} | "
                    f"Decor {decor_cost.item():.4f} | "
                    f"Ortho {orthogonality_loss.item():.2f} | "
                    f"om_l2 {occurrence_map_lnorm.item():.4f} | "
                    f"om_trns {occurrence_map_trans.item():.2f} | "
                    f"fc_l1 {fc_lnorm.item():.4f} | "
                    f"MAE: {mae_batch:.2f} | "
                    f"R2: {r2_batch:.2f} | "
                    f"MSE: {mse_batch:.2f} | " 
                    f"RMSE: {rmse_batch:.2f} | "
                    f"Sparsity: {sparsity_batch:.1f}",
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
                            f"batch_{mode}/loss_CE": ce_loss,
                            f"batch_{mode}/loss_MSE": mse_loss.item(),
                            f"batch_{mode}/loss_MAE": mae_loss.item(),
                            f"batch_{mode}/loss_Clst": cluster_cost.item(),
                            f"batch_{mode}/loss_PSD": psd_cost.item(),
                            f"batch_{mode}/loss_Decor": decor_cost.item(),
                            f"batch_{mode}/loss_Ortho": orthogonality_loss.item(),
                            f"batch_{mode}/loss_RoiNorm": occurrence_map_lnorm.item(),
                            f"batch_{mode}/loss_RoiTrans": occurrence_map_trans.item(),
                            f"batch_{mode}/loss_fcL1Norm": fc_lnorm.item(),
                            # ######################## Eval metrics #######################
                            f"batch_{mode}/MAE": mae_batch,
                            f"batch_{mode}/R2": r2_batch,
                            f"batch_{mode}/MSE": mse_batch,
                            f"batch_{mode}/RMSE": rmse_batch,
                            f"batch_{mode}/sparsity": sparsity_batch,
                        }
                    )
                    
                    # logging all information
                    wandb.log(batch_log_dict)

                # save model y_pred_all in CSV
                if mode == "VAL_push" or mode == "TEST": 
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


        ### loss
        total_loss /= n_batches
        ## reg metrics
        mae_total = mean_absolute_error(y_true_all, y_pred_all)
        # per group mae
        mae_50 = mean_absolute_error(y_true_all[y_true_all>=50], y_pred_all[y_true_all>=50])
        mae_40 = mean_absolute_error(y_true_all[(y_true_all>=40) & (y_true_all<50)], y_pred_all[(y_true_all>=40) & (y_true_all<50)])
        mae_30 = mean_absolute_error(y_true_all[y_true_all<40], y_pred_all[y_true_all<40])
        mae = np.array([mae_30, mae_40, mae_50]) # shape (3,)

        r2 = r2_score(y_true_all, y_pred_all)
        mse = mean_squared_error(y_true_all, y_pred_all)
        rmse = root_mean_squared_error(y_true_all, y_pred_all)
        f1 = f1_score(y_true_all < 40, y_pred_all < 40)
        
        ### Diversity Metric Calculations
        # count how many prototypes were activated in at least 1% of the samples
        div_threshold = 0.05
        diversity = np.sum(count_array[:num_class_prototypes] > div_threshold * len(y_true_all))
        diversity_log = f"diversity: {diversity}"
        if self.config["abstain_class"]:
            diversity_abstain = np.sum(count_array[num_class_prototypes:] > div_threshold * len(y_true_all))
            diversity_log += f" | diversity_abstain: {diversity_abstain}"
        sorted_simscore_cumsum, sorted_indices = torch.sort(simscore_cumsum, descending=True)
        logging.info(f"sorted_simscore_cumsum is {sorted_simscore_cumsum}")

        sparsity_epoch = getattr(self, f"{mode.lower()}_sparsity_80").compute().item()

        #################################################################################
        # #################################### Consol Logs ##############################
        #################################################################################
        if mode == "TEST":
            logging.info(f"predicted labels for {mode} dataset are :\n {y_pred_all}")

        logging.info(
            f"Epoch:{epoch}_{mode} | Time:{end - start:.0f} | Total_Loss:{total_loss.sum() :.3f} | " #TODO: double check the sequence of losses
            f"[ce, mse, clst, psd, ortho, om_l2, om_trns, fc_l1]={[f'{total_loss[j]:.3f}' for j in range(total_loss.shape[0])]} \n"
            #f"Acc: {accu:.2%} | f1: {[f'{f1[j]:.2%}' for j in range(f1.shape[0])]} | f1_avg: {f1_mean:.4f} | AUC: {AUC} \n"
            f"MAE: {mae_total:.2f} | R2: {r2:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} \n"
            f"Sparsity: {sparsity_epoch}  |  {diversity_log}"
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
                "sparsity": sparsity_epoch,
                "diversity": diversity,

            }
            # save dic as csv
            pd.DataFrame.from_dict(epoch_log_dic, orient='index').to_csv(
                os.path.join(path_to_csv, f"e{epoch:02d}_metrics.csv"))

        # ########################## Logging epoch information on Wandb ###############################
        if self.config["wandb_mode"] != "disabled":
            epoch_log_dict = {
                # mode is 'val', 'val_push', or 'train
                f"epoch": epoch,
                # ######################## Loss Values #######################
                f"epoch/{mode}/loss_all": total_loss.sum(),
                # ######################## Eval metrics #######################
                
                f"epoch/{mode}/MAE": mae_total,
                f"epoch/{mode}/R2": r2,
                f"epoch/{mode}/MSE": mse,
                f"epoch/{mode}/RMSE": rmse,
                f"epoch/{mode}/diversity": diversity,
                f"epoch/{mode}/sparsity": sparsity_epoch,
            }
            if self.config["abstain_class"]:
                epoch_log_dict.update({f"epoch/{mode}/diversity_abstain": diversity_abstain})
            self.log_lr(epoch_log_dict)
            # log f1 scores separately
            loss_names = [ #ce, mse, mae, cluster, psd, ortho, om_l2, om_trns, fc_l1
                "loss_CE",
                "loss_MSE",
                "loss_MAE",
                "loss_Clst",
                "loss_PSD",
                "loss_Decor",
                "loss_Ortho",
                "loss_RoiNorm",
                "loss_RoiTrans",
                "loss_fcL1Norm",
            ]
            epoch_log_dict.update(
                {f"epoch/{mode}/{loss_name}": value for loss_name, value in zip(loss_names, total_loss)}
            )
            # logging all information
            wandb.log(epoch_log_dict)
        return mae_total, r2, mse, rmse

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
        num_class_prototypes = 40

        n_batches = 0
        total_loss = np.zeros(10)  # ce, mse, mae, cluster, psd, ortho, om_l2, om_trns, fc_l1

        y_pred_all = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        epoch_pred_log_df = pd.DataFrame()

        start = time.time()

        # Diversity Metric
        count_array = np.zeros(self.model.prototype_shape[0])
        simscore_cumsum = torch.zeros(self.model.prototype_shape[0])

        # Reset sparsity metric
        getattr(self, f"{mode.lower()}_sparsity_80").reset()

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
                LV_masks = data_sample["lv_mask"].to(self.device)
                
                logits = torch.zeros(input.shape[0], input.shape[1]).to(self.device)
                similarities_all = torch.zeros(input.shape[0], input.shape[1], self.model.prototype_shape[0]).to(self.device)
                for clip in range(input.shape[1]):
                    vid = input[:, clip].squeeze(1) # shape: (batch, 1, T, H, W) -> (batch, T, H, W)
                    logit, similarities, occurrence_map, beta = self.model(vid)
                    logits[:, clip] = logit.view(-1)
                    similarities_all[:, clip] = similarities
                logit = logits.mean(dim=1).view(-1) # shape (batch, 1) -> (batch)
                similarities = similarities_all.mean(dim=1) # doesnt make sense to average similarities
                ####### evaluation statistics ##########
                y_pred_all = torch.concat([y_pred_all, logit.cpu().detach()])
                y_true = target.detach().cpu()
                y_true_all = torch.concat([y_true_all, y_true])
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

                sparsity_batch = getattr(self, f"{mode.lower()}_sparsity_80")(similarities).item()

                # Determine the top 5 most similar prototypes to data
                # sort similarities in descending order
                sorted_similarities, sorted_indices = torch.sort(similarities[:, :num_class_prototypes].detach().cpu(),
                                                                 descending=True)
                # Add the type 5 most similar prototypes to the count array
                np.add.at(count_array[:num_class_prototypes], sorted_indices[:, :5], 1)

                if self.config["abstain_class"]:
                    # sort similarities in descending order
                    sorted_similarities, sorted_indices = torch.sort(
                        similarities[:, num_class_prototypes:].detach().cpu(), descending=True
                    )
                    # Add the type 5 most similar prototypes to the count array
                    np.add.at(count_array[num_class_prototypes:], sorted_indices[:, :2], 1)

                simscore_cumsum += similarities.sum(dim=0).detach().cpu()

                # ########################## Logging batch information on console ###############################
                # cm_flattened = [list(cm[j].flatten()) for j in range(cm.shape[0])]
                iterator.set_description(
                    f"Epoch: {epoch} | {mode} | "
                    f"MAE: {mae_batch:.2f} | "
                    f"R2: {r2_batch:.2f} | "
                    f"MSE: {mse_batch:.2f} | " 
                    f"RMSE: {rmse_batch:.2f} | "
                    #f"Acc: {accu_batch:.2%} | f1: {f1_batch.mean():.2f} |"
                    f"Sparsity: {sparsity_batch:.1f}",
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
                            f"batch_{mode}/sparsity": sparsity_batch,
                        }
                    )
                    #batch_log_dict.update(
                    #    {f"batch_{mode}/r2_{as_label}": value for as_label, value in zip(label_names, r2_batch)}
                    #)
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

        y_pred_all = y_pred_all.numpy()
        y_true_all = y_true_all.numpy()

        ### loss
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
        
        ### Diversity Metric Calculations
        # count how many prototypes were activated in at least 1% of the samples
        div_threshold = 0.05
        diversity = np.sum(count_array[:num_class_prototypes] > div_threshold * len(y_true_all))
        diversity_log = f"diversity: {diversity}"
        if self.config["abstain_class"]:
            diversity_abstain = np.sum(count_array[num_class_prototypes:] > div_threshold * len(y_true_all))
            diversity_log += f" | diversity_abstain: {diversity_abstain}"
        sorted_simscore_cumsum, sorted_indices = torch.sort(simscore_cumsum, descending=True)
        logging.info(f"sorted_simscore_cumsum is {sorted_simscore_cumsum}")

        sparsity_epoch = getattr(self, f"{mode.lower()}_sparsity_80").compute().item()
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
            f"Epoch:{epoch}_{mode} | Time:{end - start:.0f} | Total_Loss:{total_loss.sum() :.3f} | " #TODO: double check the sequence of losses
            f"MAE: {mae_total:.2f} | R2: {r2:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | f1<40: {f1:.2f}| mae: {[f'{mae[j]:.2f}' for j in range(mae.shape[0])]} \n"
            f"Sparsity: {sparsity_epoch}  |  Diversity: {diversity_log}"
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
                "sparsity": sparsity_epoch,
                "diversity": diversity,

            }
            # save dic as csv
            pd.DataFrame.from_dict(epoch_log_dic, orient='index').to_csv(
                os.path.join(path_to_csv, f"e{epoch:02d}_metrics.csv"))
                
        # ########################## Logging epoch information on Wandb ###############################
        if self.config["wandb_mode"] != "disabled":
            epoch_log_dict = {
                # mode is 'val', 'val_push', or 'train
                f"epoch": epoch,
                # ######################## Loss Values #######################
                f"epoch/{mode}/loss_all": total_loss.sum(),
                # ######################## Eval metrics #######################
                f"epoch/{mode}/f1<40": f1,
                f"epoch/{mode}/MAE": mae_total,
                f"epoch/{mode}/R2": r2,
                f"epoch/{mode}/MSE": mse,
                f"epoch/{mode}/RMSE": rmse,
                f"epoch/{mode}/diversity": diversity,
                f"epoch/{mode}/sparsity": sparsity_epoch,
            }
            if self.config["abstain_class"]:
                epoch_log_dict.update({f"epoch/{mode}/diversity_abstain": diversity_abstain})
            self.log_lr(epoch_log_dict)
            # logging all information
            wandb.log(epoch_log_dict)
        return mae_total, r2, mse, rmse
    
    def print_model_summary(self):
        img_size = self.data_config["img_size"]
        frames = self.data_config["frames"]
        summary(self.model, (3, frames, img_size, img_size), device="cpu")

    def run_extract_features(self):
        # save the classifier layer
        torch.save(self.model.last_layer.state_dict(), os.path.join(self.config["save_dir"], "regression_layer.pth"))

        train_features, train_similarities, train_metadata = self.extract_features(mode="TRAIN", epoch=self.current_epoch)
        val_features, val_similarities, val_metadata = self.extract_features(mode="VAL", epoch=self.current_epoch)
        test_features, test_similarities, test_metadata = self.extract_features(mode="TEST", epoch=self.current_epoch)

        prototypes = self.model.prototype_vectors.squeeze()  # shape (P, D)
        prototype_labels = self.model.proto_classes  # shape (P,)
        torch.save(prototypes, os.path.join(self.config["save_dir"], f"e{self.current_epoch:02d}_prototypes.pt"))
        torch.save(prototype_labels, os.path.join(self.config["save_dir"], f"e{self.current_epoch:02d}_prototype_labels.pt"))
        
        train_labels = train_metadata["target_EF"].to_numpy()
        test_labels = test_metadata["target_EF"].to_numpy()
        val_labels = val_metadata["target_EF"].to_numpy()

        logging.info("Loaded all features and metadata \n")
        logging.info(f"Train Features:{train_features.shape} | Test Features:{test_features.shape}| Val Features:{val_features.shape} | " #TODO: double check the sequence of losses
                    f"Train Similarities:{train_similarities.shape} | Test Similarities:{test_similarities.shape} | Val Similarities:{val_similarities.shape} |"
                    f"Train labels:{train_labels.shape} | Test labels:{test_labels.shape} | Val labels:{val_labels.shape} |"
                    f"Prototypes:{prototypes.shape} | Prototype Labels:{prototype_labels.shape}")
    
        # ############# UMAP ###############
        # make path dir for current epoch
        plot_embeddings(prototypes, prototype_labels, self.config["save_dir"], embed_type='umap', dim = '2D', k=100, train_points=train_features, test_points=test_features, train_labels=train_labels, test_labels=test_labels, train_similarities=train_similarities, test_similarities=test_similarities, test_type='TEST')
        plot_embeddings(prototypes, prototype_labels, self.config["save_dir"], embed_type='PCA', dim = '2D', k=100, train_points=train_features, test_points=test_features, train_labels=train_labels, test_labels=test_labels, train_similarities=train_similarities, test_similarities=test_similarities, test_type='TEST')
        plot_embeddings(prototypes, prototype_labels, self.config["save_dir"], embed_type='PCA', dim = '2D', k=100, train_points=train_features, test_points=val_features, train_labels=train_labels, test_labels=val_labels, train_similarities=train_similarities, test_similarities=val_similarities, test_type='VAL')

    def extract_features(self, epoch, mode="TRAIN"):
        
        logging.info(f"Starting {mode}")
        if mode == "TRAIN":
            self.model.train()
        else:
            self.model.eval()

        data_loader = self.data_loaders[mode]

        #label_names = class_labels[self.data_config["label_scheme_name"]]
        #logit_names = label_names + ["abstain"] if self.config["abstain_class"] else label_names

        epoch_pred_log_df = pd.DataFrame()
        features_extracted_list = torch.tensor([])
        similarity_list = torch.tensor([])

        with torch.set_grad_enabled(False):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            for i in iterator:
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)
                if mode == "TRAIN" or mode == "VAL": # single clip
                    logit, similarity, features_extracted = self.model.extract_features(input)
                else: # multiple clips then choose only the second clip          
                    if input.shape[1] > 1:  
                        vid = input[:, 1].squeeze(1)  # Select the second clip
                    else:  
                        vid = input[:, 0].squeeze(1)  # Only one clip available, select the first one
                    logit, similarity, features_extracted = self.model.extract_features(vid)
                logit = logit.view(-1) # shape (batch, 1) -> (batch)

                # Concatenate extracted features and similarities
                features_extracted_list = torch.cat((features_extracted_list, features_extracted.detach().cpu()))
                similarity_list = torch.cat((similarity_list, similarity.detach().cpu()))

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

                iterator.set_description(f"Mode: {mode} | ", refresh=True,)

        #################################################################################
        ################################### CSV Log #####################################
        #################################################################################
        torch.save(features_extracted_list, os.path.join(self.config["save_dir"], f"e{epoch}_clips_{mode}.pt"))
        torch.save(similarity_list, os.path.join(self.config["save_dir"], f"e{epoch}_similarities_{mode}.pt"))

        epoch_pred_log_df.reset_index(drop=True).to_csv(os.path.join(self.config["save_dir"], f"e{epoch}_clips_{mode}.csv"))
        metadata = epoch_pred_log_df.reset_index(drop=True)
        return features_extracted_list, similarity_list, metadata