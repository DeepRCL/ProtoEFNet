#%%
import torch.nn as nn
import torch
from torchvision.ops import sigmoid_focal_loss
from torchvision.transforms.functional import affine, InterpolationMode
import random
import logging
import matplotlib.pyplot as plt

class MSE(object):  # TODO Regression formulation
    def __init__(self, loss_weight=1, reduction="mean"):
        self.criterion = nn.MSELoss(reduction=reduction)
        self.loss_weight = loss_weight
        logging.info(f"setup MSE Loss with loss_weight:{loss_weight}, and reduction:{reduction}")

    def compute(self, pred, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        loss = self.criterion(input=pred, target=target)  # (Num_nodes, 1)
        return self.loss_weight * loss

class MAE(object):  # TODO Regression formulation
    def __init__(self, loss_weight=1, reduction="mean"):
        self.criterion = nn.L1Loss(reduction=reduction)
        self.loss_weight = loss_weight
        logging.info(f"setup MAE Loss with loss_weight:{loss_weight}, and reduction:{reduction}")

    def compute(self, pred, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        loss = self.criterion(input=pred, target=target)  # (Num_nodes, 1)
        return self.loss_weight * loss


class CeLoss(object):
    ''' Cross-entropy loss for multi-class classification'''
    def __init__(self, loss_weight=1, reduction="mean"):
        # TODO maybe add weight parameter if sampler is random in dataset
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)  # , weight=tbd)
        self.loss_weight = loss_weight
        logging.info(f"setup CE Loss with loss_weight:{loss_weight}, and reduction:{reduction}")

    def compute(self, logits, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        loss = self.criterion(input=logits, target=target)  # (Num_nodes, num_classes)
        return self.loss_weight * loss


# modified sigmoid_focal_loss for multilabel
class FocalLoss(object):
    def __init__(self, loss_weight=1, gamma=2, reduction="sum"):
        # self.alpha = num_neg / (num_neg+num_pos)
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.reduction = reduction
        logging.info(f"setup Focal Loss with loss_weight:{loss_weight}, gamma:{gamma} and reduction:{reduction}")

    def compute(self, pred, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        p = torch.sigmoid(pred)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        self.alpha = self.alpha.to(pred.device)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # alpha_t = (1 / num_pos.to(pred.device)) * target + (1 / num_neg.to(pred.device)) * (1 - target)
        loss = alpha_t * loss  # shape (N, 2)

        if self.reduction == "mean":
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss


class ClusterPatch(object):
    """
    Cluster cost based on ProtoPNet architecture, using distance between patches and prototypes
    """

    def __init__(self, loss_weight, num_classes=4, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        logging.info(
            f"setup Path-Based Cluster Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, "
            f"and reduction:{reduction}"
        )

    def compute(self, min_distances, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)  # shape (N, classes)
        min_distances = min_distances.reshape((min_distances.shape[0], self.num_classes, -1))
        class_specific_min_distances, _ = min_distances.min(dim=2)  # Shape = (N, classes)
        positives = class_specific_min_distances * target_one_hot  # shape (N, classes)

        if self.reduction == "mean":
            loss = positives.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = positives.sum()

        return self.loss_weight * loss


class SeparationPatch(object):
    """
    Cluster cost based on ProtoPNet architecture, using distance between patches and prototypes
    """

    def __init__(self, loss_weight, num_classes=4, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        logging.info(
            f"setup Patch-Based Separation Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, "
            f"and reduction:{reduction}"
        )

    def compute(self, min_distances, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)
        min_distances = min_distances.reshape((min_distances.shape[0], self.num_classes, -1))
        class_specific_min_distances, _ = min_distances.min(dim=2)  # Shape = (N, classes)
        negatives = class_specific_min_distances * (1 - target_one_hot)  # shape (N, classes)
       

        if self.reduction == "mean":
            loss = -negatives.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = -negatives.sum()

        return self.loss_weight * loss


class ClusterRoiFeat(object):
    """
    Cluster cost based on XprotoNet architecture, using similarities between ROI features and prototypes
    """

    def __init__(self, loss_weight, num_classes=4, k=1, reduction="sum"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.k = k
        logging.info(
            f"setup ROI-Based Cluster Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, using top{k} most similar prototypes,"
            f"and reduction:{reduction}"
        )

    def compute(self, similarities, target):
        """
        compute loss given the similarity scores
        :param similarities: the cosine similarities calculated. shape (N, P). P=num_classes x numPrototypes
        :param target: labels, with shape of (N)
        :return: cluster loss using the similarities between the ROI features and prototypes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        # turning labels into one hot
        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)
        # reshaping similarities to group based on class they belong to.
        similarities = similarities.reshape((similarities.shape[0], self.num_classes, -1))
        # get largest prototype-ROIfeature similarity scores per class
        if self.k == 1:
            class_specific_max_similarity, _ = similarities.max(dim=2)
        else:
            # top k maximum similarities contribute
            class_specific_max_similarity, _ = similarities.topk(self.k, dim=2)
            class_specific_max_similarity = class_specific_max_similarity.mean(dim=2)
          
        # pick similarity scores of classes the input belongs to
        positives = class_specific_max_similarity * target_one_hot
        loss = -1 * positives  # loss is negative sign of similarity scores
        # aggregate loss values
        if self.reduction == "mean":  # average across batch size
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss


class SeparationRoiFeat(object):
    """
    Separation cost based on XProtoNet paper, suitable for multi-label data
    """

    def __init__(self, loss_weight, num_classes=4, k=1, reduction="sum", abstain_class=True):
        self.num_classes = num_classes
        self.k = k
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.abstain_class = abstain_class
        logging.info(
            f"setup ROI-Based Separation Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, using top{k} most similar prototypes, "
            f"and reduction:{reduction}"
        )

    def compute(self, similarities, target):
        """
        compute loss given the similarity scores
        :param similarities: the cosine similarities calculated. shape (N, P). P=num_classes x numPrototypes
        :param target: labels, with shape of (N)
        :return: separation loss using the similarities between the ROI features and prototypes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        # turning labels into one hot
        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)
        # reshaping similarities to group based on class they belong to.
        similarities = similarities.reshape((similarities.shape[0], self.num_classes, -1))
        # get largest prototype-ROIfeature similarity scores per class
        if self.k == 1:
            class_specific_max_similarity, _ = similarities.max(dim=2)
        else:
            # top k maximum similarities contribute
            class_specific_max_similarity, _ = similarities.topk(self.k, dim=2)
            class_specific_max_similarity = class_specific_max_similarity.mean(dim=2)        
        
        # pick similarity scores of classes the input belongs to
        negatives = class_specific_max_similarity * (1 - target_one_hot)
        loss = negatives
        
        # aggregate loss values
        if self.reduction == "mean":  # average across batch size
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss


class OrthogonalityLoss(object):
    """
    orthogonality loss to encourage diversity in learned prototype vectors
    """

    def __init__(self, loss_weight, num_classes=4, mode="per_class"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.mode = mode  # one of 'per_class' or 'all'
        if mode == "per_class":
            self.cosine_similarity = nn.CosineSimilarity(dim=3)
        elif mode == "all":
            self.cosine_similarity = nn.CosineSimilarity(dim=2)
        logging.info(
            f"setup Orthogonality Loss with loss_weight:{loss_weight}, "
            f"for num_classes:{num_classes}, in {mode} mode"
        )

    def compute(self, prototype_vectors):
        """
        compute loss given the prototype_vectors
        :param prototype_vectors: shape (P, channel, 1, 1). P=num_classes x numPrototypes
        :return: orthogonality loss either across each class, summed (or averaged), or across all classes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=prototype_vectors.device)

        # per class diversity
        if self.mode == "per_class":
            # reshape to (num_classes, num_prot_per_class, channel):
            prototype_vectors = prototype_vectors.reshape(self.num_classes, -1, prototype_vectors.shape[1])
            # shape of similarity matrix is (num_classes, num_prot_per_class, num_prot_per_class)
            sim_matrix = self.cosine_similarity(prototype_vectors.unsqueeze(1), prototype_vectors.unsqueeze(2))
        elif self.mode == "all":
            # shape of similarity matrix is (num_prot_per_class, num_prot_per_class)
            sim_matrix = self.cosine_similarity(
                prototype_vectors.squeeze().unsqueeze(1),
                prototype_vectors.squeeze().unsqueeze(0),
            )
        # use upper traingle elements of similarity matrix (excluding main diagonal)
        loss = torch.triu(sim_matrix, diagonal=1).sum()

        return self.loss_weight * loss


class L_norm(object):
    def __init__(self, p=1, loss_weight=1e-4, mask = None, reduction="sum"):
        self.mask = mask  # mask determines which elements of tensor to be used for Lnorm calculations is used for fc norm
        self.p = p
        self.loss_weight = loss_weight
        self.reduction = reduction
        
        logging.info(f"setup L{p}-Norm Loss with loss_weight:{loss_weight}, with reduction:{reduction}")

    def compute(self, tensor,  mask_tensor=None, dim=None):
    
        if self.loss_weight == 0:
            return torch.tensor(0, device=tensor.device)

        if mask_tensor != None:
            loss = (mask_tensor * tensor).norm(p=self.p, dim=dim)
        else:
            if self.mask != None:
                loss = (self.mask * tensor).norm(p=self.p, dim=dim)
            else:
                loss = tensor.norm(p=self.p, dim=dim)

        if self.reduction == "mean":
            loss = loss.mean(dim=0).sum() # mean across batch data, sum across prototypes
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss

class L2_LVloss(object):
    '''
    L2 loss between the occurrence map and the LV mask
    '''
    def __init__(self, loss_weight=0.6, reduction="mean"):
        self.criterion = nn.MSELoss(reduction=reduction) #L2 loss
        self.loss_weight = loss_weight
        self.reduction = reduction
        logging.info(f"setup L2-LV Loss with loss_weight:{loss_weight}, with reduction:{reduction}")

    def compute(self, occurrence_map, LV_mask): 
        # 1 is # clips: occurance_map: (N, P, 1, T, H, W), LV_mask: (N, 1, H, W)
        # [N, 40, 1, 8, 14, 14].
        if self.loss_weight == 0:
            return torch.tensor(0, device=occurrence_map.device)

        # compute L2 loss
        # LV mask must be broadcasted to the same shape as occurrence_map
        LV_mask = LV_mask.unsqueeze(1).unsqueeze(3).expand_as(occurrence_map)
        loss = self.criterion(occurrence_map, LV_mask)
        
        return self.loss_weight * loss


def get_affine_config():
    config = {
        "angle": random.uniform(-20, 20),
        "translate": (
            0,
            0,
        ),
        "scale": random.uniform(0.6, 1.5),
        "shear": 0.0,
        "fill": 0,
        "interpolation": InterpolationMode.BILINEAR,
    }
    return config


class TransformLoss(object):
    """
    the loss applied on generated ROIs!
    """

    def __init__(self, loss_weight=1e-4, reduction="sum"):
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction="sum")
        self.reduction = reduction
        logging.info(f"setup Transformation Loss with loss_weight:{loss_weight}, with reduction:{reduction}")

    def compute(self, x, occurrence_map, model):
        if self.loss_weight == 0:
            return torch.tensor(0, device=x.device)

        if len(x.shape) == 5: 
            video_based = True
        else:
            video_based = False

        # get the affine transform randomly sampled configuration
        config = get_affine_config()

        # transform input and get its new occurrence map
        if video_based: 
            N, D, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, D, H, W)  # shape (NxT, D, H, W)
        transformed_x = affine(x, **config)  # shape (NxT, D, H, W), T is 1 for image-based
        if video_based:
            transformed_x = transformed_x.reshape(N, T, D, H, W).permute(0, 2, 1, 3, 4)  # shape (N, D, T, H, W)
        occurrence_map_transformed = model.compute_occurence_map(transformed_x).squeeze(2)  # shape (N, P, (T), H, W)

        # transform initial occurence map
        occurrence_map = occurrence_map.squeeze(2)  # shape (N, P, H, W) or (N, P, T, H, W) for video-based
        if video_based: 
            N, P, T, H, W = occurrence_map.shape
            occurrence_map = occurrence_map.permute(0, 2, 1, 3, 4).reshape(-1, P, H, W)  # shape (NxT, P, H, W)
        transformed_occurrence_map = affine(occurrence_map, **config)  # shape (NxT, P, H, W), T is 1 for image-based
        if video_based:
            transformed_occurrence_map = transformed_occurrence_map.reshape(N, T, P, H, W).permute(
                0, 2, 1, 3, 4
            )  # shape (N, P, T, H, W)

        # compute L1 loss
        loss = self.criterion(occurrence_map_transformed, transformed_occurrence_map)
        if self.reduction == "mean":
            loss = loss / (occurrence_map_transformed.shape[0] * occurrence_map_transformed.shape[1])

        return self.loss_weight * loss


class CeLossAbstain(object):
    """
    Cross-entropy-like loss. Introduces a K+1-th class, abstention, which is a
    learned estimate of aleatoric uncertainty. When the network abstains,
    the loss will be computed with the ground truth, but, the network incurs
    loss for using the abstension
    """

    def __init__(self, loss_weight=1, ab_weight=0.3, reduction="sum", ab_logitpath="joined"):
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.ab_weight = ab_weight
        self.ab_logitpath = ab_logitpath
        self.criterion = nn.NLLLoss(reduction=reduction)
        assert self.ab_logitpath == "joined" or self.ab_logitpath == "separate"
        logging.info(
            f"setup CE Abstain Loss with loss_weight:{loss_weight}, "
            + f"ab_penalty:{ab_weight}, ab_path:{ab_logitpath} and reduction:{reduction}"
        )

    def to(self, device):  # TODO why is this here? Do we ever use these?
        self.criterion = self.criterion.to(device)

    def compute(self, logits, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        B, K_including_abs = logits.shape  # (B, K+1)
        K = K_including_abs - 1
        assert K >= 2, "CeLossAbstain input must have >= 2 classes not including abstention"

        # virtual_pred = (1-alpha) * pred + alpha * target
        if self.ab_logitpath == "joined":
            abs_pred = logits.softmax(dim=1)[:, K : K + 1]  # (B, 1)
        elif self.ab_logitpath == "separate":
            abs_pred = logits.sigmoid()[:, K : K + 1]  # (B, 1)
        class_pred = logits[:, :K].softmax(dim=1)  # (B, K)
        target_oh = nn.functional.one_hot(target, num_classes=K)
        virtual_pred = (1 - abs_pred) * class_pred + abs_pred * target_oh  # (B, K)

        loss_pred = self.criterion(torch.log(virtual_pred), target)
        loss_abs = -torch.log(1 - abs_pred).squeeze()  # (B)

        if self.reduction == "mean":
            loss_abs = loss_abs.mean()
        elif self.reduction == "sum":
            loss_abs = loss_abs.sum()

        return self.loss_weight * (loss_pred + self.ab_weight * loss_abs)

class ClusterLoss_Regr(object):
    """ Class to compute the regression cluster loss 
    """
    def __init__(self, loss_weight, model, class_specific=True, delta=0.5, kmin=1):
        self.max_dist = 2.0 #model.prototype_shape[1] * model.prototype_shape[2] * model.prototype_shape[3]
        self.class_specific = class_specific
        self.delta = delta
        self.kmin = kmin
        self.loss_weight = loss_weight
        logging.info(f"ClusterLossRegr initialized with class_specific={class_specific}, delta={delta}, kmin={kmin}, loss_weight={loss_weight}")

    def compute(self, max_similarities, target, model):
        # proto_classID can change after protopush
        proto_classID = model.proto_classes.to(target.device)

        # pull closest prototype
        if not self.class_specific:
            max_sim, _ = torch.max(
                max_similarities, dim=1) # shape (N) each sample to the closest prototype
            cluster_cost = torch.mean(- max_sim)


        # pull closest prototype within distance
        else:
            # get prototypes within delta of image (based on label differenc)
            proto_correct_class = torch.stack(
                [torch.le(torch.abs(proto_classID - x), self.delta) for x in target])

            # Convert boolean to 0 and 1
            proto_correct_class = proto_correct_class.type_as(
                target.floor().long())

            # Convert to distances
            max_sim, _ = torch.topk((max_similarities)
                                               * proto_correct_class, k=self.kmin, dim=1) # shape (N, kmin) for each batch sample the k closest prototypes

            cluster_cost = torch.mean(- max_sim)

        return self.loss_weight * cluster_cost

class ProtoSampleDist(object):
    """ Class to compute the prototype sample distance loss
    """
    def __init__(self, loss_weight, model):
        self.max_dist = 2.0 #model.prototype_shape[1] * model.prototype_shape[2] * model.prototype_shape[3]
        self.loss_weight = loss_weight
        logging.info(f"ProtoSampleDist initialized with max_dist={self.max_dist}, loss_weight={loss_weight}")

    def compute(self, min_distances, target):
        # compute min distance from each prototype to a sample (to keep them close around)
        assert not torch.max(min_distances) > self.max_dist

        # dists is min_distance from each prototype to sample (irrespective of class)
        dists, _ = torch.min(min_distances, dim=0)
        dists_norm = dists / self.max_dist
        reg_cost = -torch.mean(torch.log(1-dists_norm))

        return self.loss_weight * reg_cost
    
class ProtoDecorelation(object):
    """ Class to compute the prototype decorrelation loss
    """
    def __init__(self, loss_weight,  class_specific=True, delta=0.5, kmin=1, eps=1e-6):
        self.loss_weight = loss_weight
        self.class_specific = class_specific
        self.delta = delta 
        self.kmin = kmin
        self.eps = eps # for numerical stability
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
        logging.info(f"ProtoDecorelation initialized with class_specific={self.class_specific}, delta={self.delta}, k_min={self.kmin}, loss_weight={loss_weight}")

    def compute(self, model):
        '''shape (P, channel, 1, 1, 1),
        '''
        # proto_classID can change after protopush
        proto_classID = model.proto_classes
        prototype_vectors = model.prototype_vectors

        # get the cosine similarity between prototype vectors
        similarities = self.cosine_similarity( prototype_vectors.squeeze().unsqueeze(1), 
                prototype_vectors.squeeze().unsqueeze(0)) # shape (P, channel, 1, 1, 1) -> (P, channel) -> (P, 1, channel)  * (1, P, Channel) -> (P, P)
        # get scaled cos sim known as angular distance
        cosine_sim = torch.clamp(similarities, -1 + self.eps, 1 - self.eps)  # Avoid instability in arccos, shape (P, P)

        # Compute angular distance using atan2 (more stable than acos)
        theta = torch.atan2(torch.sqrt(1 - cosine_sim**2), cosine_sim)  # theta = arccos(cosine_sim), shape (P, P)
        
        # Convert to angular similarity
        angular_sim = 1 - (theta / torch.pi) # shape (P, P)
        # get the top triangle of the matrix (excluding diagonal) No this messes up the comparison
        # push away closest prototypes
        if not self.class_specific: 
            # implement similar to else condition, but we have a predifined classes with a range of delta
            # first class/bin starts from the first classID, and we push away the closest prototypes outside the bin
            # get the max similarity for each prototype
            max_similarities, _ = torch.max(torch.log(1.0-angular_sim), dim=1) # shape (P) for each prototype the closest prototype, could add top k for later
            decor_cost = torch.mean(- max_similarities) # shape scaler: push away closest prototypes 

        # push away closest prototypes within distance delta
        else:
            # get prototypes within delta of each other and ignore the prototype itself: it must be false if it is equal to 0
            proto_correct_class_list = []
            for i, x in enumerate(proto_classID):
                row = torch.gt(torch.abs(proto_classID - x), self.delta) #& (torch.abs(proto_classID - x) != 0)
                # ensure diagonal element is false
                row[i] = False
                proto_correct_class_list.append(row)
            proto_correct_class = torch.stack(proto_correct_class_list)

            # Convert boolean to 0 and 1
            proto_correct_class = proto_correct_class.type_as(
                proto_classID.floor().long())

            # Convert to distances, torch.log is ln(1-angular_sim)
            sims_logs = torch.log(1.0-angular_sim+ self.eps) * proto_correct_class
            max_similarities = torch.mean(
                sims_logs,
                dim=1
            )
            decor_cost = torch.mean(- max_similarities) # shape scaler: push away closest prototypes given delta

        return self.loss_weight * decor_cost
