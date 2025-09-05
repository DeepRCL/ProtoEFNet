import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.receptive_field import compute_proto_layer_rf_info_v2 # Not used here. TODO: maybe check what it does
from src.models.ProtoPNet import PPNet, base_architecture_to_features # Base of XProtoNet is protoPNet


class Video_protoEF(PPNet):
    def __init__(
        self, cnn_backbone, img_size, prototype_shape, proto_layer_rf_info, similarity_metric, prototype_activation_function="linear", num_classes=1, proto_minrange=10,
        proto_maxrange=90, init_weights=True, init_ll='ones', fc_method="weighted_sum", tau=1.0, **kwargs
    ):
        super(PPNet, self).__init__()  # not calling init of PPNet and directly going to its parent!

        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.proto_minrange, self.proto_maxrange = proto_minrange, proto_maxrange
        self.similarity_metric = similarity_metric
        self.prototype_activation_function = prototype_activation_function
        self.num_classes = num_classes
        self.prototype_class_identity = self.get_prototype_class_identity()
        self.proto_layer_rf_info = proto_layer_rf_info  
        self.init_ll = init_ll
        self.fc_method = fc_method # or soft attention
        self.tau = tau
        self.epsilon = 1e-4

        # CNN Backbone module
        self.cnn_backbone = cnn_backbone
        cnn_backbone_out_channels = self.get_cnn_backbone_out_channels(self.cnn_backbone)

        # feature extractor module on top of resnet
        self.add_on_layers = nn.Sequential(
            nn.Conv3d(
                in_channels=cnn_backbone_out_channels,
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.prototype_shape[1],
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
            #nn.Sigmoid() # Yeg added to Xprotonet
        )

        # Occurrence map module
        self.occurrence_module = nn.Sequential(
            nn.Conv3d(
                in_channels=cnn_backbone_out_channels,
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.prototype_shape[1],
                out_channels=self.prototype_shape[1] // 2,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.prototype_shape[1] // 2,
                out_channels=self.prototype_shape[0],
                kernel_size=1,
                bias=False,
            ),
            nn.Sigmoid()
        )

        self.om_softmax = nn.Softmax(dim=-1)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
        self.euclidean_distance = nn.PairwiseDistance(p=2, keepdim=True)

        # Learnable prototypes
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
         # initialise the prototype classes
        proto_classes = torch.linspace(self.proto_minrange, self.proto_maxrange, self.num_prototypes, requires_grad=False) # require grad is false by default
        self.register_buffer("proto_classes", proto_classes) # call this using self.proto_classes

       
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)  # do not use bias

        if init_weights:
            self._initialize_weights(self.add_on_layers)
            self._initialize_weights(self.occurrence_module)
            self.set_last_layer_connection()
        
    def forward(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        feature_map = self.add_on_layers(x).unsqueeze(1)
        occurrence_map = self.get_occurence_map_absolute_val(x)

        # occurance_map * feature map has shape ([N, P, D=256, T=8, H=14, W=14])
        if self.similarity_metric == "cosine":
            features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3).sum(dim=3)  # shape (N, P, D)
            similarity = self.cosine_similarity(
                features_extracted, self.prototype_vectors.squeeze().unsqueeze(0) #shape of prototype (P, D) -> (1, P, D)
            )  
        elif self.similarity_metric == "l2":
            features_extracted = (occurrence_map * feature_map) # shape ([N, P, D=256, T=8, H=14, W=14]) but must be 5D for L2 conv
            distances = self.euclidean_distance(features_extracted.sum(dim=3).sum(dim=3).sum(dim=3) 
                                                , self.prototype_vectors.squeeze().unsqueeze(0)) # shape (N, P, D)
            # global min pooling
            min_distances = -F.max_pool1d(-distances ** 2,
                                        kernel_size=(distances.size()[2],)) # N, P 
                                                    


            min_distances = min_distances.view(-1, self.num_prototypes)
            similarity = self.distance_2_similarity(min_distances)
        ######## Get the final prediction as the sum of weighted mean of all prototypes ########
        # Prototype Layer
        class_IDx = torch.unsqueeze(self.proto_classes, dim=0).to(x.device)
        # assert class IDx does not contain 0's
        assert class_IDx.nonzero(as_tuple = False).size()[0] == class_IDx.shape[1], "Network class IDx contains zero's which causes division by 0"

        if self.fc_method == "weighted_sum":
            
            # multiply by class labels (to get final weighted mean prediction) # shouldnt this not be square? TODO
            ll_noclass = self.last_layer.weight.square() / class_IDx # theta / l element wise. (1,p) / (1,p) = (1,p)

            # Weights consist of everything except class_IDx
            sum_of_weights = torch.sum(
                similarity * ll_noclass, dim=1) # B

            # apply squired weights
            logits = self.last_layer(similarity) # B
            # Final prediction is divided by sum of weights
            logits = logits / torch.unsqueeze(sum_of_weights, 1) 

        elif self.fc_method == "soft_attention":
            # Compute soft weights using negative exponentiation
            beta = nn.functional.softmax((similarity * self.last_layer.weight.squeeze(0)) / self.tau, dim=1)  # shape (B, P), row sum is 1
            # Compute weighted sum, weighted aggregation
            logits = torch.matmul(beta, class_IDx.T)  # (batch_size, 1)
            return logits, similarity, occurrence_map, beta
        
        return logits, similarity, occurrence_map, None

    def extract_features(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)  # shape (N, 512 or 256, T, H, W) It is 256 for resnet18
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, D, T, H, W)
        occurrence_map = self.get_occurence_map_absolute_val(x)  # shape (N, P, 1, T, H, W) 


        # Prototype Layer
        if self.similarity_metric == "cosine":
            features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3).sum(dim=3)  # shape (N, P, D)
            similarity = self.cosine_similarity(
                features_extracted, self.prototype_vectors.squeeze().unsqueeze(0) #shape of prototype (P, D) -> (1, P, D)
            ) 
        elif self.similarity_metric == "l2":
            features_extracted = (occurrence_map * feature_map) # shape ([N, P, D=256, T=8, H=14, W=14]) but must be 5D for L2 conv
            distances = self.euclidean_distance(features_extracted.sum(dim=3).sum(dim=3).sum(dim=3) 
                                                , self.prototype_vectors.squeeze().unsqueeze(0)) # shape (N, P, D)
          
            # global min pooling
            min_distances = -F.max_pool1d(-distances ** 2,
                                        kernel_size=(distances.size()[2],)) # N, P 

            min_distances = min_distances.view(-1, self.num_prototypes)
            similarity = self.distance_2_similarity(min_distances)

        ######## Get the final prediction as the sum of weighted mean of all prototypes ########
        # Prototype Layer
        class_IDx = torch.unsqueeze(self.proto_classes, dim=0).to(x.device)
        # assert class IDx does not contain 0's
        assert class_IDx.nonzero(as_tuple = False).size()[0] == class_IDx.shape[1], "Network class IDx contains zero's which causes division by 0"

        if self.fc_method == "weighted_sum":
            
            # multiply by class labels (to get final weighted mean prediction) # shouldnt this not be square? 
            ll_noclass = self.last_layer.weight.square() / class_IDx 

            # Weights consist of everything except class_IDx
            sum_of_weights = torch.sum(
                similarity * ll_noclass, dim=1) # B

            # apply squired weights
            logits = self.last_layer(similarity) # B
            # Final prediction is divided by sum of weights
            logits = logits / torch.unsqueeze(sum_of_weights, 1) 

        elif self.fc_method == "soft_attention":
            # Compute soft weights using negative exponentiation
            beta = nn.functional.softmax((similarity * self.last_layer.weight.squeeze(0)) / self.tau, dim=1)  # shape (B, P), row sum is 1
            # Compute weighted sum, weighted aggregation
            logits = torch.matmul(beta, class_IDx.T)  # (batch_size, 1)
            return logits, similarity, features_extracted #, beta
        
        return logits, similarity, features_extracted #, None

    def compute_occurence_map(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        occurrence_map = self.get_occurence_map_absolute_val(x)  # shape (N, P, 1, T, H, W)
        return occurrence_map

    def get_occurence_map_softmaxed(self, x):
        occurrence_map = self.occurrence_module(x)  # shape (N, P, H, W)
        n, p, h, w = occurrence_map.shape
        occurrence_map = occurrence_map.reshape((n, p, -1))
        occurrence_map = (
            self.om_softmax(occurrence_map).reshape((n, p, h, w)).unsqueeze(2)
        )  # shape (N, P, 1, H, W) #TODO CHECK
        return occurrence_map

    def get_occurence_map_absolute_val(self, x):
        occurrence_map = self.occurrence_module(x)  # shape (N, P, T, H, W)
        occurrence_map = torch.abs(occurrence_map).unsqueeze(2)  # shape (N, P, 1, T, H, W)
        return occurrence_map

    def push_forward(self, x):
        """
        this method is needed for the pushing operation
        """
        # Feature Extractor Layer
        x = self.cnn_backbone(x)  # shape (N, 512 or 256, T, H, W)
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, D, T, H, W)
        occurrence_map = self.get_occurence_map_absolute_val(x)  # shape (N, P, 1, T, H, W)
        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3).sum(dim=3)  # shape (N, P, D)

        # Prototype Layer
        similarity = self.cosine_similarity(
            features_extracted, self.prototype_vectors.squeeze().unsqueeze(0)
        )  # shape (N, P)

        ######## Get the final prediction as the sum of weighted mean of all prototypes ########
        # Prototype Layer
        class_IDx = torch.unsqueeze(self.proto_classes, dim=0).to(x.device)
        # assert class IDx does not contain 0's
        assert class_IDx.nonzero(as_tuple = False).size()[0] == class_IDx.shape[1], "Network class IDx contains zero's which causes division by 0"

        if self.fc_method == "weighted_sum":
            
            # multiply by class labels (to get final weighted mean prediction) # shouldnt this not be square? TODO
            ll_noclass = self.last_layer.weight.square() / class_IDx # theta / l element wise. (1,p) / (1,p) = (1,p)

            # Weights consist of everything except class_IDx
            sum_of_weights = torch.sum(
                similarity * ll_noclass, dim=1) # B

            # apply squired weights
            logits = self.last_layer(similarity) # B
            # Final prediction is divided by sum of weights
            logits = logits / torch.unsqueeze(sum_of_weights, 1) 

        elif self.fc_method == "soft_attention":
            # Compute soft weights using negative exponentiation
            beta = nn.functional.softmax((similarity * self.last_layer.weight.squeeze(0)) / self.tau, dim=1)  # shape (B, P), row sum is 1
            # Compute weighted sum, weighted aggregation
            logits = torch.matmul(beta, class_IDx.T)  # (batch_size, 1)
            #return logits, similarity, occurrence_map, beta
        return features_extracted, 1 - similarity, occurrence_map, logits
    
    def _l2_convolution(self, x):
        """
        apply self.prototype_vectors as l2-convolution filters on input x for video input is extracted features

         apply self.prototype_vectors as l2-convolution filters on input x
        implemented from: https://ieeexplore.ieee.org/abstract/document/8167877
        (eq. 17 with S =self.ones, M = prototypes, X = input)

        """
        x2 = x ** 2 # shape (N, P, D, T=8, H=14, W=14) , ones shape is (P=10, D=256, T=1, H=1, W=1)
        #x2_patch_sum = F.conv3d(input=x2, weight=self.ones) # shape (N, P, 1, 1, 1)
        # this conv is identical to sum over patches: T, H, W
        x2_patch_sum = torch.sum(x2, dim=(2, 3, 4, 5), keepdim=True)  # shape (N, P, D, 1, 1, 1)

        p2 = self.prototype_vectors ** 2 # shape (P, D, T=1, H=1, W=1)
        p2 = torch.sum(p2, dim=(1, 2, 3, 4))  # sum over D, T, H, W
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1, 1) p, t, H, W
        p2_reshape = p2.view(-1, 1, 1, 1)

        # get the sum of localtion features to lower dim #TODO: think if this makes sense
        x_reshape = torch.sum(x, dim=(4, 5), keepdim=True).squeeze(-1)  # shape (N, P, D, T=8, H=14)
        p_reshape = self.prototype_vectors.squeeze(-1)  # shape (P, D, T=1, H=1, W=1) -> (P, D, T=1, H=1)
        xp = F.conv3d(input=x, weight=self.prototype_vectors)  # shape (N, P, T=8, H=14 * W=14)
        print("xp:", xp.shape)
        intermediate_result = -2 * xp + p2_reshape  # use broadcast, shape (N, P, T=8, H=14 * W=14)
        print("intermediate_result:", intermediate_result.shape)
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances
    
    def prototype_distances(self, x): # extra TODO
        """
        x is the raw input video
        """
        conv_features = self.conv_features(x)  # shape (N, 512, T, H=7, W=7)
        distances = self._l2_convolution(conv_features)  # shape (N, P, D, T, 7, 7)
        return distances
    
    def __repr__(self):
        # XProtoNet(self, backbone, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            "PPNet(\n"
            "\tcnn_backbone: {},\n"
            "\timg_size: {},\n"
            "\tprototype_shape: {},\n"
            "\tproto_layer_rf_info: {},\n"
            "\tproto_minrange: {},\n"
            "\tproto_maxrange: {},\n"
            "\tnum_classes: {},\n"
            ")"
        )

        return rep.format(
            self.cnn_backbone,
            self.img_size,
            self.prototype_shape,
            self.proto_layer_rf_info,
            self.proto_minrange,
            self.proto_maxrange,
            self.num_classes,
        )

class LinearSquared(nn.Linear):
    """ Class that applies squared weights to the input so that these are always positive
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return F.linear(input, self.weight.square(), self.bias)
        #return F.linear(input, self.weight.square(), bias=False)
    
def construct_Video_protoEF(
    base_architecture,
    pretrained=True,
    img_size=224,
    prototype_shape=(40, 256, 1, 1, 1),
    similarity_metric="cosine",
    prototype_activation_function="linear",
    init_ll="ones",
    fc_method="weighted_sum",
    tau=1.0,
    num_classes=1,
    backbone_last_layer_num=-3,
    proto_minrange=10.0,
    proto_maxrange=90.0,
    **kwargs,
):
    cnn_backbone = base_architecture_to_features[base_architecture](
        pretrained=pretrained, last_layer_num=backbone_last_layer_num
    )
    return Video_protoEF(
        cnn_backbone=cnn_backbone,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=None, # TODO: check why this is not used
        similarity_metric=similarity_metric,
        prototype_activation_function=prototype_activation_function,
        init_ll=init_ll,
        fc_method=fc_method,
        tau=tau,
        num_classes=num_classes,
        proto_minrange=proto_minrange,
        proto_maxrange=proto_maxrange,
        init_weights=True,
    )

