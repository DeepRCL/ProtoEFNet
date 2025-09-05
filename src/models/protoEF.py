import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.receptive_field import compute_proto_layer_rf_info_v2
from src.models.ProtoPNet import PPNet, base_architecture_to_features

# Dont use this. Img based

class protoEF(PPNet):
    def __init__(self, **kwargs):
        super(protoEF, self).__init__(**kwargs)

        self.cnn_backbone = self.features
        del self.features
        cnn_backbone_out_channels = self.get_cnn_backbone_out_channels(self.cnn_backbone)
        self.output_num = 1
        # feature extractor module
        self._initialize_weights(self.add_on_layers)

        # Occurrence map module
        self.occurrence_module = nn.Sequential(
            nn.Conv2d(
                in_channels=cnn_backbone_out_channels,
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.prototype_shape[1],
                out_channels=self.prototype_shape[1] // 2,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.prototype_shape[1] // 2,
                out_channels=self.prototype_shape[0],
                kernel_size=1,
                bias=False,
            ),
        )
        self._initialize_weights(self.occurrence_module)

        # initialise the prototype classes
        self.proto_classes = torch.linspace(self.proto_minrange, self.proto_maxrange, self.num_prototypes)
       
        # Last layer applies squared weights so that they are always positive
        self.last_layer = LinearSquared(self.num_prototypes, self.output_num,
                                        bias=False)
        # Required for l2 convolution
        self.ones = nn.Parameter(torch.ones(self.proto_shape),
                                 requires_grad=False)
        
        self.om_softmax = nn.Softmax(dim=-1)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def set_last_layer_connection(self):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''

        class_IDx = torch.unsqueeze(self.proto_classes, dim=0)

        # set weights to prototype class label
        if self.init_ll == 'class_idx':
            self.last_layer.weight = nn.Parameter(torch.sqrt(class_IDx))
        
        elif self.init_ll == 'ones':
            self.last_layer.weight = nn.Parameter(torch.ones_like(class_IDx))

    def forward(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        feature_map = self.add_on_layers(x).unsqueeze(1) 
        occurrence_map = self.get_occurence_map_absolute_val(x)  
        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3) 

        # Prototype Layer
        similarity = self.cosine_similarity(
            features_extracted, self.prototype_vectors.squeeze().unsqueeze(0)
        )  # shape (N, P)
        similarity = (similarity + 1) / 2.0  # normalizing to [0,1] for positive reasoning 

        # classification layer
        logits = self.last_layer(similarity)

        return logits, similarity, occurrence_map

    def compute_occurence_map(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        occurrence_map = self.get_occurence_map_absolute_val(x)  
        return occurrence_map

    def get_occurence_map_softmaxed(self, x):
        occurrence_map = self.occurrence_module(x) 
        n, p, h, w = occurrence_map.shape
        occurrence_map = occurrence_map.reshape((n, p, -1))
        occurrence_map = self.om_softmax(occurrence_map).reshape((n, p, h, w)).unsqueeze(2)  
        return occurrence_map

    def get_occurence_map_absolute_val(self, x):
        occurrence_map = self.occurrence_module(x) 
        occurrence_map = torch.abs(occurrence_map).unsqueeze(2)  
        return occurrence_map

    def push_forward(self, x):
        """
        this method is needed for the pushing operation
        """
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        feature_map = self.add_on_layers(x).unsqueeze(1)  
        occurrence_map = self.get_occurence_map_absolute_val(x)  
        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3)  

        # Prototype Layer
        similarity = self.cosine_similarity(
            features_extracted, self.prototype_vectors.squeeze().unsqueeze(0)
        )  
        similarity = (similarity + 1) / 2.0  

        # classification layer
        logits = self.last_layer(similarity)

        return features_extracted, 1 - similarity, occurrence_map, logits

    def __repr__(self):
        # protoEF(self, backbone, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            "PPNet(\n"
            "\tcnn_backbone: {},\n"
            "\timg_size: {},\n"
            "\tprototype_shape: {},\n"
            "\tproto_layer_rf_info: {},\n"
            "\tnum_classes: {},\n"
            "\tepsilon: {}\n"
            ")"
        )

        return rep.format(
            self.cnn_backbone,
            self.img_size,
            self.prototype_shape,
            self.proto_layer_rf_info,
            self.num_classes,
            self.epsilon,
        )

class LinearSquared(nn.Linear):
    """ Class that applies squared weights to the input so that these are always positive
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return F.linear(input, self.weight.square(), self.bias)

def construct_protoEF(
    base_architecture,
    pretrained=True,
    img_size=224,
    prototype_shape=(2000, 512, 1, 1),
    num_classes=200,
    prototype_activation_function="log",
    add_on_layers_type="bottleneck",
    proto_minrange=0.05,
    proto_maxrange=0.95
):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(
        img_size=img_size,
        layer_filter_sizes=layer_filter_sizes,
        layer_strides=layer_strides,
        layer_paddings=layer_paddings,
        prototype_kernel_size=prototype_shape[2],
        proto_minrange=proto_minrange,
        proto_maxrange=proto_maxrange
    )
    return protoEF(
        features=features,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=proto_layer_rf_info,
        num_classes=num_classes,
        proto_minrange=proto_minrange,
        proto_maxrange=proto_maxrange,
        init_weights=True,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type,
    )
