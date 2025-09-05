from src.models.DenseNet import DenseNet
from src.models.resnet_features import resnet2p1d_18_cl, resnet2p1d_18_reg
from src.models.ProtoPNet import construct_PPNet
from src.models.XProtoNet import construct_XProtoNet
from src.models.Video_XProtoNet import construct_Video_XProtoNet
from src.models.Video_protoEF import construct_Video_protoEF
from copy import deepcopy
import logging

MODELS = {
    "DenseNet": DenseNet,
    "ProtoPNet": construct_PPNet,
    "XProtoNet": construct_XProtoNet,
    "Video_XProtoNet": construct_Video_XProtoNet,
    "Video_protoEF": construct_Video_protoEF,
    "resnet2p1d_18_cl": resnet2p1d_18_cl,
    "resnet2p1d_18_reg": resnet2p1d_18_reg,
}


def build(model_config):
    config = deepcopy(model_config)
    _ = config.pop("checkpoint_path") #Ignores the value of checkpoint.
    if "prototype_shape" in config.keys():
        config["prototype_shape"] = eval(config["prototype_shape"])

    # Build the model
    model_name = config.pop("name")
    model = MODELS[model_name](**config)
    logging.info(f"Model {model_name} is created.")

    return model
