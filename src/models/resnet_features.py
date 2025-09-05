"""
From ProtoPNet github codebase
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models
from torchsummary import summary

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnet2p1d_18": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth",
    "r3d_18": "https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
}

model_dir = "./pretrained_models"


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # class attribute
    expansion = 1
    num_layers = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # only conv with possibly not 1 stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # the residual connection
        out += identity
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [3, 3]
        block_strides = [self.stride, 1]
        block_paddings = [1, 1]

        return block_kernel_sizes, block_strides, block_paddings


class Bottleneck(nn.Module):
    # class attribute
    expansion = 4
    num_layers = 3

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        # only conv with possibly not 1 stride
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [1, 3, 1]
        block_strides = [1, self.stride, 1]
        block_paddings = [0, 1, 0]

        return block_kernel_sizes, block_strides, block_paddings


class ResNet_features(nn.Module):
    """
    the convolutional layers of ResNet
    the average pooling and final fully convolutional layer is removed
    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_features, self).__init__()

        self.inplanes = 64

        # the first convolutional layer before the structured sequence of blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # comes from the first conv and the following max pool
        self.kernel_sizes = [7, 3]
        self.strides = [2, 2]
        self.paddings = [3, 1]

        # the following layers, each layer is a sequence of blocks
        self.block = block
        self.layers = layers
        self.layer1 = self._make_layer(block=block, planes=64, num_blocks=self.layers[0])
        self.layer2 = self._make_layer(block=block, planes=128, num_blocks=self.layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, num_blocks=self.layers[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, num_blocks=self.layers[3], stride=2)

        # initialize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # only the first block has downsample that is possibly not None
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        # keep track of every block's conv size, stride size, and padding size
        for each_block in layers:
            (
                block_kernel_sizes,
                block_strides,
                block_paddings,
            ) = each_block.block_conv_info()
            self.kernel_sizes.extend(block_kernel_sizes)
            self.strides.extend(block_strides)
            self.paddings.extend(block_paddings)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        """
        the number of conv layers in the network, not counting the number
        of bypass layers
        """

        return (
            self.block.num_layers * self.layers[0]
            + self.block.num_layers * self.layers[1]
            + self.block.num_layers * self.layers[2]
            + self.block.num_layers * self.layers[3]
            + 1
        )

    def __repr__(self):
        template = "resnet{}_features"
        return template.format(self.num_layers() + 1)


def resnet18_features(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls["resnet18"], model_dir=model_dir)
        my_dict.pop("fc.weight")
        my_dict.pop("fc.bias")
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet34_features(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls["resnet34"], model_dir=model_dir)
        my_dict.pop("fc.weight")
        my_dict.pop("fc.bias")
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet50_features(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls["resnet50"], model_dir=model_dir)
        my_dict.pop("fc.weight")
        my_dict.pop("fc.bias")
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet101_features(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls["resnet101"], model_dir=model_dir)
        my_dict.pop("fc.weight")
        my_dict.pop("fc.bias")
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet152_features(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls["resnet152"], model_dir=model_dir)
        my_dict.pop("fc.weight")
        my_dict.pop("fc.bias")
        model.load_state_dict(my_dict, strict=False)
    return model


class resnet2p1d_18(nn.Module):
    def __init__(self, pretrained=True, last_layer_num=-3, **kwargs):
        """
        :param pretrained: to load the pretrained weights or not
        :param last_layer_num: for input shape (32,224,224)
            if -2 => output shape (512, 4,14,14).
            if -3 -> output shape (256, 8,28,28)
        """
        super().__init__()
        r2p1d = models.video.r2plus1d_18(pretrained=False)  # output shape = NxD = Nx512
        if pretrained:
            my_dict = model_zoo.load_url(model_urls["resnet2p1d_18"], model_dir=model_dir)
            r2p1d.load_state_dict(my_dict, strict=False)
        self.backbone = nn.Sequential(*(list(r2p1d.children())[:last_layer_num]))

    def forward(self, x):
        """
        x(batch of tensors)   shape = (N, C, T, H, W)
        """
        x = self.backbone(x)
        return x

class resnet2p1d_18_cl(nn.Module):
    def __init__(self, pretrained=True, 
                 last_layer_num=-3,  # not currently used TODO
                 num_classes: int = 4,
                 cnn_dropout_p: float = 0.2,
                 classifier_hidden_dim: int = 128,
                 classifier_dropout_p: float = 0.5,
                 **kwargs):
        """
        :param pretrained: to load the pretrained weights or not
        :param last_layer_num: for input shape (32,112,112)
            if -2 => output shape (512, 4,7,7).
            if -3 -> output shape (256, 8,14,14)
        """
        super().__init__()
        r2p1d = models.video.r2plus1d_18(pretrained=False)  # output shape = NxD = Nx512
        if pretrained:
            my_dict = model_zoo.load_url(model_urls["resnet2p1d_18"], model_dir=model_dir,  map_location='cpu')
            r2p1d.load_state_dict(my_dict, strict=False)
        #self.backbone = nn.Sequential(*(list(r2p1d.children())[:-3]))
        #self.backbone.add_module("adaptive_avgpool", nn.AdaptiveAvgPool3d((1, 1, 1)))
        #self.backbone.add_module("flatten", nn.Flatten())
        self.backbone = r2p1d
        self.backbone.fc = nn.Linear(in_features=512, out_features=num_classes) # TODO: hard coded
        # this is wrong. as there is no weights for this one. must be either this plus fc layer seperate or remove this and use hook
        #self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1]) # I think extractor method should be identical to forward pass
        # 2 dense blocks and transition blocks
        # output shape (N, 256, 8, 14, 14)
        # self.backbone = nn.Sequential(*list(densenet.children())[0][:-4])
        #self.backbone.classifier = nn.Identity()
        # the FC layer applied to the output of convolutional network it was 400 before changing to the last 3 channels
        # Must remove this as it has no effect: 
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(in_features= 256, out_features=classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout_p),
            nn.Linear(in_features=classifier_hidden_dim, out_features=num_classes),
        )

    def forward(self, x):
        """
        x(batch of tensors)   shape = (N, C, T, H, W)
        """
        x = self.backbone(x)
        #print(x.shape)
        #x = self.classifier(x)
        return x
    '''
    def extract_features(self, x):
        # Feature Extractor Layer
        embed = self.feature_extractor(x)  # shape (N, 512 or 256, T, H, W)
        # classification layer
        embed = embed.sum(dim=2).sum(dim=2).sum(dim=2) # sum over T, H, W (N, 512)

        logits = self.backbone.fc(embed) # TODO: not sure if this is what it is done inside the layers.
        #logits = self.forward(x) # two times forward pass not efficient TODO: this is ugly fix
        #embed = embed.sum(dim=2).sum(dim=2).sum(dim=2) # sum over T, H, W (N, 512)
        # shape (N, C)       (N, P, D)
        return logits, embed'''

class resnet2p1d_18_reg(nn.Module): # TODO: merge this
    def __init__(self, pretrained=True, 
                 #last_layer_num=-3,  # not currently used TODO
                 **kwargs):
        """
        :param pretrained: to load the pretrained weights or not
        :param last_layer_num: for input shape (32,112,112)
            if -2 => output shape (512, 4,7,7).
            if -3 -> output shape (256, 8,14,14)
        """
        pretrained = True
        
        super().__init__()
        self.model = models.video.r2plus1d_18(pretrained=True)  # output shape = NxD = Nx512

        #if pretrained:
        #    my_dict = model_zoo.load_url(model_urls["resnet2p1d_18"], model_dir=model_dir,  map_location='cpu')
        #    self.model.load_state_dict(my_dict, strict=False)
        
        #self.backbone = nn.Sequential(
        #    *(list(r2p1d.children())[:-1]),
        #    nn.AdaptiveAvgPool3d((1, 1, 1)),
        #   nn.Flatten()
        #)
        #self.backbone = r2p1d #this must output embeddings
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=1)
        self.model.fc.bias.data[0] = 55.6
    
        # Initialize linear layer with larger weights and bias
        #nn.init.constant_(self.backbone.fc.weight, 0.1)
        #nn.init.constant_(self.backbone.fc.bias, 0.5)
        # regress head for range 0-1
        #self.regressor = nn.Sequential(
        #    nn.Linear(in_features=400, out_features=1),
        #    nn.Sigmoid()
        # )

        # MLP regressor head
        #self.regressor = nn.Sequential(
        #    nn.Linear(in_features=512, out_features=128),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(p=0.5),
        #    nn.Linear(in_features=128, out_features=1),
        #    nn.Sigmoid()
        #)

    def forward(self, x):
        """
        x(batch of tensors)   shape = (N, C, T, H, W)
        """
        x = self.model(x)
        #x = self.regressor(x)
        return x
    
    def feature_extractor(self, x):
        """
        x(batch of tensors)   shape = (N, C, T, H, W)
        """
        embeds = self.backbone(x)
        pred = self.regressor(embeds)
        return pred, embeds

class ResNet3D18(nn.Module):
    def __init__(self, in_channels=1, pretrained=True, dropout=0, out_features=2, **kwargs):
        super(ResNet3D18, self).__init__()

        r3d = models.video.r2plus1d_18(pretrained=False)  # output shape = NxD = Nx512
        if pretrained:
            my_dict = model_zoo.load_url(model_urls["r3d_18"], model_dir=model_dir)
            r3d.load_state_dict(my_dict, strict=False)
        self.backbone = nn.Sequential(*(list(r3d.children())[:-2]))

    def forward(self, x):
        """
        x(batch of tensors)   shape = (N, C, T, H, W)
        """
        x = self.backbone(x)
        return x
    
if __name__ == "__main__":
    #r18_features = resnet18_features(pretrained=True)
    #print(r18_features)

    #r34_features = resnet34_features(pretrained=True)
    #print(r34_features)

    #r50_features = resnet50_features(pretrained=True)
    #print(r50_features)

    #r101_features = resnet101_features(pretrained=True)
    #print(r101_features)

    #r152_features = resnet152_features(pretrained=True)
    #print(r152_features)
    import torch
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.utils.utils import print_cuda_statistics
    
    #resnet2p1d_18_cl = resnet2p1d_18_cl(pretrained=True, num_classes=2) # this is only used for benchmark 
    #resnet2p1d_18 = resnet2p1d_18(pretrained=True) # this is used for prototype model. -3
    frames = 32
    img_size = 112
    model = resnet2p1d_18_reg(pretrained=True, last_layer_num=-1)#.to(device)
    summary(model, (3, frames, img_size, img_size), device="cpu")
    print(model)

    #torch.cuda.set_device(1)
    #torch.cuda.empty_cache()
    device = torch.device("cuda:1") #This breaks in test mode
    #self.device = torch.device("cuda") # TODO: to avoid cuda error invalid device ordinal

    # test feature extractor
    #model = model.to(device)
    #model.eval()

    # Container to store the extracted features
    #x = torch.randn(2, 3, 32, 112, 112).to(device)
    # Pass the input through the model
    #pred = model.forward(x) # forward pass
    #y_pred, features = model.feature_extractor(x) # feature extractor
    # Extracted features are now in the `extracted_features` dictionary
    #print("Extracted Features Shape:", features.shape)
    #print(y_pred.dtype, pred)