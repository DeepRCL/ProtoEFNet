import torch.nn as nn
import torch
from torchsummary import summary


class DenseNet(nn.Module):
    """
    a simple multilabel image classifier using densenet
    """

    def __init__(
        self,
        num_classes: int = 4,
        cnn_dropout_p: float = 0.2,
        classifier_hidden_dim: int = 128,
        classifier_dropout_p: float = 0.5,
        **kwargs,
    ):
        """
        :param num_classes: int, number of classes
        :param cnn_dropout_p: float, dropout ratio of the CNN
        :param classifier_hidden_dim: int, the dimension of the hidden FC layer
        :param classifier_dropout_p: float, dropout ratio of the FC layer
        """
        super().__init__()

        # the backbone CNN(DenseNet), output shape (N, 1024)
        self.backbone = torch.hub.load("pytorch/vision:v0.10.0", "densenet121", pretrained=True)
        self.backbone.classifier = nn.Identity()

        # 2 dense blocks and transition blocks
        # output shape (N, 256, 14, 14)
        # self.backbone = nn.Sequential(*list(densenet.children())[0][:-4])

        # the FC layer applied to the output of convolutional network
        self.classifier: torch.nn.Sequential = nn.Sequential(
            nn.Linear(in_features=1024, out_features=classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout_p),
            nn.Linear(in_features=classifier_hidden_dim, out_features=num_classes),
        )

    def forward(self, x):
        """
        :param x: torch.tensor, input torch.tensor of image frames,
                  normalized with imagenet's mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
                  shape (N, 3, img_size, img_size)
        :return:  multi-hot Vector of logits with shape of (N, num_classes)
        """
        x = self.backbone(x)  # shape (N, 1024)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    batch_size = 20
    model = DenseNet()
    summary(model, torch.rand((batch_size, 3, 112, 112)))  # (N,C,T,H,W)
