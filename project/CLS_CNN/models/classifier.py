import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.vgg import vgg19, VGG19_Weights, vgg13, VGG13_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.efficientnet import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class ResNet50(nn.Module):
    def __init__(self, feature_dim=128):
        super(ResNet50, self).__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        # encoder
        self.encoder = create_feature_extractor(resnet50(weights=weights), return_nodes=['avgpool']).to("cuda")
        # clasification head
        self.classification_head = nn.Sequential(
            nn.Linear(2048, 1200),
            nn.ReLU(),
            nn.Linear(1200, 600),
            nn.ReLU(),
            nn.Linear(600, 3)
        )


    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)["avgpool"]
        x = x.view(batch_size, -1)
        x = self.classification_head(x)
        return x
    

class ResNet50_1layer(nn.Module):
    def __init__(self, feature_dim=128):
        super(ResNet50_1layer, self).__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        # encoder
        self.encoder = create_feature_extractor(resnet50(weights=weights), return_nodes=['avgpool']).to("cuda")
        # clasification head
        self.classification_head = nn.Sequential(
            nn.Linear(2048, 3)
        )


    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)["avgpool"]
        x = x.view(batch_size, -1)
        x = self.classification_head(x)
        return x
    
class VGG19(nn.Module):
    def __init__(self, feature_dim=128):
        super(VGG19, self).__init__()
        weights = VGG19_Weights.IMAGENET1K_V1
        # encoder
        self.encoder = create_feature_extractor(vgg19(weights=weights), return_nodes=['avgpool']).to("cuda")
        # clasification head
        self.classification_head = nn.Sequential(
            nn.Linear(25088, 5000),
            nn.ReLU(),
            nn.Linear(5000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 3)
        )


    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)["avgpool"]
        x = x.view(batch_size, -1)
        x = self.classification_head(x)
        return x
    
class VGG13(nn.Module):
    def __init__(self, feature_dim=128):
        super(VGG13, self).__init__()
        weights = VGG13_Weights.IMAGENET1K_V1
        # encoder
        self.encoder = create_feature_extractor(vgg13(weights=weights), return_nodes=['avgpool']).to("cuda")
        # clasification head
        self.classification_head = nn.Sequential(
            nn.Linear(25088, 5000),
            nn.ReLU(),
            nn.Linear(5000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 3)
        )


    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)["avgpool"]
        x = x.view(batch_size, -1)
        x = self.classification_head(x)
        return x


class CONVNEXT_BASE(nn.Module):
    def __init__(self):
        super(CONVNEXT_BASE, self).__init__()
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        # encoder
        self.encoder = create_feature_extractor(convnext_base(weights=weights), return_nodes=['avgpool']).to("cuda")
        # clasification head
        self.classification_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)["avgpool"]
        x = x.view(batch_size, -1)
        x = self.classification_head(x)
        return x

class EFFICIENTNET(nn.Module):
    def __init__(self, out=3):
        super(EFFICIENTNET, self).__init__()
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        # encoder
        self.encoder = create_feature_extractor(efficientnet_v2_m(weights=weights), return_nodes=['avgpool']).to("cuda")
        # clasification head
        self.classification_head = nn.Sequential(
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.Linear(320, out)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)["avgpool"]
        x = x.view(batch_size, -1)
        x = self.classification_head(x)
        x = self.softmax(x)
        return x