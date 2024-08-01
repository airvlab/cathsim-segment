from sys import _xoptions
import torch
import torchvision
import pytorch_lightning as pl

# For local debugging on Intel Arc GPU.
#import intel_extension_for_pytorch as ipex

from cathseg.attention.hyperparams import (
        ENC_LEARNING_RATE,
        ENC_MOMENTUM,
        DEC_LEARNING_RATE,
        DEC_MOMENTUM
        )
from torch import nn
from torchsummary import summary

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dev = torch.device("xpu")

class Encode(nn.Module):
    def __init__(self, output_dim):
        super(Encode, self).__init__()
        # Pretrained ResNet50 model. Better pretrained as model might have already learnt shapes and contours!
        resnet = torchvision.models.resnet50(pretrained = True)
        # Remove FCN.
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Final layer: 2048 -> 3 in our case.
        self.additional_layer = nn.Conv2d(2048, output_dim, kernel_size = 1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.additional_layer(x)
        return x

class Decode(nn.Module):
    def __init__(self, input_dim, output_c):
        super(Decode, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(input_dim, 512, kernel_size = 4, stride = 2, padding = 1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)
        self.upconv5 = nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1)
        self.final_conv = nn.Conv2d(32, output_c, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.upconv3(x))
        x = self.relu(self.upconv4(x))
        x = self.relu(self.upconv5(x))
        x = self.final_conv(x)
        return x

