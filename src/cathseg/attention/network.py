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
from torch.optim.sgd import SGD
from torchsummary import summary

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dev = torch.device("xpu")

class EncodeDecodePL(pl.LightningModule):
    def __init__(self):
        super(EncodeDecodePL, self).__init__()

        self.model_enc = Encode()
        self.model_dec = Decode()
        self.loss = nn.BCEWithLogitsLoss()

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, _ = batch

        enc_opt, dec_opt = self.optimizers()

        enc_opt.zero_grad()
        dec_opt.zero_grad()

        enc_x = self.model_enc(x)
        dec_x = self.model_dec(enc_x)
        loss = self.loss(dec_x, x)
        
        self.manual_backward(loss)
        enc_opt.step()
        dec_opt.step()

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch):
        x, _ = batch

        enc_x = self.model_enc(x)
        dec_x = self.model_dec(enc_x)
        loss = self.loss(dec_x, x)

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        enc_opt = SGD(self.model_enc.parameters(),
                      lr = ENC_LEARNING_RATE,
                      momentum = ENC_MOMENTUM)
        dec_opt = SGD(self.model_dec.parameters(),
                      lr = DEC_LEARNING_RATE,
                      momentum = DEC_MOMENTUM)

        return enc_opt, dec_opt

class Encode(nn.Module):
    def __init__(self, output_dim = 412):
        super(Encode, self).__init__()
        # Pretrained ResNet50 model. Better pretrained as model might have already learnt shapes and contours!
        resnet = torchvision.models.resnet50(pretrained = True)
        # Remove FCN.
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Final layer: 2048 -> 3 in our case.
        self.additional_layer = nn.Conv2d(2048, output_dim, kernel_size = 1)
    
        #for param in resnet.parameters():
        #    param.requires_grad = False

        #for param in resnet.layer4.parameters():
        #    param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.additional_layer(x)
        return x

class Decode(nn.Module):
    def __init__(self, input_dim = 412, output_c = 3):
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

