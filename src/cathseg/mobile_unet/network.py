import wandb

import cathseg.mobile_unet.network as model
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torchvision import transforms
from torchmetrics.classification import Dice, JaccardIndex, BinaryJaccardIndex

import matplotlib.pyplot as plt

class depthwise_pointwise_conv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super(depthwise_pointwise_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, groups=1),

        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            depthwise_pointwise_conv(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LightweightUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(LightweightUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return torch.sigmoid(x)

class MobileUNetLightning(pl.LightningModule):
    def __init__(self):
        super(MobileUNetLightning, self).__init__()
        self.model = model.LightweightUNet(1, 2)
        self.loss = nn.BCEWithLogitsLoss()
        self.dice = Dice(ignore_index = 0)
        self.jaccardindex = JaccardIndex(task = "multiclass", num_classes = 2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        x, y = RandomCropper(x, y)

        y_hat = self.model(x)
        loss = self.loss(y_hat[:, 1, :, :], y.float()[:, 0, :, :])

        if batch_idx == 0:
            table = wandb.Table(columns = ["X cropped", "Y cropped"])

            for i in range(min(5, x.size(0))):
                cropped_img = wandb.Image(x[i])
                cropped_mask = wandb.Image(y[i])

                table.add_data(cropped_img, cropped_mask)

            self.logger.experiment.log({"cropped_samples": table})

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        y_hat = self.model(x)
        #plt.imshow(torch.sigmoid(y_hat).cpu().numpy()[0, 0, :, :], cmap = "gray")
        #plt.savefig("validation.png")

        loss = self.loss(y_hat[:, 1, :, :], y.float()[:, 0, :, :])
        #dice = self.dice(y_hat, y.int())
        #ji = self.jaccardindex(y_hat[:, 1, :, :], y.float()[:, 0, :, :])

        if batch_idx == 0:
            table = wandb.Table(columns = ["X", "Y preds", "Y preds sigmoid"])

            for i in range(min(5, x.size(0))):
                input_img = wandb.Image(x[i])
                pred_img = wandb.Image(y_hat[i])
                pred_img_sigmoid = wandb.Image(torch.sigmoid(y_hat[i]))

                table.add_data(input_img, pred_img, pred_img_sigmoid)

            self.logger.experiment.log({"img_samples": table})

        self.log("val_loss", loss)
        #self.log("val_dice", dice)
        #self.log("val_jaccardindex", ji)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y.float())
        dice = self.dice(y_hat, y.float())
        ji = self.jaccardindex(y_hat, y.float())

        self.log("test_loss", loss)
        self.log("test_dice", dice)
        self.log("test_jaccardindex", ji)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)

    def predict(self, x):
        return self.model(x)


def RandomCropper(imgs, masks):
    i, j, h, w = transforms.RandomCrop.get_params(imgs, (412, 412))

    return TF.crop(imgs, i, j, h, w), TF.crop(masks, i, j, h, w)

if __name__ == "__main__":
    X = torch.rand((1, 3, 1024, 1024))  # Shape imitates that of our data.
    model = MobileUNetLightning()
    print(model)

    Y_preds = model(X)

    print(Y_preds)
    print(Y_preds.shape)
