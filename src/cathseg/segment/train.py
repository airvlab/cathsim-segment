import cathseg.utils as utils
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from cathseg.metrics import compute_all_metrics
from PIL import Image

INDEX = 0


def apply_blue_mask(grayscale_img, binary_mask, output_path):
    # print(grayscale_img.shape, grayscale_img.min(), grayscale_img.max())
    # plt.imshow(grayscale_img)
    # plt.show()
    # exit()
    grayscale_img = (grayscale_img + grayscale_img.min()) * (grayscale_img.max() - grayscale_img.min())
    grayscale_img = Image.fromarray(grayscale_img * 255)
    binary_mask = Image.fromarray(binary_mask)

    # Convert grayscale image to RGB for applying color mask
    grayscale_img_rgb = grayscale_img.convert("RGB")
    # grayscale_img_rgb.show("grayscale img")
    grayscale_img_array = np.array(grayscale_img_rgb)

    # Convert mask to binary (0 or 1)
    binary_mask_array = np.array(binary_mask) > 0.001

    blue_overlay = np.zeros_like(grayscale_img_array)
    blue_overlay[..., 2] = 255

    result = np.where(binary_mask_array[..., None], blue_overlay, grayscale_img_array)

    result_img = Image.fromarray(result.astype("uint8"))

    result_img.save(output_path)


class SegmentModule(pl.LightningModule):
    def __init__(self, model, **model_kwargs):
        super().__init__()
        self.model = model(**model_kwargs)
        self.loss = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self(x))
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self(x))
        loss = self.loss(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        global INDEX
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y = (y >= 0.001).to(int)
        losses = compute_all_metrics(y, y_hat)
        for k, v in losses.items():
            self.log(k, v)

        apply_blue_mask(
            x[0].squeeze().cpu().numpy(), y_hat[0].squeeze().cpu().numpy(), f"samples/segmentation/{INDEX}.png"
        )
        INDEX += 1
        # plt.imshow(x[0][0].cpu().numpy(), cmap="gray")
        # y = y.cpu().numpy()
        # y_hat = y_hat.cpu().numpy()
        # axs[0].imshow(y[0].squeeze(), cmap="gray")
        # axs[1].imshow(y_hat[0].squeeze(), cmap="gray")
        # for ax in axs:
        #     ax.axis("off")
        # plt.show()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def train(model, ckpt_name, **kwargs):
    import pytorch_lightning as pl
    from cathseg.dataset import Guide3DSegmentModule
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torchvision import transforms

    mask_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    wandb_logger = pl.loggers.WandbLogger(
        project="segment",
        log_model=True,
        version=ckpt_name,
    )

    dm = Guide3DSegmentModule(batch_size=1, image_size=1024, mask_transforms=mask_transforms)
    model = SegmentModule(model, **kwargs)
    model_checkpoint_callback = ModelCheckpoint(f"models/{ckpt_name}", monitor="val/loss", mode="min")

    trainer = pl.Trainer(logger=wandb_logger, max_epochs=200, callbacks=[model_checkpoint_callback])
    trainer.fit(model, datamodule=dm)


def test(model, ckpt_name):
    import pytorch_lightning as pl
    from cathseg.dataset import Guide3DSegmentModule
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torchvision import transforms

    mask_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.unsqueze(1)),
        ]
    )

    dm = Guide3DSegmentModule(
        batch_size=2,
        image_size=1024,
        mask_transforms=mask_transforms,
        download=True,
    )
    model = SegmentModule(model)
    model_checkpoint_callback = ModelCheckpoint(f"models/{ckpt_name}", monitor="val/loss", mode="min")

    trainer = pl.Trainer(max_epochs=200, callbacks=[model_checkpoint_callback])
    trainer.test(
        model,
        datamodule=dm,
        ckpt_path=utils.get_latest_ckpt(f"models/{ckpt_name}"),
    )


if __name__ == "__main__":
    from cathseg.segment.swin_unet import SwinTransformerSys as SwinUNet
    from cathseg.segment.unet import UNet

    # train(UNet, ckpt_name="unet")
    test(UNet, ckpt_name="unet")
    exit()
    test(SwinUNet, ckpt_name="swinunet")
