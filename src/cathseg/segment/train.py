import cathseg.utils as utils
import pytorch_lightning as pl
import torch
import torch.nn as nn
from cathseg.metrics import compute_all_metrics


class SegmentModule(pl.LightningModule):
    def __init__(self, model, **model_kwargs):
        super().__init__()
        self.model = model(**model_kwargs)
        self.loss = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test/loss", loss)
        losses = compute_all_metrics(y, y_hat)
        for k, v in losses.items():
            self.log(k, v)
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

    dm = Guide3DSegmentModule(batch_size=2, image_size=1024, mask_transforms=mask_transforms)
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

    dm = Guide3DSegmentModule(batch_size=2, image_size=1024, mask_transforms=mask_transforms)
    model = SegmentModule(model)
    model_checkpoint_callback = ModelCheckpoint(f"models/{ckpt_name}", monitor="val/loss", mode="min")

    trainer = pl.Trainer(max_epochs=200, callbacks=[model_checkpoint_callback])
    trainer.test(
        model,
        datamodule=dm,
        ckpt_path=utils.get_latest_ckpt(f"models/{ckpt_name}"),
    )


if __name__ == "__main__":
    from cathseg.segment.unet import UNet

    train(UNet, ckpt_name="unet")
    # test(UNet, ckpt_name="unet")
