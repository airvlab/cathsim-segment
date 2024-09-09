import pytorch_lightning as pl
import torch
import torch.nn as nn


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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
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

    dm = Guide3DSegmentModule(batch_size=1, image_size=1024, mask_transforms=mask_transforms)
    model = SegmentModule(model, **kwargs)
    model_checkpoint_callback = ModelCheckpoint(f"models/{ckpt_name}", monitor="val/loss", mode="min")

    trainer = pl.Trainer(max_epochs=200, callbacks=[model_checkpoint_callback])
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    from cathseg.segment.unet import UNet

    train(UNet, ckpt_name="unet")
