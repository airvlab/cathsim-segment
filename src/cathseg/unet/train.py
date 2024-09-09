import pytorch_lightning as pl
import torch
from cathseg.dataset import Guide3DSegmentModule
from cathseg.unet.model import LightningUNet as Model
from pytorch_lightning.callbacks import ModelCheckpoint

torch.cuda.empty_cache()

dm = Guide3DSegmentModule(image_size=256)
model = Model()
model_checkpoint_callback = ModelCheckpoint("models/unet", monitor="val/loss", mode="min")


def train():
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=200, callbacks=[model_checkpoint_callback])
    trainer.fit(model, datamodule=dm)
    exit()
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    train()
