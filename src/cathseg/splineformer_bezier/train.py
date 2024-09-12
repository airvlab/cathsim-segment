from pathlib import Path

import cathseg.utils as utils
import numpy as np
import pytorch_lightning as pl
import torch
from cathseg.callbacks import ImageCallbackLoggerBezier
from cathseg.dataset import Guide3DBezierModule
from cathseg.splineformer_bezier.pl_module import SplineFormer as Model
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")

wandb.require("core")
# os.environ["WANDB_MODE"] = "offline"


MODEL_VERSION = "1024_bezier"
PROJECT = "transformer-7"
BATCH_SIZE = 8
IMAGE_SIZE = 1024
NUM_CHANNELS = 1
PATCH_SIZE = 32
D_MODEL = 1024
LIGHTNING_MODEL_DIR = f"lightning_model/{PROJECT}_{MODEL_VERSION}"


def img_untransform(img):
    img = img * 0.5 + 0.5
    return img


def c_untransform(c):
    return c


def c_transform(c):
    return c


def t_untransform(t):
    return np.clip(t * 1500, 0, 1500)


def t_transform(t):
    return np.clip(t / 1500, 0, 1)


dm = Guide3DBezierModule(
    dataset_path=Path.home() / "data/segment-real/",
    batch_size=BATCH_SIZE,
    n_channels=NUM_CHANNELS,
    image_size=IMAGE_SIZE,
    c_transform=c_transform,
    t_transform=t_transform,
)

model = Model(
    tgt_max_len=8,
    img_size=IMAGE_SIZE,
    num_channels=NUM_CHANNELS,
    d_model=D_MODEL,
    patch_size=PATCH_SIZE,
    num_heads=8,
    dropout=0.1,
)

image_callback = ImageCallbackLoggerBezier(
    img_untransform=img_untransform, c_untransform=c_untransform, t_untransform=t_untransform
)
model_checkpoint_callback = ModelCheckpoint(f"models/{PROJECT}-{MODEL_VERSION}", monitor="train/total_loss", mode="min")


def train():
    wandb_logger = pl.loggers.WandbLogger(project=PROJECT, version=MODEL_VERSION, log_model=True)
    trainer = pl.Trainer(
        default_root_dir=LIGHTNING_MODEL_DIR,
        max_epochs=600,
        logger=wandb_logger,
        callbacks=[image_callback, model_checkpoint_callback],
        gradient_clip_val=1,
    )
    trainer.fit(
        model,
        datamodule=dm,
        # ckpt_path=utils.get_latest_ckpt(f"models/{PROJECT}-{MODEL_VERSION}"),,
    )
    trainer.test(model, datamodule=dm)


def dummy_run_2():
    dm = Guide3DBezierModule(
        dataset_path=Path.home() / "data/segment-real/",
        batch_size=1,
        n_channels=NUM_CHANNELS,
        image_size=IMAGE_SIZE,
        c_transform=c_transform,
        t_transform=t_transform,
    )
    trainer = pl.Trainer(
        fast_dev_run=True, default_root_dir=LIGHTNING_MODEL_DIR, callbacks=[image_callback, model_checkpoint_callback]
    )
    trainer.fit(model, datamodule=dm)


def test():
    trainer = pl.Trainer(
        # fast_dev_run=True,
        default_root_dir=LIGHTNING_MODEL_DIR,
        callbacks=[image_callback, model_checkpoint_callback],
    )
    trainer.test(
        model,
        datamodule=dm,
        ckpt_path=utils.get_latest_ckpt(f"models/{PROJECT}-{MODEL_VERSION}"),
    )


def predict():
    dm = Guide3DBezierModule(
        dataset_path=Path.home() / "data/segment-real/",
        batch_size=1,
        n_channels=NUM_CHANNELS,
        image_size=IMAGE_SIZE,
        c_transform=c_transform,
        t_transform=t_transform,
    )

    trainer = pl.Trainer(
        default_root_dir=LIGHTNING_MODEL_DIR, max_epochs=200, callbacks=[image_callback, model_checkpoint_callback]
    )
    predictions = trainer.predict(
        model,
        datamodule=dm,
        return_predictions=False,
        ckpt_path=utils.get_latest_ckpt(f"models/{PROJECT}-{MODEL_VERSION}"),
    )

    exit()
    dm.setup("test")
    dl = dm.test_dataloader()
    # model = Model.load_from_checkpoint(utils.get_latest_ckpt(f"models/{PROJECT}-{MODEL_VERSION}"))
    # model.eval()
    for i, batch in enumerate(dl):
        imgs, tgt, tgt_mask = batch
        print((tgt * 1024).to(int))

        img, generated_seq, encoder_atts, decoder_atts, memory = model.predict_step(batch, i)
        print((generated_seq).to(int))
        encoder_atts = utils.process_attention_maps(
            decoder_atts,
            img_size=IMAGE_SIZE,
            channels=NUM_CHANNELS,
            patch_size=PATCH_SIZE,
            layer=-1,
            aggreg_func=lambda x: torch.max(x, dim=2)[0],
            discard_ratio=0.6,
        )
        utils.plot_attention_maps(generated_seq[0], encoder_atts[0], img.squeeze())
        exit()


if __name__ == "__main__":
    # dummy_run_2()
    train()
    # test()
    # predict()
