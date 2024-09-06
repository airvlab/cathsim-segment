from pathlib import Path

import cathseg.utils as utils
import pytorch_lightning as pl
import torch
from cathseg.dataset import Guide3D, Guide3DModule
from cathseg.splineformer_w_tip_pred.callback import ImageCallbackLogger
from cathseg.splineformer_w_tip_pred.pl_module import SplineFormer as Model
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

torch.manual_seed(0)
torch.set_float32_matmul_precision("medium")

wandb.require("core")
# os.environ["WANDB_MODE"] = "offline"


MODEL_VERSION = "with_tip_predictor-1024-rand_crop"
PROJECT = "transformer_3"
BATCH_SIZE = 32
IMAGE_SIZE = 1024
NUM_CHANNELS = 1
PATCH_SIZE = 32
D_MODEL = 256
LIGHTNING_MODEL_DIR = f"lightning_model/{PROJECT}_{MODEL_VERSION}"


def img_untransform(img):
    img = img * 0.5 + 0.5
    return img


def c_untransform(c):
    return c * IMAGE_SIZE


def t_untransform(t):
    return t * IMAGE_SIZE


def c_transform(c):
    return c / 1024


def t_transform(t):
    return t / 1024


dm = Guide3DModule(
    dataset_path=Path.home() / "data/segment-real/",
    annotations_file="sphere_wo_reconstruct.json",
    batch_size=BATCH_SIZE,
    n_channels=NUM_CHANNELS,
    image_size=IMAGE_SIZE,
    c_transform=c_transform,
    t_transform=t_transform,
)

model = Model(
    tgt_max_len=Guide3D.max_seq_len,
    img_size=IMAGE_SIZE,
    num_channels=NUM_CHANNELS,
    d_model=D_MODEL,
    patch_size=PATCH_SIZE,
    num_heads=8,
    dropout=0.1,
)

image_callback = ImageCallbackLogger(
    img_untransform=img_untransform, c_untransform=c_untransform, t_untransform=t_untransform
)
model_checkpoint_callback = ModelCheckpoint(f"models/{PROJECT}-{MODEL_VERSION}", monitor="val/loss", mode="min")


def train():
    wandb_logger = pl.loggers.WandbLogger(project=PROJECT, log_model=True)
    trainer = pl.Trainer(max_epochs=300, logger=wandb_logger, callbacks=[image_callback, model_checkpoint_callback])
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


def dummy_run_2():
    dm = Guide3DModule(
        dataset_path=Path.home() / "data/segment-real/",
        annotations_file="sphere_wo_reconstruct.json",
        batch_size=1,
        n_channels=NUM_CHANNELS,
        image_size=IMAGE_SIZE,
        c_transform=c_transform,
        t_transform=t_transform,
    )
    trainer = pl.Trainer(fast_dev_run=True, callbacks=[image_callback, model_checkpoint_callback])
    trainer.fit(model, datamodule=dm)


def test():
    trainer = pl.Trainer(callbacks=[image_callback, model_checkpoint_callback])
    trainer.test(model, datamodule=dm, ckpt_path="best_model_path")


def predict():
    dm = Guide3DModule(
        dataset_path=Path.home() / "data/segment-real/",
        annotations_file="sphere_wo_reconstruct.json",
        batch_size=1,
        n_channels=NUM_CHANNELS,
        image_size=IMAGE_SIZE,
        c_transform=c_transform,
        t_transform=t_transform,
    )

    trainer = pl.Trainer(max_epochs=200, fast_dev_run=True, callbacks=[image_callback, model_checkpoint_callback])
    pred = trainer.predict(
        model,
        datamodule=dm,
        return_predictions=True,
        ckpt_path=utils.get_latest_ckpt(f"models/{PROJECT}-{MODEL_VERSION}"),
        # ckpt_path="models/transformer_3-with_tip_predictor-1024-rand_crop/epoch=0-step=205.ckpt",
    )

    for batch in pred:
        img, generated_seq, encoder_atts, decoder_atts = batch
        # if generated_seq.shape[1] != 10:
        #     continue
        encoder_atts = utils.process_attention_maps(
            decoder_atts,
            img_size=IMAGE_SIZE,
            channels=NUM_CHANNELS,
            patch_size=PATCH_SIZE,
            layer=-1,
            aggreg_func=lambda x: torch.max(x, dim=2)[0],
            discard_ratio=0.9,
        )
        utils.plot_attention_maps(generated_seq[0], encoder_atts[0], img.squeeze())


if __name__ == "__main__":
    # dummy_run_2()
    # train()
    # test()
    predict()
