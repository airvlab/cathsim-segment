from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from cathseg.dataset import Guide3D, Guide3DModule
from cathseg.transformer_4.network import ImageToSequenceTransformer as Model
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.interpolate import splev
from tqdm import tqdm

import wandb

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")

wandb.require("core")
# os.environ["WANDB_MODE"] = "offline"


IMAGE_SIZE = 1024
N_CHANNELS = 1
MODEL_VERSION = "decoupled_heads_0"
PROJECT = "transformer"


def c_untransform(c):
    return c * IMAGE_SIZE


def t_untransform(t):
    return t * Guide3D.t_max


def img_untransform(img):
    img = img * 0.5 + 0.5
    return img


def plot_images(img_true, img_pred, img_gen):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_true)
    ax[1].imshow(img_pred)
    ax[2].imshow(img_gen)

    ax[0].set_title("GT")
    ax[1].set_title("Pred")
    ax[2].set_title("Inference")

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")

    plt.show()
    plt.close()
    # exit()


class ImageCallbackLogger(Callback):
    def __init__(self, interval=5):
        self.epoch = 0
        self.interval = interval

    def make_points(self, img, c, t, color=(0, 255, 0)):
        def in_bounds(x, y):
            return 0 <= x < img.shape[1] and 0 <= y < img.shape[0]

        samples = np.linspace(0, t[-1], 50)
        sampled_c = splev(samples, (t, c, 3))

        for control_point in c.astype(np.int32).T:
            if not in_bounds(control_point[0], control_point[1]):
                continue
            img = cv2.circle(img, tuple(control_point), 4, color, -1)

        img = cv2.polylines(
            img,
            np.array(sampled_c).T.astype(np.int32)[None, ...],
            False,
            (255, 255, 255),
        )

        return img

    def make_images(self, instance):
        def unpack_instance(instance):
            img = instance["img"].detach().cpu().numpy()
            seq_len = instance["seq_len"].detach().cpu().numpy().astype(int)
            c_pred = instance["c_pred"].detach().cpu().numpy()[:seq_len]
            c_true = instance["c_true"].detach().cpu().numpy()[:seq_len]
            c_gen = instance["c_gen"].detach().cpu().numpy()  # already in shape
            t_pred = instance["t_pred"].detach().cpu().numpy()[:seq_len]
            t_true = instance["t_true"].detach().cpu().numpy()[:seq_len]
            t_gen = instance["t_gen"].detach().cpu().numpy()  # already in shape

            return img, c_pred, c_true, c_gen, t_pred, t_true, t_gen

        img, c_pred, c_true, c_gen, t_pred, t_true, t_gen = unpack_instance(instance)
        img = img_untransform(img)
        c_pred = c_untransform(c_pred).T
        c_true = c_untransform(c_true).T
        c_gen = c_untransform(c_gen).T
        t_pred = t_untransform(t_pred).flatten()
        t_true = t_untransform(t_true).flatten()
        t_gen = t_untransform(t_gen)

        if img.shape[0] == 1:
            img = img[0]
        else:
            img = img.transpose(1, 2, 0)

        # add 4 zeroes to t at the beginning
        t_pred = np.concatenate([np.zeros((4)), t_pred], axis=0)
        t_true = np.concatenate([np.zeros((4)), t_true], axis=0)
        t_gen = np.concatenate([np.zeros((4)), t_gen], axis=0)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img * 255
        img = img.astype(np.uint8)

        img_true = self.make_points(img.copy(), c_true, t_true, (255, 255, 255))
        img_pred = self.make_points(img.copy(), c_pred, t_pred, (255, 255, 255))
        img_gen = self.make_points(img.copy(), c_gen, t_gen, (255, 255, 255))

        return [img_true, img_pred, img_gen]

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.epoch % self.interval == 0:
            instances = pl_module.training_step_output
            table = wandb.Table(columns=["GT", "Preds", "Inference"])

            for i, instance in tqdm(enumerate(instances)):
                generated = pl_module.inference_step(instance["img"])
                instances[i]["c_gen"] = generated["c"]
                instances[i]["t_gen"] = generated["t"]

            for instance in tqdm(instances):
                img_true, img_pred, img_gen = self.make_images(instance)

                table.add_data(
                    wandb.Image(img_true[:, :, 0]), wandb.Image(img_pred[:, :, 0]), wandb.Image(img_gen[:, :, 0])
                )

            trainer.logger.experiment.log({"img_samples": table})

        self.epoch += 1


def train():
    wandb_logger = pl.loggers.WandbLogger(project=PROJECT, log_model=True)

    dm = Guide3DModule(
        dataset_path=Path.home() / "data/segment-real/",
        annotations_file="sphere_wo_reconstruct.json",
        batch_size=32,
        n_channels=N_CHANNELS,
        image_size=IMAGE_SIZE,
    )
    model = Model(
        max_seq_len=Guide3D.max_seq_len,
        img_size=IMAGE_SIZE,
        n_channels=N_CHANNELS,
        d_model=256,
        num_decoder_layers=8,
        nhead=8,
    )

    trainer = pl.Trainer(
        max_epochs=200,
        logger=wandb_logger,
        callbacks=[
            ImageCallbackLogger(),
            ModelCheckpoint(f"models/{PROJECT}-{MODEL_VERSION}", monitor="val/loss", mode="min"),
            # EarlyStopping(monitor="val/loss", min_delta=1e-6, patience=5, mode="min", verbose=True),
        ],
    )
    trainer.fit(model, datamodule=dm)


def dummy_run_2():
    dm = Guide3DModule(
        dataset_path=Path.home() / "data/segment-real/",
        annotations_file="sphere_wo_reconstruct.json",
        batch_size=8,
        n_channels=N_CHANNELS,
        image_size=IMAGE_SIZE,
    )
    model = Model(max_seq_len=Guide3D.max_seq_len, img_size=IMAGE_SIZE, n_channels=N_CHANNELS)

    trainer = pl.Trainer(
        max_epochs=200,
        fast_dev_run=True,
        callbacks=[ImageCallbackLogger()],
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    # dummy_run_2()
    train()
    # test()
