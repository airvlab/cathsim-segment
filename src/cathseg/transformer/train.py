import os
from pathlib import Path

import cathseg.utils as utils
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from cathseg.transformer.network import ImageToSequenceTransformer as Model
from guide3d.dataset.image.spline import Guide3D
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.interpolate import splev
from tqdm import tqdm

import wandb

torch.manual_seed(0)

wandb.require("core")
# os.environ["WANDB_MODE"] = "offline"

torch.set_float32_matmul_precision("high")

IMAGE_SIZE = 1024
N_CHANNELS = 1
MODEL_VERSION = "1"


vit_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert image to PIL image
        # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize image to 224x224
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(N_CHANNELS, 1, 1)),
        transforms.Normalize(  # Normalize with mean and std
            mean=[0.5 for _ in range(N_CHANNELS)],
            std=[0.5 for _ in range(N_CHANNELS)],
        ),
    ]
)


def c_transform(c):
    max_val = IMAGE_SIZE
    min_val = -0.6102361976879171
    return (c - min_val) / (max_val - min_val)


def t_transform(t):
    return t / 1500


def c_untransform(c):
    max_val = IMAGE_SIZE
    min_val = -0.6102361976879171
    return c * (max_val + min_val) + min_val


def t_untransform(t):
    return t * 1500


def unnorm(img):
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
            c_gen = instance["c_gen"].detach().cpu().numpy()  # alredy in shape
            t_pred = instance["t_pred"].detach().cpu().numpy()[:seq_len]
            t_true = instance["t_true"].detach().cpu().numpy()[:seq_len]
            t_gen = instance["t_gen"].detach().cpu().numpy()  # alredy in shape
            return img, c_pred, c_true, c_gen, t_pred, t_true, t_gen

        img, c_pred, c_true, c_gen, t_pred, t_true, t_gen = unpack_instance(instance)
        img = unnorm(img)
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

        img_true = self.make_points(img.copy(), c_true, t_true, (255, 0, 0))
        img_pred = self.make_points(img.copy(), c_pred, t_pred, (0, 255, 0))
        img_gen = self.make_points(img.copy(), c_gen, t_gen, (0, 0, 255))

        return [img_true, img_pred, img_gen]

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.epoch % self.interval == 0:
            instances = pl_module.training_step_output
            table = wandb.Table(columns=["GT", "Preds", "Inference"])

            for i, instance in tqdm(enumerate(instances)):
                generated = pl_module.inference_step(instance["img"])
                instances[i]["c_gen"] = generated["c"]
                instances[i]["t_gen"] = generated["t"]

            # Pass through to make images for logging. Do this row-wise.
            for instance in tqdm(instances):
                img_true, img_pred, img_gen = self.make_images(instance)

                table.add_data(
                    wandb.Image(img_true[:, :, 0]), wandb.Image(img_pred[:, :, 0]), wandb.Image(img_gen[:, :, 0])
                )

            trainer.logger.experiment.log({"img_samples": table})

        self.epoch += 1


def train():
    wandb_logger = pl.loggers.WandbLogger(
        project="transformer-2",
        log_model=True,
    )
    root = Path.home() / "data/segment-real/"
    train_ds = Guide3D(
        root=root,
        annotations_file="sphere_wo_reconstruct.json",
        image_transform=vit_transform,
        c_transform=c_transform,
        t_transform=t_transform,
        add_init_token=True,
        split="train",
    )
    val_ds = Guide3D(
        root=root,
        annotations_file="sphere_wo_reconstruct.json",
        image_transform=vit_transform,
        c_transform=c_transform,
        t_transform=t_transform,
        add_init_token=True,
        split="val",
    )
    train_dl = data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=os.cpu_count() // 2)
    val_dl = data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=os.cpu_count() // 2)
    model = Model(max_seq_len=train_ds.max_length, img_size=IMAGE_SIZE, n_channels=N_CHANNELS)
    trainer = pl.Trainer(
        max_epochs=200,
        logger=wandb_logger,
        callbacks=[
            ImageCallbackLogger(),
            ModelCheckpoint(f"models/{MODEL_VERSION}", monitor="val/loss", mode="min"),
        ],
    )
    trainer.fit(model, train_dl, val_dl)


def test():
    root = Path.home() / "data/segment-real/"
    test_ds = Guide3D(
        root=root,
        annotations_file="sphere_wo_reconstruct.json",
        image_transform=vit_transform,
        c_transform=c_transform,
        t_transform=t_transform,
        add_init_token=True,
        split="train",
    )

    model = Model(max_seq_len=test_ds.max_length, img_size=IMAGE_SIZE, n_channels=N_CHANNELS)
    trainer = pl.Trainer()
    trainer.test(model, test_ds, ckpt_path="lightning_logs/version_0/checkpoints/epoch=74-step=61425.ckpt")


def dummy_run():
    MAX_LEN = 20
    model = Model(max_seq_len=MAX_LEN, img_size=IMAGE_SIZE)
    dataloader = data.DataLoader(utils.DummyData(64, (3, 224, 224), MAX_LEN), batch_size=8, shuffle=True)

    trainer = pl.Trainer(
        max_epochs=200,
        fast_dev_run=True,
        callbacks=[ImageCallbackLogger()],
    )
    trainer.fit(model, dataloader)


def dummy_run_2():
    root = Path.home() / "data/segment-real/"
    train_ds = Guide3D(
        root=root,
        annotations_file="sphere_wo_reconstruct.json",
        image_transform=vit_transform,
        c_transform=c_transform,
        t_transform=t_transform,
        add_init_token=True,
        split="train",
    )
    model = Model(max_seq_len=train_ds.max_length, img_size=IMAGE_SIZE, n_channels=N_CHANNELS)
    dataloader = data.DataLoader(train_ds, batch_size=8, shuffle=True)

    trainer = pl.Trainer(
        max_epochs=200,
        fast_dev_run=True,
        callbacks=[ImageCallbackLogger()],
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    # dummy_run_2()
    train()
    # test()
