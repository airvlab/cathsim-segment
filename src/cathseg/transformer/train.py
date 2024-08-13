import os
from pathlib import Path

import cathseg.utils as utils
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from cathseg.transformer.network import ImageToSequenceTransformer as Model
from guide3d.dataset.image.spline import Guide3D
from pytorch_lightning import Callback
from scipy.interpolate import splev

import wandb

wandb.require("core")
# os.environ["WANDB_MODE"] = "offline"

torch.set_float32_matmul_precision("high")

IMAGE_SIZE = 1024
N_CHANNELS = 1


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
    return c / IMAGE_SIZE


def t_transform(t):
    return t / 2000


class ImageCallbackLogger(Callback):
    def __init__(self):
        self.epoch = 0

    def unnorm(self, img):
        img = img * 0.5 + 0.5
        return img

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

    def make_images(self, instance, plmodule):
        img = instance["img"].detach().cpu().numpy()
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = img.transpose(1, 2, 0)
        img = self.unnorm(img)

        seq_len = instance["seq_len"].detach().cpu().numpy().astype(int)
        c_pred = instance["c_pred"].detach().cpu().numpy()[1:seq_len].T * 1024
        c_true = instance["c_true"].detach().cpu().numpy()[1:seq_len].T * 1024
        t_pred = instance["t_pred"].detach().cpu().numpy()[1:seq_len].flatten()
        t_true = instance["t_true"].detach().cpu().numpy()[1:seq_len].flatten()

        generated = plmodule.inference_step(instance["img"])
        t_gen = generated["t"].detach().cpu().numpy()
        c_gen = generated["c"].detach().cpu().numpy()

        # add 4 zeroes to t at the beginning
        t_pred = np.concatenate([np.zeros((4)), t_pred], axis=0) * 2000
        t_true = np.concatenate([np.zeros((4)), t_true], axis=0) * 2000

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img * 255
        img = img.astype(np.uint8)

        img_true = self.make_points(img.copy(), c_true, t_true, (255, 0, 0))
        img_pred = self.make_points(img.copy(), c_pred, t_pred, (0, 255, 0))
        img_gen = self.make_points(img.copy(), c_gen, t_gen, (0, 0, 255))

        return [img_true, img_pred, img_gen]

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.epoch % 5 == 0:
            instance = pl_module.training_step_output

            trainer.logger.log_image(
                key="Images",
                images=self.make_images(instance, pl_module),
                caption=["GT", "Pred", "Inference"],
            )

        self.epoch += 1

        instance = pl_module.training_step_output
        img_true, img_pred = self.make_images(instance)



def train():
    wandb_logger = pl.loggers.WandbLogger(
        project="transformer",
        log_model=True,
    )
    #root = Path.home() / "data/segment-real/"
    root = "/home/shayandoust/Desktop/cathsim-segment/guide3d/data/annotations"
    train_ds = Guide3D(
        root=root,
        annotations_file="sphere_wo_reconstruct.json",
        image_transform=vit_transform,
        c_transform=c_transform,
        t_transform=t_transform,
        split="train",
    )
    val_ds = Guide3D(
        root=root,
        annotations_file="sphere_wo_reconstruct.json",
        image_transform=vit_transform,
        c_transform=c_transform,
        t_transform=t_transform,
        split="val",
    )
    train_dl = data.DataLoader(
        train_ds, batch_size=8, shuffle=True, num_workers=os.cpu_count() // 2
    )
    val_dl = data.DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=os.cpu_count() // 2
    )
    model = Model(
        max_seq_len=train_ds.max_length, img_size=IMAGE_SIZE, n_channels=N_CHANNELS
    )
    trainer = pl.Trainer(
        max_epochs=200,
        logger=wandb_logger,
        callbacks=[ImageCallbackLogger()],
    )
    trainer.fit(model, train_dl, val_dl)


def dummy_run():
    MAX_LEN = 20
    model = Model(max_seq_len=MAX_LEN, img_size=IMAGE_SIZE)
    dataloader = data.DataLoader(
        utils.DummyData(64, (3, 224, 224), MAX_LEN), batch_size=8, shuffle=True
    )

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
        split="train",
    )
    model = Model(
        max_seq_len=train_ds.max_length, img_size=IMAGE_SIZE, n_channels=N_CHANNELS
    )
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
