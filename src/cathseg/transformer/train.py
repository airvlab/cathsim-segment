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

os.environ["WANDB_MODE"] = "offline"

torch.set_float32_matmul_precision("high")

IMAGE_SIZE = 224
MAX_LEN = 20

vit_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert image to PIL image
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize image to 224x224
        # grayscale to RGB
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(  # Normalize with mean and std
            mean=[0.5, 0.5, 0.5],  # ViT models usually use these normalization values
            std=[0.5, 0.5, 0.5],
        ),
    ]
)


class ImageCallbackLogger(Callback):
    def __init__(self):
        self.epoch = 0

    def unnorm(self, img):
        img = img * 0.5 + 0.5
        return img

    def make_points(self, img, c, t):
        def in_bounds(x, y):
            return 0 <= x < img.shape[1] and 0 <= y < img.shape[0]

        samples = np.linspace(0, t[-1], 30)
        sampled_c = splev(samples, (t, c, 3))

        for control_point in c.astype(np.int32).T:
            if not in_bounds(control_point[0], control_point[1]):
                continue
            img = cv2.circle(img, tuple(control_point), 4, (255, 0, 0), -1)
        return img

    def make_images(self, instance):
        import matplotlib.pyplot as plt

        img = instance["img"].detach().cpu().numpy().transpose(1, 2, 0)
        img = self.unnorm(img)
        seq_len = instance["seq_len"].detach().cpu().numpy().astype(int)
        c_pred = instance["c_pred"].detach().cpu().numpy()[:seq_len]
        c_true = instance["c_true"].detach().cpu().numpy()[:seq_len]
        t_pred = instance["t_pred"].detach().cpu().numpy()[:seq_len]
        t_true = instance["t_true"].detach().cpu().numpy()[:seq_len]

        # add 4 zeroes to t at the beginning
        t_pred = np.concatenate([np.zeros((4, 1)), t_pred], axis=0)
        t_true = np.concatenate([np.zeros((4, 1)), t_true], axis=0)

        c_pred = c_pred.transpose(1, 0)
        c_true = c_true.transpose(1, 0)

        plt.imshow(img)
        plt.show()

        exit()
        # img_pred = self.make_points(img.copy(), c_pred, t_pred)
        img_true = self.make_points(img.copy(), c_true, t_true)

        plt.imshow(img_true)
        plt.show()

        return [img_true]

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.epoch % 10 == 0:
            instance = pl_module.training_step_output
            imgs = self.make_images(instance)

            trainer.logger.log_image(
                key="reprojection", images=imgs, caption=["GT", "Pred"]
            )
        self.epoch += 1


def train():
    # wandb_logger = pl.loggers.WandbLogger(
    #     project="transformer",
    #     log_model=True,
    # )
    root = Path.home() / "data/segment-real/"
    train_ds = Guide3D(
        root=root,
        annotations_file="sphere_wo_reconstruct.json",
        image_transform=vit_transform,
        split="train",
    )
    val_ds = Guide3D(
        root=root,
        annotations_file="sphere_wo_reconstruct.json",
        image_transform=vit_transform,
        split="val",
    )
    train_dl = data.DataLoader(
        train_ds, batch_size=8, shuffle=True, num_workers=os.cpu_count() // 2
    )
    val_dl = data.DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=os.cpu_count() // 2
    )
    model = Model(max_seq_len=train_ds.max_length)
    trainer = pl.Trainer(
        max_epochs=200,
        # logger=wandb_logger,
    )
    trainer.fit(model, train_dl, val_dl)


def dummy_run():
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
    model = Model(max_seq_len=train_ds.max_length, img_size=IMAGE_SIZE)
    dataloader = data.DataLoader(train_ds, batch_size=8, shuffle=True)

    trainer = pl.Trainer(
        max_epochs=200,
        fast_dev_run=True,
        callbacks=[ImageCallbackLogger()],
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    dummy_run_2()
    # train()
