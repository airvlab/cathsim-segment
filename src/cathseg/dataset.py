import os
from pathlib import Path

import pytorch_lightning as pl
import torch.utils.data as data
from guide3d.dataset.image.spline import Guide3D
from torchvision.transforms import transforms


class Guide3DModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path=Path.home() / "data/segment-real",
        annotations_file="sphere_wo_reconstruct.json",
        batch_size=32,
        n_channels=1,
        image_size=1024,
        c_transform: callable = None,
        t_transform: callable = None,
        download=False,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.annotations_file = annotations_file
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.image_size = image_size
        self.download = download

        self.c_transform = c_transform
        self.t_transform = t_transform

    def setup(self, stage: str):
        vit_transform = transforms.Compose(
            [
                transforms.ToPILImage(),  # Convert image to PIL image
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(self.n_channels, 1, 1)),
                transforms.Normalize(  # Normalize with mean and std
                    mean=[0.5 for _ in range(self.n_channels)],
                    std=[0.5 for _ in range(self.n_channels)],
                ),
            ]
        )

        if stage == "fit":
            self.train_ds = Guide3D(
                dataset_path=self.dataset_path,
                annotations_file="sphere_wo_reconstruct.json",
                split="train",
                download=self.download,
                image_transform=vit_transform,
                c_transform=self.c_transform,
                t_transform=self.t_transform,
            )

            self.val_ds = Guide3D(
                dataset_path=self.dataset_path,
                annotations_file="sphere_wo_reconstruct.json",
                split="val",
                image_transform=vit_transform,
                c_transform=self.c_transform,
                t_transform=self.t_transform,
                download=self.download,
            )

        if stage == "test" or stage == "predict":
            self.test_ds = Guide3D(
                dataset_path=self.dataset_path,
                annotations_file="sphere_wo_reconstruct.json",
                split="test",
                image_transform=vit_transform,
                c_transform=self.c_transform,
                t_transform=self.t_transform,
                download=self.download,
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 2,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
        )


if __name__ == "__main__":
    datamodule = Guide3DModule()
