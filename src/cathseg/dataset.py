import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from guide3d.dataset.image.spline import Guide3D
from torchvision.transforms import transforms


class Guide3DModule(pl.LightningDataModule):
    max_seq_len = Guide3D.max_seq_len

    def __init__(
        self,
        dataset_path=Path.home() / "data/segment-real",
        annotations_file="sphere_wo_reconstruct.json",
        batch_size=32,
        test_batch_size=1,
        n_channels=1,
        image_size=1024,
        c_transform: callable = None,
        t_transform: callable = None,
        transform_both: callable = None,
        download=False,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.annotations_file = annotations_file
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.n_channels = n_channels
        self.image_size = image_size
        self.download = download

        self.c_transform = c_transform
        self.t_transform = t_transform
        self.transform_both = transform_both

    def setup(self, stage: str):
        assert stage in ["fit", "test", "predict"], f"Expected 'fit', 'test' or 'predict' but found {stage}"

        image_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(self.n_channels, 1, 1)),
                transforms.Normalize(
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
                image_transform=image_transform,
                c_transform=self.c_transform,
                t_transform=self.t_transform,
                transform_both=self.transform_both,
            )

            self.val_ds = Guide3D(
                dataset_path=self.dataset_path,
                annotations_file="sphere_wo_reconstruct.json",
                split="val",
                image_transform=image_transform,
                c_transform=self.c_transform,
                t_transform=self.t_transform,
                # transform_both=self.transform_both,
                download=self.download,
            )

        if stage == "test" or stage == "predict":
            self.test_ds = Guide3D(
                dataset_path=self.dataset_path,
                annotations_file="sphere_wo_reconstruct.json",
                split="test",
                image_transform=image_transform,
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
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
        )


class ResizeAndCropWithPoints:
    def __init__(self, upscale_factor):
        self.upscale_factor = upscale_factor

    def __call__(self, image, t, c):
        num_channels, original_size, _ = image.shape
        upsample_size = int(original_size * self.upscale_factor)

        image = F.resize(image, upsample_size)

        scale_factor = upsample_size / original_size

        c_resized = torch.clone(c)
        c_resized[:, 0] = c[:, 0] * scale_factor
        c_resized[:, 1] = c[:, 1] * scale_factor

        t_resized = torch.clone(t)
        t_resized = t * scale_factor

        i, j, h, w = transforms.RandomCrop.get_params(image, (original_size, original_size))
        image = F.crop(image, i, j, h, w)

        c_resized[:, 0] -= j
        c_resized[:, 1] -= i

        return image, t_resized, c_resized


if __name__ == "__main__":
    datamodule = Guide3DModule()
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    print("num train batches:", len(train_dataloader))
