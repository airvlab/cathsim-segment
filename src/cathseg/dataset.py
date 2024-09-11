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
        annotations_file="sphere_wo_reconstruct_2.json",
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


class Guide3DSegmentModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path=Path.home() / "data/segment-real",
        annotations_file="sphere_wo_reconstruct.json",
        batch_size=32,
        test_batch_size=1,
        n_channels=1,
        image_size=512,
        mask_transforms=None,
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
        self.mask_transforms = mask_transforms

    def setup(self, stage: str):
        assert stage in ["fit", "test", "predict"], f"Expected 'fit', 'test' or 'predict' but found {stage}"
        from guide3d.dataset.image.segment import Guide3D as Guide3DSegment

        image_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Grayscale(),
            ]
        )

        if stage == "fit":
            self.train_ds = Guide3DSegment(
                dataset_path=self.dataset_path,
                annotations_file="sphere_wo_reconstruct.json",
                split="train",
                download=self.download,
                image_transform=image_transform,
                mask_transform=self.mask_transforms,
            )

            self.val_ds = Guide3DSegment(
                dataset_path=self.dataset_path,
                annotations="sphere_wo_reconstruct.json",
                split="val",
                image_transform=image_transform,
                mask_transform=self.mask_transforms,
                download=self.download,
            )

        if stage == "test" or stage == "predict":
            self.test_ds = Guide3DSegment(
                dataset_path=self.dataset_path,
                annotations_file="sphere_wo_reconstruct.json",
                split="test",
                image_transform=image_transform,
                mask_transform=self.mask_transforms,
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


def sample_spline(tck: tuple, n: int = None, delta: float = None):
    import numpy as np
    from scipy.interpolate import splev

    assert delta or n, "Either delta or n must be provided"
    assert not (delta and n), "Only one of delta or n must be provided"

    def is2d(tck):
        return len(tck[1]) == 2

    u_max = tck[0][-1]
    num_samples = int(u_max / delta) + 1 if delta else n
    u = np.linspace(0, u_max, num_samples)
    if is2d(tck):
        x, y = splev(u, tck)
        return np.column_stack([x, y]).astype(np.int32)
    else:
        x, y, z = splev(u, tck)
        return np.column_stack([x, y, z])


def visualize_batch(batch, batch_idx):
    import math

    import matplotlib.pyplot as plt
    import torch

    imgs, tgts, tgt_pad_mask = batch
    batch_size, seq_len, _ = tgts.shape

    # Determine the grid size based on batch_size
    cols = min(4, batch_size)  # Limit to 4 columns at most
    rows = math.ceil(batch_size / cols)

    # Set figure size dynamically based on the grid
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Ensure axs is always a 2D array
    axs = axs.ravel() if batch_size > 1 else [axs]

    for instance in range(batch_size):
        img, tgt, seq_len = imgs[instance], tgts[instance], tgt_pad_mask[instance].sum().to(int)

        t = tgt[:seq_len, 0]
        t = torch.cat([torch.zeros(4), t])
        c = tgt[:seq_len, 1:3]
        pts = sample_spline((t, c.T, 3), delta=10)

        axs[instance].imshow(img[0], cmap="gray")
        axs[instance].plot(c[:, 0], c[:, 1], "ro", markersize=0.1)
        axs[instance].plot(pts[:, 0], pts[:, 1], "b", linewidth=0.5)

    # Turn off axes for all subplots
    for ax in axs:
        ax.axis("off")

    # Remove any unused subplots
    for idx in range(batch_size, len(axs)):
        axs[idx].remove()

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from cathseg.custom_modules import BSpline

    bspline = BSpline(3)
    print(bspline)
    datamodule = Guide3DModule(batch_size=2)
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    print("num train batches:", len(train_dataloader))
    for batch_idx, batch in enumerate(train_dataloader):
        imgs, tgts, tgt_masks = batch
        t_values_batched = torch.linspace(0, 1, steps=100).unsqueeze(0).repeat(2, 1)
        batch_out = bspline.forward(tgts[:, :, 1:3], tgts[:, :, 0], tgt_masks, num_samples=10, batched=True)

        print(tgt_masks)
        seq_len = tgt_masks.sum(1).to(int)[0]
        t = tgts[0, :seq_len, 0]
        c = tgts[0, :seq_len, 1:3]

        print("Knots (t)", t.shape)
        print("Coefficients (c)", c.shape)
        print("Knots", t)
        print("Coefficients", c)
        print("Sequence length", seq_len)
        t_values = torch.linspace(0, 1, steps=100)
        spline_values, _ = bspline(c, t, delta=10)
        print("Spline values", spline_values.shape)
        # print("Spline values", spline_values)
        plt.imshow(imgs[0, 0], cmap="gray")
        plt.plot(spline_values[:, 0], spline_values[:, 1], "ro")
        plt.show()

        # visualize_batch(batch, 0)
        exit()
