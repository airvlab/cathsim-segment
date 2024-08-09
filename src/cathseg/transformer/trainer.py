import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from guide3d.dataset.image.spline import Guide3D as Dataset

# from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from torchvision import transforms

dataset_path = Path.home() / "data" / "segment-real"

image_transforms = transforms.Compose(
    [
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize((0.5,), (0.5,)),
        # gray to RGB
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ]
)


def pad_sequences(batch: list, target_len: int, padding_value: int = 0) -> torch.Tensor:
    padded_batch = []
    for tensor in batch:
        padded_tensor = F.pad(
            tensor, (0, 0, 0, target_len - tensor.size(0)), value=padding_value
        )
        padded_batch.append(padded_tensor)
    return torch.stack(padded_batch, dim=0)


def collate_fn(batch):
    imgs, ts, cs, us = zip(*batch)

    imgs = torch.stack(imgs)

    ts = pad_sequences(ts, 20)
    cs = pad_sequences(cs, 20)

    return imgs, ts, cs


train_dataset = Dataset(
    root=dataset_path,
    annotations_file="sphere_wo_reconstruct.json",
    split="train",
    image_transform=image_transforms,
)

val_dataset = Dataset(
    root=dataset_path,
    annotations_file="sphere_wo_reconstruct.json",
    split="val",
    image_transform=image_transforms,
)

train_dataloader = data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=os.cpu_count() // 2,
    collate_fn=collate_fn,
)

val_dataloader = data.DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=os.cpu_count() // 2,
)

for batch in train_dataloader:
    img, t, c, u = batch
    print("Image shape:", img.shape)
    print("Knot Vector shape:", t.shape)
    print("Coefficients shape:", c.shape)
    print("U shape:", u.shape)
    break

exit()

model = None


def train():
    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(model, train_dataloader, val_dataloader)


def _debug_training():
    torch.cuda.empty_cache()

    trainer = pl.Trainer(max_epochs=3, fast_dev_run=2)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    _debug_training()
    # train()
