from pathlib import Path

import cathseg.utils as utils
import pytorch_lightning as pl
import torch.utils.data as data
import torchvision.transforms as transforms
from cathseg.transformer.network import ImageToSequenceTransformer as Model
from guide3d.dataset.image.spline import Guide3D

MAX_LEN = 20

vit_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert image to PIL image
        transforms.Resize((224, 224)),  # Resize image to 224x224
        # grayscale to RGB
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(  # Normalize with mean and std
            mean=[0.5, 0.5, 0.5],  # ViT models usually use these normalization values
            std=[0.5, 0.5, 0.5],
        ),
    ]
)


def train():
    root = Path.home() / "data/segment-real/"
    dataset = Guide3D(
        root=root,
        annotations_file="sphere_wo_reconstruct.json",
        image_transform=vit_transform,
        split="train",
    )
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
    model = Model(max_seq_len=dataset.max_length)
    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(model, dataloader)


def dummy_run():
    model = Model(max_seq_len=MAX_LEN)
    dataloader = data.DataLoader(
        utils.DummyData(64, (3, 224, 224), MAX_LEN), batch_size=8, shuffle=True
    )

    trainer = pl.Trainer(max_epochs=200, fast_dev_run=True)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    # dummy_run()
    train()
