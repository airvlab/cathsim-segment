from pathlib import Path

import pytorch_lightning as pl
import torch.utils.data as data
from cathseg.mobile_unet.network import MobileUNetLightning
from guide3d.dataset.segment import Guide3D
from torchvision import transforms

image_transforms = transforms.Compose(
    [
        # transforms.Resize((256, 256)),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize((0.5,), (0.5,)),
        # gray to RGB
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ]
)


def main():
    root = Path.home() / "data" / "segment-real"
    model = MobileUNetLightning()
    dataset = Guide3D(root=root)

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    trainer = pl.Trainer()
    # trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
