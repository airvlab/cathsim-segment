import pytorch_lightning as pl
import torch
import torch.utils.data as data
from cathseg.mobile_unet.network import MobileUNetLightning
from guide3d.dataset.segment import Guide3D
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms

image_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ]
)

mask_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def main():
    # Set up WandB logger.
    wandb_logger = WandbLogger(log_model="all")
    ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best_model_gpunorm",
        dirpath="./",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=200, mode="min", verbose=True
    )
    scheduler = StochasticWeightAveraging(
        swa_epoch_start=2, swa_lrs=1e-5, annealing_epochs=15, annealing_strategy="cos"
    )

    torch.set_float32_matmul_precision("high")

    # TODO: Don't use absolute paths.
    root = "../../../guide3d/data/annotations/"

    model = MobileUNetLightning()

    ds_train = Guide3D(
        root=root,
        image_transform=image_transforms,
        mask_transform=mask_transforms,
        split="train",
    )
    ds_val = Guide3D(
        root=root,
        image_transform=image_transforms,
        mask_transform=mask_transforms,
        split="val",
    )

    dl_train = data.DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=8)
    dl_val = data.DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=8)

    trainer = pl.Trainer(
        default_root_dir="./",
        max_epochs=200,
        logger=wandb_logger,
        callbacks=[ckpt_callback, early_stopping, scheduler],
    )
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)


if __name__ == "__main__":
    main()
