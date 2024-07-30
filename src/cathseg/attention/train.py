import torch
import wandb
import numpy as np
import pytorch_lightning as pl
import torch.utils.data as data

from torchvision import transforms
from cathseg.attention.hyperparams import MAX_EPOCHS
from cathseg.attention.network import EncoderWithAttentionDecoder
from guide3d.dataset.segment import Guide3D
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

image_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32)
    ]
)
mask_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: np.where(x != 0.0, 1, 0))
    ]
)

def main():
    #wandb_logger = WandbLogger(log_model = "all")
    # TODO: Don't use absolute paths.
    root = "../../../guide3d/data/annotations/"
    model = EncoderWithAttentionDecoder()
    
    torch.set_float32_matmul_precision("high")

    ds_train = Guide3D(root = root,
                       image_transform = image_transforms,
                       mask_transform = mask_transforms,
                       split = "train")
    ds_test = Guide3D(root = root,
                      image_transform = image_transforms,
                      mask_transform = mask_transforms,
                      split = "test")
    ds_val = Guide3D(root = root,
                     image_transform = image_transforms,
                     mask_transform = mask_transforms,
                     split = "val")

    dl_train = data.DataLoader(ds_train, batch_size = 2,
                               shuffle = True, num_workers = 2)
    dl_test = data.DataLoader(ds_test, batch_size = 2, 
                              shuffle = False, num_workers = 2)
    dl_val = data.DataLoader(ds_val, batch_size = 2, 
                             shuffle = False, num_workers = 2)

    trainer = pl.Trainer(max_epochs = MAX_EPOCHS)
    trainer.fit(model,
                train_dataloaders = dl_train,
                val_dataloaders = dl_val)

if __name__ == "__main__":
    main()

