import pytorch_lightning as pl
import torch.utils.data as data

from cathseg.mobile_unet.network import MobileUNetLightning
from guide3d.dataset.segment import Guide3D
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb
import torch

import numpy as np
import matplotlib.pyplot as plt

image_transforms = transforms.Compose(
    [
        #transforms.Lambda(lambda x: x / 255.0),
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
	    transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,)),
        # gray to RGB
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
	    transforms.Lambda(lambda x: np.array(x, dtype = np.float32)),
    ]
)

mask_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        #transforms.Lambda(lambda x: np.array(x, dtype = np.int32))
    ]
)

def main():
    # Set up WandB logger.
    wandb_logger = WandbLogger(log_model = "all")
    ckpt_callback = ModelCheckpoint(monitor = "val_dice",
                                    mode = "max",
                                    filename = "best_model_gpunorm",
                                    dirpath = "./",
                                    save_top_k = 1)
    early_stopping = EarlyStopping(monitor = "val_dice",
                                   min_delta = 0.0,
                                   patience = 15,
                                   mode = "max",
                                   verbose = True)
    scheduler = StochasticWeightAveraging(swa_epoch_start = 69,
                                          swa_lrs = 1e-4,
                                          annealing_epochs = 5,
                                          annealing_strategy = "cos")



    #torch.set_float32_matmul_precision("high")

    # TODO: Don't use absolute paths.
    root = "../../../guide3d/data/annotations/"

    model = MobileUNetLightning()
    #model = torch.compile(model)

    ds_train = Guide3D(root = root,
                       image_transform = image_transforms,
                       mask_transform = mask_transforms,
                       split = "train")
    ds_val = Guide3D(root = root,
                     image_transform = image_transforms,
                     mask_transform = mask_transforms,
                     split = "val")

    dl_train = data.DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=16)
    dl_val = data.DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=16)

    trainer = pl.Trainer(default_root_dir="/users/sgsdoust/sharedscratch/checkpoints/",
                         max_epochs = 200,
                         logger = wandb_logger,
                         fast_dev_run = False,
                         callbacks = [ckpt_callback, early_stopping, scheduler])
    trainer.fit(model,
                train_dataloaders = dl_train,
                val_dataloaders = dl_val)

def visualise(data):
    plt.imshow(data, cmap = "gray")
    plt.savefig("data.png")

if __name__ == "__main__":
    main()
