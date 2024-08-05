import torch
import wandb
import numpy as np
import pytorch_lightning as pl
import torch.utils.data as data

from torchvision import transforms
from cathseg.attention.hyperparams import MAX_EPOCHS
from cathseg.attention.network import EncodeDecodePL
from guide3d.dataset.segment import Guide3D
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

image_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels = 3),
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
    wandb_logger = WandbLogger(log_model = "all")
    # TODO: Don't use absolute paths.
    root = "/users/sgsdoust/sharedscratch/cathsim-segment/guide3d/data/annotations"
    enc_dec_model = EncodeDecodePL()
    
    torch.set_float32_matmul_precision("high")

    ckpt_callback = ModelCheckpoint(monitor = "train_loss",
                                    mode = "min",
                                    filename = "best_model",
                                    save_top_k = 2)
    early_stopping = EarlyStopping(monitor = "train_loss",
                                   min_delta = 0.0,
                                   patience = 5,
                                   mode = "min",
                                   verbose = True)
    #scheduler = StochasticWeightAveraging(swa_epoch_start = 10,
    #                                      swa_lrs = 1e-5,
    #                                      annealing_epochs = 10,
    #                                      annealing_strategy = "cos")

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

    dl_train = data.DataLoader(ds_train, batch_size = 4,
                               shuffle = True, num_workers = 1)
    dl_test = data.DataLoader(ds_test, batch_size = 4, 
                              shuffle = False, num_workers = 1)
    dl_val = data.DataLoader(ds_val, batch_size = 4, 
                             shuffle = False, num_workers = 1)

    trainer = pl.Trainer(max_epochs = MAX_EPOCHS, logger = wandb_logger,
                         callbacks = [ckpt_callback, early_stopping])
    trainer.fit(enc_dec_model,
                train_dataloaders = dl_train,
                val_dataloaders = dl_val)

if __name__ == "__main__":
    main()

