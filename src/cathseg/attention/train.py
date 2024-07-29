import torch
import pytorch_lightning as pl
import torch.utils.data as data

from cathseg.attention.hyperparams import MAX_EPOCHS
from cathseg.attention.network import EncoderLightning
from guide3d.dataset.segment import Guide3D
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main():
    wandb_logger = WandbLogger(log_model = "all")
    # TODO: Don't use absolute paths.
    root = "../../../guide3d/data/annotations/"
    #model = AttentionNet()
    enc = EncoderLightning()
    
    torch.set_float32_matmul_precision("high")

    ds_train = Guide3D(root = root,
                       split = "train")
    ds_test = Guide3D(root = root,
                      split = "test")
    ds_val = Guide3D(root = root,
                     split = "val")

    dl_train = data.DataLoader(ds_train, batch_size = 2,
                               shuffle = True, num_workers = 2)
    dl_test = data.DataLoader(ds_test, batch_size = 2,
                              shuffle = False, num_workers = 2)
    dl_val = data.DataLoader(ds_val, batch_size = 2,
                             shuffle = False, num_workers = 2)

    trainer = pl.Trainer(default_root_dir = "./",
                         max_epochs = MAX_EPOCHS,
                         logger = wandb_logger)
    trainer.fit(enc,
                train_dataloaders = dl_train,
                val_dataloaders = dl_val)

if __name__ == "__main__":
    main()

