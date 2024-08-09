import torch
import wandb

import pytorch_lightning as pl
import torch.utils.data as data
import numpy as np

from cathseg.transformer.network import EncoderDecoderAttentionPL
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



def main():
    # Set up WandB logger.
    wandb_logger = WandbLogger


if __name__ == "__main__":
    main()
