import torch
import torchvision
import pytorch_lightning as pl

from cathseg.attention.hyperparams import (
        ENC_LEARNING_RATE,
        ENC_MOMENTUM,
        DEC_LEARNING_RATE,
        DEC_MOMENTUM
        )
from torch import nn
from torchsummary import summary

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, enc_dim):
        super(Encoder, self).__init__()
       
        self.linear_1 = nn.Linear(input_dim * input_dim, (input_dim * input_dim) // input_dim)
        self.linear_2 = nn.Linear((input_dim * input_dim) // input_dim, (input_dim * input_dim) // 4)
        self.linear_3 = nn.Linear(input_dim // 3, input_dim // 4)
        self.linear_4 = nn.Linear(input_dim // enc_dim, enc_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu(x)
        print(out.shape)
        out = self.linear_2(x)
        out = self.relu(x)
        out = self.linear_3(x)
        out = self.relu(x)
        out = self.linear_4(x)

        return out

class Decoder(nn.Module):
    def __init__(self, input_dim, dec_dim):
        super(Decoder, self).__init__()

    def forward(self, x):
        return NotImplemented


if __name__ == "__main__":
    enc = Encoder(1024, 256)
    X = torch.rand(1, 3, 1024, 1024)
    X = X.reshape(-1, 1024 * 1024)
    X_enc = enc(X)

    print(X_enc.shape)
