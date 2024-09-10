import math
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor


class FeatureExtractor(nn.Module):
    def __init__(self, d_model):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.projection = nn.Linear(512, d_model)

    def forward(self, x):
        features = self.model(x)
        batch_size, channels, height, width = features.shape
        features = features.view(batch_size, channels, height * width).permute(0, 2, 1)
        projected_features = self.projection(features)
        return projected_features


class TipPredictor(nn.Module):
    def __init__(self, num_channels):
        super(TipPredictor, self).__init__()

        self.resize = nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        conv_output_size = 64 * 64 * 32

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        out = self.fc_layers(x)
        return out


class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super(SinusoidalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)  # ignore parameter

    def forward(self, x):
        seq_len = x.size(1)  # [batch_size, seq_len, d_model]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden_size: int = 256 * 4):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_dim),
        )

    def forward(self, x):
        return self.ff(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.sa_norm = nn.LayerNorm(d_model)
        self.sa_dropout = nn.Dropout(dropout)

        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mha_norm = nn.LayerNorm(d_model)
        self.mha_dropout = nn.Dropout(dropout)

        self.ff = FeedForward(d_model, ff_dim)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, enc_outputs: Tensor, dec_inputs: Tensor, tgt_mask: Tensor, tgt_pad_mask: Tensor):
        """
        returns output (batch_size, seq_len, d_model), attentions (batch_size, n_heads, seq_len, patch_size^2)
        """
        output, _ = self.sa(dec_inputs, dec_inputs, dec_inputs, attn_mask=tgt_mask, key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.sa_dropout(output)
        output = self.sa_norm(output)

        output2, attentions = self.mha(output, enc_outputs, enc_outputs, average_attn_weights=False)
        output = output + self.mha_dropout(output2)
        output = self.mha_norm(output)

        output2 = self.ff(output)
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attentions


class TransformerDecoder(nn.Module):
    def __init__(self, layer: TransformerDecoderLayer, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, memory: Tensor, tgt: Tensor, tgt_mask: Tensor = None, tgt_pad_mask: Tensor = None, output_attentions=False
    ) -> Tuple[Tensor, Tensor]:
        tgt = self.dropout(tgt)

        all_attentions = []
        for layer in self.layers:
            tgt, attentions = layer(memory, tgt, tgt_mask, tgt_pad_mask)
            if output_attentions:
                all_attentions.append(attentions)
            all_attentions.append(attentions)
        if output_attentions:
            all_attentions = torch.stack(all_attentions, dim=1)
            return tgt, all_attentions
        return tgt, None
