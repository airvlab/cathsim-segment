import math
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super(SinusoidalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Unsqueeze and transpose to make it [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as a buffer so it's not considered as a parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x is assumed to have shape [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        # Add positional encoding to the input tensor
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class PatchEmbeddings(nn.Module):
    def __init__(self, img_size: int = 224, num_channels: int = 3, patch_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            self.num_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x: torch.Tensor):
        x = self.projection(x)  # (batch_size, emb_size, n_patches, n_patches)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, n_patches^2, emb_size)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden_size: int = 256 * 4, dropout_prob: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.mlp(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        ff_dim: int = 256 * 4,
        dropout: float = 0.00,
    ):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            batch_first=True,
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.mlp = MLP(d_model, ff_dim, dropout)

        self.mha_norm = nn.LayerNorm(d_model)
        self.mlp_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_outtput, attentions = self.mha(x, x, x)
        x = self.mha_norm(x + attention_outtput)
        mlp_output = self.mlp(x)
        x = self.mlp_norm(x + mlp_output)

        return x, attentions


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        layer=TransformerEncoderLayer,
        num_layers: int = 6,
    ):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

    def forward(self, src):
        all_attentions = []
        for layer in self.layers:
            x, attentions = layer(src)
            all_attentions.append(attentions)
        all_attentions = torch.stack(all_attentions)
        return x, all_attentions


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.sa_norm = nn.LayerNorm(d_model)
        self.mha_norm = nn.LayerNorm(d_model)

        self.sa_dropout = nn.Dropout(dropout)
        self.mha_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(ff_dim, d_model),
        )

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, enc_outputs: Tensor, dec_inputs: Tensor, tgt_mask: Tensor, tgt_pad_mask: Tensor):
        output, _ = self.sa(dec_inputs, dec_inputs, dec_inputs, attn_mask=tgt_mask, key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.sa_dropout(output)
        output = self.sa_norm(output)

        output2, attentions = self.mha(output, enc_outputs, enc_outputs)
        output = output + self.mha_dropout(output2)
        output = self.mha_norm(output)

        output2 = self.ff(output)
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attentions


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        layer: TransformerDecoderLayer,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, memory: Tensor, tgt: Tensor, tgt_mask: Tensor = None, tgt_pad_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        tgt = self.dropout(tgt)

        all_attentions = []
        for layer in self.layers:
            tgt, attentions = layer(memory, tgt, tgt_mask, tgt_pad_mask)
            all_attentions.append(attentions)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        all_attentions = torch.stack(all_attentions)
        return tgt, all_attentions
