import math
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
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
    def __init__(self, img_size: int = 224, num_channels: int = 3, patch_size: int = 16, dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = dim

        patch_width, patch_height = patch_size, patch_size
        patch_dim = num_channels * patch_size * patch_size

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.to_patch_embedding(x)
        return x


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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8, ff_dim: int = 256 * 4, dropout: float = 0.00):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            batch_first=True,
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ff = FeedForward(d_model, ff_dim)

        self.mha_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_outtput, attentions = self.mha(x, x, x)
        x = self.mha_norm(x + attention_outtput)
        mlp_output = self.ff(x)
        x = self.ff_norm(x + mlp_output)

        return x, attentions


class TransformerEncoder(nn.Module):
    def __init__(self, layer=TransformerEncoderLayer, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

    def forward(self, src):
        all_attentions = []
        for layer in self.layers:
            x, attentions = layer(src)
            all_attentions.append(attentions)
        all_attentions = torch.stack(all_attentions, dim=1)
        return x, all_attentions


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
    def __init__(self, layer: TransformerDecoderLayer, num_layers: int, dropout: float):
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
        all_attentions = torch.stack(all_attentions, dim=1)
        return tgt, all_attentions
