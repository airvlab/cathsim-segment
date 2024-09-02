import functools as ft
import math

import numpy as np
import torch
import torch as th
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, nb_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.nbr_heads = nb_heads
        self.heads_dim = in_dim // nb_heads
        self.qkv_layer = nn.Linear(in_dim, 3 * in_dim)
        self.out_layer = nn.Linear(in_dim, in_dim)

    def forward(self, src, mask=None, key_padding_mask=None):
        bt_size, seq_length, _ = src.shape  # unpack shape

        qkv = self.qkv_layer(src)  # extract query, key and value
        qkv = qkv.reshape(bt_size, seq_length, self.nbr_heads, 3 * self.heads_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # permute head and sequence
        qry, key, val = th.chunk(qkv, 3, dim=-1)

        dim = qry.shape[-1]
        wgt = qry @ key.transpose(-2, -1)  # hidden_dim and sequence_dim
        wgt = wgt / np.sqrt(dim)  # normalize
        if mask is not None:
            wgt = wgt.masked_fill(mask, float("-inf"))
        if key_padding_mask is not None:
            cnd = key_padding_mask[:, None, None, :]
            wgt = wgt.masked_fill(cnd, float("-inf"))
        wgt = th.softmax(wgt, dim=-1)

        res = wgt @ val
        res = res.permute(0, 2, 1, 3)  # permute head and sequence
        res = th.flatten(res, start_dim=2)  # concat over heads
        res = self.out_layer(res)

        return res


class FeedForwardNetwork(nn.Module):
    __THETA = {  # map id to non_linear
        0: nn.Identity(),
        1: nn.ReLU(),
        2: nn.GELU(),
        3: nn.Sigmoid(),
        4: nn.Tanh(),
        5: nn.Softmax(dim=-1),
    }

    def __init__(self, layer_cfg, activations, drop_vals):
        super(FeedForwardNetwork, self).__init__()
        self.shapes = list(zip(layer_cfg[:-1], layer_cfg[1:]))
        self.linears = nn.ModuleList([])
        for idx, (in_dim, out_dim) in enumerate(self.shapes):
            fn_id = activations[idx]
            proba = drop_vals[idx]
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(proba) if proba > 0.0 else nn.Identity(),
                FeedForwardNetwork.__THETA.get(fn_id, nn.Identity()),
            )
            self.linears.append(block)

    def forward(self, input_batch):
        output_batch = ft.reduce(  # functools
            lambda acc, crr: crr(acc), self.linears, input_batch
        )
        return output_batch


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, ff_dim, nb_heads, drop_val=0.1, pre_norm=False):
        super(EncoderBlock, self).__init__()
        assert in_dim % nb_heads == 0

        self.nbr_heads = nb_heads
        self.heads_dim = in_dim // nb_heads

        self.mha_layer = MultiHeadSelfAttention(in_dim, nb_heads)
        self.ffn_layer = FeedForwardNetwork([in_dim, ff_dim, in_dim], [1, 0], [drop_val, 0.0])

        self.dropout_layer = nn.ModuleDict({"mha": nn.Dropout(drop_val), "ffn": nn.Dropout(drop_val)})
        self.layer_normalz = nn.ModuleDict(
            {
                "mha": nn.ModuleList(
                    [
                        nn.LayerNorm(in_dim) if pre_norm else nn.Identity(),
                        nn.LayerNorm(in_dim) if not pre_norm else nn.Identity(),
                    ]
                ),
                "ffn": nn.ModuleList(
                    [
                        nn.LayerNorm(in_dim) if pre_norm else nn.Identity(),
                        nn.LayerNorm(in_dim) if not pre_norm else nn.Identity(),
                    ]
                ),
            }
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # multi head self attention
        tmp = self.layer_normalz["mha"][0](src)
        out = self.mha_layer(tmp, src_mask, src_key_padding_mask)
        out = self.dropout_layer["mha"](out)
        agg = tmp + out
        agg = self.layer_normalz["mha"][1](agg)

        # feed forward network
        tmp = self.layer_normalz["ffn"][0](agg)
        out = self.ffn_layer(tmp)
        out = self.dropout_layer["ffn"](out)
        agg = tmp + out
        agg = self.layer_normalz["ffn"][1](agg)

        return agg


class PatchEmbeddings(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        n_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.emb_size = emb_size

        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(self.n_channels, self.emb_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor):
        x = self.projection(x)  # (batch_size, emb_size, n_patches, n_patches)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, n_patches^2, emb_size)
        return x


class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_intermed_size, dropout_prob):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_intermed_size),
            nn.GELU(),
            nn.Linear(mlp_intermed_size, hidden_size),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.mlp(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        mlp_intermed_size,
        qkv_bias,
        attention_dropout_probs,
        hidden_dropout_prob,
        mlp_dropout_prob,
    ):
        super().__init__()

        self.mha_layer = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_attention_heads, dropout=attention_dropout_probs, bias=qkv_bias
        )

        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.mlp_layer = MLP(hidden_size, mlp_intermed_size, mlp_dropout_prob)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_outtput, attention_weight = self.mha_layer(x, x, x)
        # Skip connection
        x = x + attention_outtput
        # Feed-forward
        mlp_output = self.mlp_layer(x)
        # Skip connection
        x = x + mlp_output
        if output_attentions:
            return (x, attention_weight)
        else:
            return x, None


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **kwargs):
        super().__init__()

        self.num_layers = num_layers
        self.blocks = nn.ModuleList([TransformerEncoderBlock(**kwargs) for _ in range(num_layers)])

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x)
            if output_attentions:
                all_attentions.append(attention_probs)

        if output_attentions:
            return x, all_attentions
        return x, None


class Encoder(nn.Module):
    def __init__(
        self,
        image_size,
        n_channels,
        patch_size,
        embed_dim,
        num_layers,
        num_heads,
        qkv_bias,
        attention_dropout_probs,
        hidden_dropout_prob,
        mlp_dropout_prob,
    ):
        super().__init__()

        self.n_patches = (image_size // patch_size) ** 2
        print(self.n_patches)

        self.patch_embedding = PatchEmbeddings(image_size, n_channels, patch_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=self.patch_embedding.num_patches)

        self.transformer_encoder = TransformerEncoder(
            num_layers,
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            mlp_intermed_size=embed_dim * 4,
            qkv_bias=qkv_bias,
            attention_dropout_probs=attention_dropout_probs,
            hidden_dropout_prob=hidden_dropout_prob,
            mlp_dropout_prob=mlp_dropout_prob,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        patch_embedding = self.patch_embedding(x)
        print("Patch Embedding: ", patch_embedding.shape)
        positional_encoding = self.positional_encoding(patch_embedding)
        print("Positional Encoding: ", positional_encoding.shape)
        out, attention_probs = self.transformer_encoder(positional_encoding, output_attentions=True)
        print("Encoder: ", out.shape)
        print("Attention Probs: ", len(attention_probs))
        return out


class SplineTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        n_channels,
        patch_size,
        embed_dim,
        num_layers,
        num_heads,
        qkv_bias,
        attention_dropout_probs,
        hidden_dropout_prob,
        mlp_dropout_prob,
    ):
        super().__init__()

        self.n_patches = (image_size // patch_size) ** 2
        self.encoder = Encoder(
            image_size,
            n_channels,
            patch_size,
            embed_dim,
            num_layers,
            num_heads,
            qkv_bias,
            attention_dropout_probs,
            hidden_dropout_prob,
            mlp_dropout_prob,
        )

        self.src_pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=self.n_patches)

    def forward(self, x):
        x = self.encoder(x)
        x = self.layer_norm(x)
        return x


def main():
    img_size = 256
    n_channels = 3
    patch_size = 16
    embed_dim = 512
    num_layers = 6
    num_heads = 8

    encoder = Encoder(
        img_size,
        n_channels,
        patch_size,
        embed_dim,
        num_layers,
        num_heads,
        qkv_bias=True,
        attention_dropout_probs=0.0,
        hidden_dropout_prob=0.0,
        mlp_dropout_prob=0.0,
    )

    X = torch.rand(1, n_channels, img_size, img_size)
    out = encoder(X)
    print("Final Output: ", out.shape)


if __name__ == "__main__":
    main()
