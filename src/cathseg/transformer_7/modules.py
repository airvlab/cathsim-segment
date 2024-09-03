import math

import torch
import torch.nn as nn


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


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_intermed_size: int = 256 * 4,
        attention_dropout_probs: float = 0.0,
        mlp_dropout_prob: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.mha_layer = nn.MultiheadAttention(
            batch_first=True,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout_probs,
            bias=qkv_bias,
        )

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.mlp_layer = MLP(embed_dim, mlp_intermed_size, mlp_dropout_prob)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x, output_attentions=False):
        attention_outtput, attention_weight = self.mha_layer(x, x, x)
        x = self.layer_norm_1(x + attention_outtput)  # skip connection
        mlp_output = self.mlp_layer(x)
        x = self.layer_norm_2(x + mlp_output)  # skip connection

        if output_attentions:
            return (x, attention_weight)
        else:
            return x, None


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int = 6, **transformer_kwargs):
        super().__init__()

        self.num_layers = num_layers
        self.blocks = nn.ModuleList([TransformerEncoderBlock(**transformer_kwargs) for _ in range(num_layers)])

    def forward(self, x, output_attentions=False):
        all_attentions = []  # per layer attention
        for layer in self.blocks:
            x, attention_probs = layer(x, output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)

        if output_attentions:
            return (x, all_attentions)
        return x, None
