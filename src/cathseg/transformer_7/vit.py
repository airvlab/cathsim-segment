import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
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
    def __init__(self, img_size: int = 224, n_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        super().__init__()
        self.img_size = img_size
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.emb_size = emb_size

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
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
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout_probs,
            bias=qkv_bias,
            batch_first=True,
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
            x, attention_probs = block(x, output_attentions)
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
        self.seq_len = self.patch_embedding.num_patches
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=self.seq_len)

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

    def forward(self, x, output_attentions=False):
        patch_embedding = self.patch_embedding(x)
        print("Patch Embedding: ", patch_embedding.shape)
        positional_encoding = self.positional_encoding(patch_embedding)
        print("Positional Encoding: ", positional_encoding.shape)
        out, attention_probs = self.transformer_encoder(positional_encoding, output_attentions=output_attentions)
        print("Encoder: ", out.shape)
        print("Attention Probs: ", len(attention_probs))

        return out, attention_probs


def visualize_attention(img, model):
    out, attention_probs = model(img, output_attentions=True)
    attention_maps = torch.stack(attention_probs, dim=0)  # Shape: [num_layers, batch_size, num_heads, seq_len, seq_len]
    attention_maps = attention_maps.transpose(0, 1)  # Shape: [batch_size, num_layers, num_heads, seq_len, seq_len]
    attention_maps = attention_maps.reshape(
        attention_maps.size(0), -1, attention_maps.size(-2), attention_maps.size(-1)
    )
    # Shape: [batch_size, num_layers*num_heads, seq_len, seq_len]

    # Example print output
    print("Concat att maps: ", attention_maps.shape)  # Should be [1, 48, 256, 256] with 8 heads and 6 layers

    # You might want to average across layers:
    attention_maps = attention_maps.mean(dim=1)  # Shape: [batch_size, seq_len, seq_len]
    print("Avg att maps: ", attention_maps.shape)

    # Reshape to the spatial dimensions
    num_patches = attention_maps.size(-1)  # 256 in your case
    size = int(math.sqrt(num_patches))  # Assuming this is square
    print("Size: ", size)
    # attention_maps = attention_maps.view(size, size)  # Shape: [size, size]
    attention_maps = attention_maps.squeeze(0).detach().cpu().numpy()  # Shape: [256, 256]
    attention_maps_resized = cv2.resize(attention_maps, (256, 256))  # Resizing for visualization

    # Optional: Interpolate to match original image size
    plt.figure(figsize=(8, 8))
    plt.imshow(np.zeros((256, 256, 3), dtype=np.uint8))
    plt.imshow(attention_maps_resized, cmap="jet", alpha=0.5)  # Adjust alpha for transparency
    plt.axis("off")
    plt.show()
    pass


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
        num_heads=num_heads,
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
