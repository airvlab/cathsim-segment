import math
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor


class TipPredictorBckup(nn.Module):
    def __init__(self, num_channels, image_size):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Calculate the size after the conv layers to connect to fully connected layers
        # Assuming image_size is square, i.e., (image_size, image_size)
        conv_output_size = image_size // (2**4)  # 4 maxpool layers reducing spatial size by 2 each
        flattened_size = 512 * conv_output_size * conv_output_size

        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        out = self.fc_layers(x)
        return out


class TipPredictor(nn.Module):
    def __init__(self, num_channels):
        super(TipPredictor, self).__init__()

        # Resize input to a fixed size
        self.resize = nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False)

        # Convolutional layers (deeper with more filters)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),  # Increase filters
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Increase filters
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Increase filters
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Deeper layer
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),  # Final downsample
        )

        # Calculate the output size after convolution and pooling layers
        conv_output_size = 16 * 16 * 256  # Based on the final output dimensions after pooling

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        out = self.fc_layers(x)
        return out


class TipPredictorMobileNetV2(nn.Module):
    def __init__(self, num_channels, image_size, num_dim=3, freeze_backbone=False):
        super().__init__()
        from torchvision import models

        # Load a lightweight pre-trained MobileNetV2 as a feature extractor
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # If num_channels is not 3 (default RGB), adjust the first layer
        if num_channels != 3:
            mobilenet.features[0][0] = nn.Conv2d(num_channels, 32, kernel_size=3, stride=8, padding=1)

        # We use all the convolutional layers of MobileNetV2, except for the last classifier
        self.feature_extractor = mobilenet.features

        # Freeze the backbone if specified
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Compute the output size of the feature extractor dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, num_channels, image_size, image_size)
            dummy_output = self.feature_extractor(dummy_input)
            conv_output_size = dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3]

        # Fully connected layers for regression
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_dim),
        )

    def forward(self, x):
        # Extract features using the pre-trained model
        x = self.feature_extractor(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        # Pass through fully connected layers
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


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable positional encodings
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))  # [1, max_len, d_model]
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)  # Initialize with a normal distribution

    def forward(self, x):
        seq_len = x.size(1)  # [batch_size, seq_len, d_model]
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
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
        attention_outtput, attentions = self.mha(x, x, x, average_attn_weights=False)
        x = self.mha_norm(x + attention_outtput)
        mlp_output = self.ff(x)
        x = self.ff_norm(x + mlp_output)

        return x, attentions


class TransformerEncoder(nn.Module):
    def __init__(self, layer=TransformerEncoderLayer, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

    def forward(self, src, output_attentions=False):
        all_attentions = []
        for layer in self.layers:
            x, attentions = layer(src)
            if output_attentions:
                all_attentions.append(attentions)
        if output_attentions:
            all_attentions = torch.stack(all_attentions, dim=1)
            return x, all_attentions
        return x, None


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
