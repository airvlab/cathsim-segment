import math
from pathlib import Path

import torch
import torch.nn as nn
from guide3d.dataset.image.spline import Guide3D as Dataset
from torchvision import models

dataset_path = Path.home() / "data" / "segment-real"


class ImageFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageFeatureExtractor, self).__init__()
        # Using ResNet as the feature extractor
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-2]
        )  # Remove last layers

    def forward(self, x):
        # Extract features
        features = self.resnet(x)  # (batch_size, 2048, H, W)
        features = features.view(
            features.size(0), features.size(1), -1
        )  # (batch_size, 2048, H*W)
        features = features.permute(0, 2, 1)  # (batch_size, H*W, 2048)
        return features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerDecoderModel(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        dropout=0.1,
    ):
        super(TransformerDecoderModel, self).__init__()

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Linear layer to project decoder output to the desired sequence space
        self.output_layer = nn.Linear(d_model, 4)

    def forward(self, memory, tgt, tgt_mask=None, memory_mask=None):
        # Positional encoding for the target sequence
        tgt = self.positional_encoding(tgt)

        # Pass through the transformer decoder
        output = self.transformer_decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask
        )

        # Project the decoder output to the prediction space (in this case, we keep it the same size)
        output = self.output_layer(output)

        return output


def main():
    model_dimensionality = 256
    n_heads = 8
    num_decoder_layers = 6
    dim_feedforward = 512
    dropout = 0.1
    max_seq_length = 20
    batch_size = 4

    train_dataset = Dataset(
        root=dataset_path,
        annotations_file="sphere_wo_reconstruct.json",
        split="train",
        with_mask=True,
        max_length=max_seq_length,
    )

    # Create the Transformer Decoder Model
    model = TransformerDecoderModel(
        model_dimensionality,
        n_heads,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        dropout,
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(max_seq_length)

    for i, batch in enumerate(dataloader):
        img, ts, cs, seq_lengths, masks = batch
        print(seq_lengths.shape)
        exit()
        ts = ts.unsqueeze(-1)
        tgt = torch.cat([ts, cs, seq_lengths], dim=-1)
        memory = torch.rand(batch_size, 216, model_dimensionality)
        output = model(memory, tgt, tgt_mask=tgt_mask)
        print(output.shape)  # Should be (batch_size, max_seq_length, d_model)
        exit()


if __name__ == "__main__":
    main()
