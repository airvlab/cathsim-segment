import math

import torch
import torch.functional as F
import torch.nn as nn


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ImageEncoder(nn.Module):
    def __init__(self, input_channels=3, feature_dim=256):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        seq_len,
        num_layers=6,
        nhead=8,
        d_model=256,
        dim_feedforward=512,
    ):
        super(TransformerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.linear_projection = nn.Linear(d_model, feature_dim)

    def forward(self, memory, tgt):
        tgt = self.positional_encoding(tgt)
        output = self.transformer_decoder(tgt, memory)
        output = self.linear_projection(output)
        return output


class ImageToSequenceModel(nn.Module):
    def __init__(
        self,
        input_channels,
        feature_dim,
        seq_len,
        num_layers=6,
        nhead=8,
        d_model=256,
        dim_feedforward=512,
    ):
        super(ImageToSequenceModel, self).__init__()
        self.encoder = ImageEncoder(input_channels=input_channels, feature_dim=d_model)
        self.decoder = TransformerDecoder(
            feature_dim, seq_len, num_layers, nhead, d_model, dim_feedforward
        )
        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def forward(self, img, tgt_seq):
        # Encode the image
        memory = self.encoder(img).unsqueeze(0)  # Add a sequence dimension

        # Create target sequence embeddings
        tgt = torch.zeros(
            (self.seq_len, memory.size(1), self.feature_dim), device=img.device
        )
        tgt[: tgt_seq.size(0), :, :] = tgt_seq

        # Decode to get the sequence
        output_seq = self.decoder(memory, tgt)
        return output_seq


def loss_function(pred_seq, true_seq, mask):
    loss = F.mse_loss(pred_seq * mask, true_seq * mask)
    return loss


def train(num_epochs, dataloader, max_len, device="cuda"):
    # Initialize model, optimizer, and loss function
    model = ImageToSequenceModel(
        input_channels=1, feature_dim=2, seq_len=max_len
    )  # Example with 2D control points and sequence length of 50
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(model)
    exit()

    # Example training loop
    for epoch in range(num_epochs):
        for img, tgt_seq, tgt_mask in dataloader:
            img = img.to(device)
            tgt_seq = tgt_seq.to(device)
            tgt_mask = tgt_mask.to(device)

            optimizer.zero_grad()
            pred_seq = model(img, tgt_seq)
            loss = loss_function(pred_seq, tgt_seq, tgt_mask)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


if __name__ == "__main__":
    from pathlib import Path

    from guide3d.dataset.image.spline import Guide3D

    max_len = 50

    root = Path("/mnt/data/data/segment-real")
    dataset = Guide3D(root, split="train", with_mask=True, max_length=max_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    train(1, dataloader, max_len)
