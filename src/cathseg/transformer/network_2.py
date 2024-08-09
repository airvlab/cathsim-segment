import math

import torch
import torch.nn as nn


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
        self.output_layer = nn.Linear(d_model, d_model)

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


# Parameters
d_model = 8
nhead = 8
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
max_seq_length = 3  # Maximum length of the sequence we want to generate
batch_size = 4

# Create the Transformer Decoder Model
model = TransformerDecoderModel(
    d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length, dropout
)

# Example input: Image features
# Let's assume we have a batch of 32 images, and each image is represented as a 512-dimensional vector (the output from some feature extractor like a CNN)
memory = torch.rand(
    batch_size, 216, d_model
)  # (batch_size, memory_seq_length, d_model)

# Example input: Target sequence (initialized as zeros or random for this example)
# For simplicity, we'll assume we start with an all-zero sequence of embeddings
tgt = torch.zeros(
    batch_size, max_seq_length, d_model
)  # (batch_size, tgt_seq_length, d_model)

# Generate target mask (causal mask for autoregression)
tgt_mask = nn.Transformer.generate_square_subsequent_mask(max_seq_length)

# Forward pass to generate the sequence
output = model(memory, tgt, tgt_mask=tgt_mask)
print(output.shape)  # Should be (batch_size, max_seq_length, d_model)
