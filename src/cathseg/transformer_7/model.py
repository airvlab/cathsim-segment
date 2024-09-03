import torch
import torch.nn as nn
import torch.nn.functional as F
from cathseg.transformer_7.modules import (
    PatchEmbeddings,
    SinusoidalEncoding,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch import Tensor


class SplineTransformer(nn.Module):
    def __init__(
        self,
        tgt_max_len: int,
        image_size: int = 224,
        num_channels: int = 3,
        patch_size: int = 32,
        d_model: int = 256,
        num_layers_encoder: int = 6,
        num_layers_decoder: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        dim_pts: int = 2,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbeddings(
            img_size=image_size, num_channels=num_channels, patch_size=patch_size, dim=d_model
        )
        self.src_seq_len = self.patch_embedding.num_patches
        self.positional_encoding = SinusoidalEncoding(d_model=d_model, max_len=self.src_seq_len)

        transformer_encoder_layer = TransformerEncoderLayer(
            d_model=d_model, num_heads=num_heads, ff_dim=d_model * 4, dropout=dropout
        )

        self.transformer_encoder = TransformerEncoder(
            layer=transformer_encoder_layer,
            num_layers=num_layers_encoder,
        )

        self.target_embedding = nn.Linear(dim_pts + 1, d_model)
        self.target_sinuisodal_encoding = SinusoidalEncoding(d_model=d_model, max_len=tgt_max_len)

        transformer_decoder_layer = TransformerDecoderLayer(
            d_model=d_model, num_heads=num_heads, ff_dim=d_model * 4, dropout=dropout
        )

        self.transformer_decoder = TransformerDecoder(
            layer=transformer_decoder_layer,
            num_layers=num_layers_decoder,
            dropout=dropout,
        )

        self.fc_seq = nn.Sequential(nn.Linear(d_model, 1 + dim_pts))  # Predicting n_dim control and 1D knots
        self.fc_eos = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())  # Predicting end-of-sequence token

    def encode(self, src: Tensor) -> torch.Tensor:
        src = self.patch_embedding(src)
        src = self.positional_encoding(src)
        memory, attentions = self.transformer_encoder(src)
        return memory, attentions

    def decode(self, memory: Tensor, tgt: Tensor, tgt_mask: Tensor = None, tgt_pad_mask: Tensor = None) -> Tensor:
        tgt = self.target_embedding(tgt)
        tgt = self.target_sinuisodal_encoding(tgt)
        output, attenntions = self.transformer_decoder(
            memory=memory, tgt=tgt, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask
        )
        return output, attenntions

    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor = None, tgt_pad_mask: Tensor = None):
        memory, encoder_attentions = self.encode(src=src)
        output, decoder_attentions = self.decode(memory=memory, tgt=tgt, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask)
        seq = self.fc_seq(output)
        eos = self.fc_eos(output)
        return seq, eos, memory, encoder_attentions, decoder_attentions


def main():
    num_channels = 1
    patch_size = 32
    image_size = 1024
    seq_len = 8
    tgt_max_len = 20

    X = torch.rand(1, num_channels, image_size, image_size)
    tgt = torch.rand(1, 8, 3)
    tgt = F.pad(tgt, (0, 0, 0, tgt_max_len - seq_len))

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_max_len)
    tgt_pad_mask = torch.ones(tgt_max_len, dtype=torch.int32)
    tgt_pad_mask[seq_len:] = 0
    tgt_pad_mask = tgt_pad_mask.unsqueeze(0).to(torch.float)

    model = SplineTransformer(
        tgt_max_len=tgt_max_len,
        image_size=image_size,
        num_channels=num_channels,
        patch_size=patch_size,
        d_model=256,
        num_layers_encoder=6,
        num_heads=8,
    )

    seq, eos, memory, encoder_atts, decoder_atts = model(src=X, tgt=tgt, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask)
    print("Input: ", X.shape)
    print("Target: ", tgt.shape)
    print("Sequence: ", seq.shape)
    print("End-of-sequence: ", eos.shape)
    print("Memory: ", memory.shape)
    print("Encoder attentions: ", encoder_atts.shape)
    print("Decoder attentions: ", decoder_atts.shape)


if __name__ == "__main__":
    main()
