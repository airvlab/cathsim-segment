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


class SplineTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        num_channels: int = 3,
        patch_size: int = 32,
        d_model: int = 256,
        num_layers_encoder: int = 6,
        num_layers_decoder: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        dim_pts: int = 2,
        tgt_max_len: int = 20,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbeddings(
            img_size=image_size, num_channels=num_channels, patch_size=patch_size, embed_dim=d_model
        )
        self.src_seq_len = self.patch_embedding.num_patches
        self.positional_encoding = SinusoidalEncoding(d_model=d_model, max_len=self.src_seq_len)

        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=d_model * 4,
            dropout=dropout,
        )

        self.transformer_encoder = TransformerEncoder(
            layer=self.transformer_encoder_layer,
            num_layers=num_layers_encoder,
        )

        self.target_embedding = nn.Linear(dim_pts + 1, d_model)
        self.target_sinuisodal_encoding = SinusoidalEncoding(d_model=d_model, max_len=tgt_max_len)

        self.transformer_decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=d_model * 4,
            dropout=dropout,
        )

        self.transformer_decoder = TransformerDecoder(
            layer=self.transformer_decoder_layer,
            num_layers=num_layers_decoder,
            dropout=dropout,
        )

        self.fc_seq = nn.Sequential(nn.Linear(d_model, 1 + dim_pts))  # Predicting n_dim control and 1D knots
        self.fc_eos = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())  # Predicting end-of-sequence token

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        src = self.patch_embedding(src)
        src = self.positional_encoding(src)
        memory, attentions = self.transformer_encoder(src)
        return memory, attentions

    def decode(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        tgt = self.target_embedding(tgt)
        tgt = self.target_sinuisodal_encoding(tgt)
        output, attenntions = self.transformer_decoder(
            memory=memory, tgt=tgt, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask
        )
        return output, attenntions

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_pad_mask: torch.Tensor = None,
    ):
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
    max_seq_len = 20

    X = torch.rand(1, num_channels, image_size, image_size)
    tgt = torch.rand(1, 8, 3)
    tgt = F.pad(tgt, (0, 0, 0, max_seq_len - seq_len))
    tgt_key_padding_mask = torch.ones(max_seq_len, dtype=torch.int32)
    tgt_key_padding_mask[seq_len:] = 0
    tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(0)

    model = SplineTransformer(
        image_size=image_size,
        num_channels=num_channels,
        patch_size=patch_size,
        d_model=256,
        num_layers_encoder=6,
        num_heads=8,
    )

    seq, eos, memory, attentions = model(src=X, tgt=tgt, tgt_key_padding_mask=tgt_key_padding_mask)
    print("Sequence: ", seq.shape)
    print("End-of-sequence: ", eos.shape)


if __name__ == "__main__":
    main()
