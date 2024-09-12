import cathseg.splineformer.modules as modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SplineTransformer(nn.Module):
    def __init__(
        self,
        tgt_max_len: int,
        image_size: int = 256,
        num_channels: int = 1,
        patch_size: int = 16,
        d_model: int = 256,
        num_layers_encoder: int = 6,
        num_layers_decoder: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        dim_pts: int = 2,
    ):
        super().__init__()

        self.patch_embedding = modules.PatchEmbeddings(
            img_size=image_size, num_channels=num_channels, patch_size=patch_size, dim=d_model
        )
        self.src_seq_len = self.patch_embedding.num_patches
        self.positional_encoding = modules.SinusoidalEncoding(d_model=d_model, max_len=self.src_seq_len)

        transformer_encoder_layer = modules.TransformerEncoderLayer(
            d_model=d_model, num_heads=num_heads, ff_dim=d_model * 4, dropout=dropout
        )

        self.transformer_encoder = modules.TransformerEncoder(
            layer=transformer_encoder_layer,
            num_layers=num_layers_encoder,
        )

        self.target_embedding = nn.Linear(dim_pts + 1, d_model)
        self.target_sinuisodal_encoding = modules.SinusoidalEncoding(d_model=d_model, max_len=tgt_max_len)

        transformer_decoder_layer = modules.TransformerDecoderLayer(
            d_model=d_model, num_heads=num_heads, ff_dim=d_model * 4, dropout=dropout
        )

        self.transformer_decoder = modules.TransformerDecoder(
            layer=transformer_decoder_layer,
            num_layers=num_layers_decoder,
            dropout=dropout,
        )

        self.fc_knot = nn.Sequential(
            nn.Linear(d_model, 1),  # Predicting n_dim control and 1D knots
            nn.Softplus(),
        )
        self.fc_coeff = nn.Sequential(
            nn.Linear(d_model + 1, dim_pts),  # Predicting n_dim control and 1D knots
            nn.Softplus(),
        )
        self.fc_eos = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())  # Predicting end-of-sequence token

    def encode(self, src: Tensor, output_attentions=False) -> torch.Tensor:
        src = self.patch_embedding(src)
        src = self.positional_encoding(src)
        memory, attentions = self.transformer_encoder(src, output_attentions=output_attentions)
        return memory, attentions

    def decode(
        self, memory: Tensor, tgt: Tensor, tgt_mask: Tensor = None, tgt_pad_mask: Tensor = None, output_attentions=False
    ) -> Tensor:
        tgt = self.target_embedding(tgt)
        tgt = self.target_sinuisodal_encoding(tgt)
        output, attenntions = self.transformer_decoder(
            memory=memory, tgt=tgt, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask, output_attentions=output_attentions
        )
        return output, attenntions

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor = None,
        tgt_pad_mask: Tensor = None,
        output_attentions=False,
        output_memory=False,
    ):
        memory, encoder_attentions = self.encode(src=src, output_attentions=output_attentions)
        output, decoder_attentions = self.decode(
            memory=memory, tgt=tgt, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask, output_attentions=output_attentions
        )
        knot = self.fc_knot(output)
        coeff = self.fc_coeff(torch.cat([knot, output], dim=-1))
        seq = torch.cat((knot, coeff), dim=-1)
        eos = self.fc_eos(output)
        if output_memory:
            return seq, eos, encoder_attentions, decoder_attentions, memory
        return seq, eos, encoder_attentions, decoder_attentions, None


def main():
    import cathseg.splineformer_w_tip_pred.utils as utils

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

    # visualize_encoder_attention(X, encoder_atts, layer=0)
    utils.visualize_decoder_attention_instance(X[0], decoder_atts[0], layer=0, patch_size=patch_size)


if __name__ == "__main__":
    main()
