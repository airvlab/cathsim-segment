import torch
import torch.nn as nn
import torch.nn.functional as F
from cathseg.transformer_7.modules import PatchEmbeddings, SinusoidalEncoding, TransformerEncoder


class SplineTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        num_channels: int = 3,
        patch_size: int = 32,
        embed_dim: int = 256,
        num_layers_encoder: int = 6,
        num_layers_decoder: int = 6,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attention_dropout_probs: float = 0.0,
        hidden_dropout_prob: float = 0.1,
        transformer_decoder_dropout_prob: float = 0.1,
        mlp_dropout_prob: float = 0.1,
        dim_pts: int = 2,
        tgt_max_len: int = 20,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbeddings(
            img_size=image_size, num_channels=num_channels, patch_size=patch_size, embed_dim=embed_dim
        )
        self.seq_len = self.patch_embedding.num_patches
        self.positional_encoding = SinusoidalEncoding(d_model=embed_dim, max_len=self.seq_len)

        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers_encoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_intermed_size=embed_dim * 4,
            qkv_bias=qkv_bias,
            attention_dropout_probs=attention_dropout_probs,
            mlp_dropout_prob=mlp_dropout_prob,
        )

        self.target_embedding = nn.Linear(dim_pts + 1, embed_dim)
        self.target_sinuisodal_encoding = SinusoidalEncoding(d_model=embed_dim, max_len=tgt_max_len)
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            batch_first=True,
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=transformer_decoder_dropout_prob,
        )
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers_decoder)

        self.fc_seq = nn.Sequential(nn.Linear(embed_dim, 1 + dim_pts))  # Predicting n_dim control and 1D knots
        self.fc_eos = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())  # Predicting end-of-sequence token

    def encode(self, src: torch.Tensor, output_attentions=False) -> torch.Tensor:
        src = self.patch_embedding(src)
        src = self.positional_encoding(src)
        memory, attention_weight = self.transformer_encoder(src, output_attentions=output_attentions)
        return memory, attention_weight

    def decode(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        tgt = self.target_embedding(tgt)
        tgt = self.target_sinuisodal_encoding(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device=tgt.device)

        output = self.transformer_decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )

        return output

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        output_attentions=False,
    ):
        memory, attention_weights = self.encode(src=src, output_attentions=output_attentions)
        output = self.decode(memory=memory, tgt=tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        seq = self.fc_seq(output)
        eos = self.fc_eos(output)
        return seq, eos, memory, attention_weights


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
        embed_dim=256,
        num_layers_encoder=6,
        num_heads=8,
    )

    seq, eos, memory, attentions = model(src=X, tgt=tgt, tgt_key_padding_mask=tgt_key_padding_mask)
    print("Sequence: ", seq.shape)
    print("End-of-sequence: ", eos.shape)


if __name__ == "__main__":
    main()
