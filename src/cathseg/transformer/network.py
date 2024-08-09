import pytorch_lightning as pl
import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, o_net_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)
        self.o_net = nn.Sequential(
            nn.Linear(embed_dim, o_net_dim), nn.ReLU(), nn.Linear(o_net_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.pre_norm(x)

        # Feed-forward
        output = self.o_net(x)
        x = x + self.dropout(output)
        x = self.post_norm(x)

        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self, input_dim, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1
    ):
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ff_out = nn.Linear(embed_dim, embed_dim)
        self.if_ffout = True
        self.downstream_decoder = nn.Linear(embed_dim, input_dim)

    def forward(self, x, mask=None):
        # x: [B, N]
        B, N, C = x.shape
        x = self.embedding(x)  # [B, N, C]
        x = x.transpose(0, 1)  # [N, B, C] for compatibility with nn.MultiheadAttention

        # Pass through layers
        for layer in self.layers:
            x = layer(x, mask)

        x = x.transpose(0, 1)  # [B, N, C]
        x = self.norm(x)
        if self.if_ffout:
            x = self.ff_out(x)

        output_mask = None
        return x, output_mask

    def generate(self, x, initial_mask=None):
        new_tokens = 0
        max_tokens = 10

        def if_terminate(x, new_tokens_):
            if new_tokens_ > max_tokens:
                return True
            return False

        output = x
        while not if_terminate(output, new_tokens):
            output_all, _ = self.forward(output, None)
            output_last = output_all[:, -1, :]
            output_last = self.downstream_decoder(output_last)
            output = torch.cat([output, output_last.unsqueeze(1)], dim=1)
            new_tokens += 1
            print(
                "The {}-th token is generated. Shape of the sequence: {}".format(
                    new_tokens, output.shape
                )
            )
        return output


class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.image_encoder = None
        self.transformer = DecoderOnlyTransformer(256 + 20, 512, 8, 2048, 6, 0.1)


batch_size = 32
num_tokens = 20
input_dim = 256 + 20
embed_dim = 512  # Embedding dimension
num_heads = 8  # Number of attention heads
o_net_dim = 2048
num_layers = 6  # Number of transformer layers
dropout = 0.1  # Dropout rate

# Initialize the model
model = DecoderOnlyTransformer(
    input_dim, embed_dim, num_heads, o_net_dim, num_layers, dropout
)

print("Test forward pass...")
input_tokens = torch.rand_like(torch.zeros((batch_size, num_tokens, input_dim)))
print("Input shape:", input_tokens.shape)
output, _ = model(input_tokens)
print("Output shape:", output.shape)

# Generate new tokens
print("Generating new tokens...")
generated = model.generate(input_tokens)
print(generated.shape)
