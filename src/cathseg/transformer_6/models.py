import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbeddings(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        n_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.emb_size = emb_size

        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(self.n_channels, self.emb_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor):
        x = self.projection(x)  # (batch_size, emb_size, n_patches, n_patches)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, n_patches^2, emb_size)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.
    """

    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        qkv_bias=True,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = qkv_bias
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size, self.attention_head_size, attention_probs_dropout_prob, self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        qkv_bias=True,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size, num_attention_heads, qkv_bias, attention_probs_dropout_prob, hidden_dropout_prob
        )
        self.layernorm_1 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, hidden_dropout_prob)
        self.layernorm_2 = nn.LayerNorm(hidden_size)

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        qkv_bias=True,
        config=None,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.1,
    ):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(num_hidden_layers):
            block = Block(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                qkv_bias,
                attention_probs_dropout_prob,
                hidden_dropout_prob,
            )
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


def main():
    n_channels = 3
    img_size = 224
    patch_size = 16
    emb_size = 768
    num_attention_heads = 12
    intermediate_size = 4 * emb_size
    hidden_dropout_prob = 0
    attention_probs_dropout_prob = 0

    X = torch.rand(1, n_channels, img_size, img_size)
    patch_embeddings_model = PatchEmbeddings(img_size, n_channels, patch_size, emb_size)
    out = patch_embeddings_model(X)
    print(out.shape)
    positional_encoding_layer = PositionalEncoding(d_model=emb_size, max_len=patch_embeddings_model.num_patches)
    out = positional_encoding_layer(out)
    print(out.shape)
    transformer_block = Block(
        emb_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob, hidden_dropout_prob
    )
    out, attention_maps = transformer_block(out, output_attentions=True)
    print(out.shape)
    print("attention_maps", attention_maps.shape)
    attention_maps = attention_maps[:, :, 0, 1:]
    print("attention_maps", attention_maps.shape)
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)

    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode="bilinear", align_corners=False)
    attention_maps = attention_maps.squeeze(1)


if __name__ == "__main__":
    main()
