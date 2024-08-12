import math

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from cathseg.utils import DummyData
from torchvision.models import ViT_B_16_Weights, vit_b_16

MAX_LEN = 20


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=MAX_LEN):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PretrainedViTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(PretrainedViTEncoder, self).__init__()

        # Load the pretrained ViT model
        if pretrained:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16(weights=None)

        # Remove the classification head
        self.vit.heads = nn.Identity()

    def forward(self, x):
        # Forward pass through the ViT model
        x = self.vit(x)
        return x  # Returns the encoded features


class ViTEncoder(nn.Module):
    def __init__(self, n_channels, image_shape, output_dim=512, pretrained=True):
        super(ViTEncoder, self).__init__()

        # Load the pretrained ViT model
        if pretrained:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16(weights=None)

        # Adjust the input layer if the number of channels is different from 3
        if n_channels != 3:
            self.vit.conv_proj = nn.Conv2d(
                n_channels, self.vit.conv_proj.out_channels, kernel_size=16, stride=16
            )

        # Remove the classification head
        self.vit.heads = nn.Identity()

        # Add a projection layer to match the desired output dimension if specified
        self.output_dim = output_dim
        if output_dim != self.vit.hidden_dim:
            self.proj = nn.Linear(self.vit.hidden_dim, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        # Forward pass through the ViT model
        x = self.vit(x)

        # Project to the desired output dimension if necessary
        x = self.proj(x)

        return x


class ImageToSequenceTransformer(pl.LightningModule):
    def __init__(
        self,
        img_size=(224, 224),
        d_model=512,
        num_decoder_layers=6,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=MAX_LEN,
    ):
        super(ImageToSequenceTransformer, self).__init__()

        # Encoder
        self.encoder = ViTEncoder(3, img_size, d_model, pretrained=True)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        # Decoder-only Transformer with masked self-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output layers for coefficients `c` and knots `t`
        self.fc_c = nn.Linear(d_model, 2)  # Predicting 2D control points
        self.fc_t = nn.Linear(d_model, 1)  # Predicting 1D knots
        self.fc_eos = nn.Linear(d_model, 1)  # Predicting stop token

        # Embedding layer for target sequence
        self.target_embedding = nn.Linear(
            3, d_model
        )  # We concatenate c (2D) and t (1D) to form a 3D input for the next step

        self.criterion_c = nn.MSELoss(reduction="none")
        self.criterion_t = nn.MSELoss(reduction="none")
        self.criterion_eos = nn.BCELoss(reduction="none")

    def forward(self, x, target_seq, target_mask):
        # Feature extraction
        features = self.encoder(x)  # Output shape: (batch_size, 512, H, W)
        features = features.unsqueeze(
            0
        )  # Shape to (1, batch_size, d_model), to match transformer input format

        # Target sequence embedding and positional encoding
        target_seq = self.target_embedding(
            target_seq
        )  # Shape to (batch_size, seq_len, d_model)
        target_seq = self.pos_encoder(
            target_seq.permute(1, 0, 2)
        )  # Shape to (seq_len, batch_size, d_model)

        # Create a mask to hide future tokens (seq_len, seq_len)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            target_seq.size(0)
        ).to(target_seq.device)

        tgt_key_padding_mask = (target_mask == 0).to(
            target_seq.device
        )  # Shape: (batch_size, seq_len)

        # Transformer Decoder with masked self-attention and key padding mask
        decoder_output = self.transformer_decoder(
            tgt=target_seq,
            memory=features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Output prediction
        c_pred = self.fc_c(decoder_output)  # Shape: (seq_len, batch_size, 2)
        t_pred = self.fc_t(decoder_output)  # Shape: (seq_len, batch_size, 1)
        eos_pred = torch.sigmoid(self.fc_eos(decoder_output))

        return (
            c_pred.permute(1, 0, 2),
            t_pred.permute(1, 0, 2),
            eos_pred.permute(1, 0, 2),
        )

    def training_step(self, batch, batch_idx):
        X, target_seq, target_mask = batch

        # Create EOS labels: 1 for the last valid token, 0 otherwise
        eos_labels = torch.zeros_like(target_mask)
        eos_labels[
            torch.arange(target_mask.size(0)), (target_mask.sum(dim=1) - 1).long()
        ] = 1
        eos_labels = eos_labels.float()

        # Forward pass
        c_pred, t_pred, eos_pred = self(X, target_seq, target_mask)

        # Compute loss
        c_true = target_seq[:, :, :2]  # Extract true coefficients
        t_true = target_seq[:, :, 2:3]  # Extract true knots

        loss_c = self.criterion_c(c_pred, c_true)
        loss_t = self.criterion_t(t_pred, t_true)
        loss_eos = self.criterion_eos(eos_pred.squeeze(-1), eos_labels)

        # Apply the mask to the losses
        loss_c = loss_c * target_mask.unsqueeze(-1)
        loss_t = loss_t * target_mask.unsqueeze(-1)
        loss_eos = loss_eos * target_mask

        loss = loss_c.sum() + loss_t.sum() + loss_eos.sum()

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def visualize_positional_encoding(d_model, max_len):
    pe = PositionalEncoding(d_model, max_len)
    positional_encodings = pe.pe.squeeze(1).detach().numpy()

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(positional_encodings, cmap="viridis")
    plt.title("Sinusoidal Positional Encodings")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.colorbar(label="Encoding Value")
    plt.show()


def main():
    NUM_SAMPLES = 64
    X_SHAPE = (3, 224, 224)

    dataset = DummyData(NUM_SAMPLES, X_SHAPE, MAX_LEN)
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)

    model = ImageToSequenceTransformer(max_seq_len=MAX_LEN).to("cuda")
    # Use reduction='none' to handle masking
    criterion_c = nn.MSELoss(reduction="none")
    criterion_t = nn.MSELoss(reduction="none")
    criterion_eos = nn.BCELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(100):  # Replace with more epochs as needed
        model.train()
        running_loss = 0.0
        for i, (X, target_seq, target_mask) in enumerate(dataloader):
            X, target_seq, target_mask = (
                X.to("cuda"),
                target_seq.to("cuda"),
                target_mask.to("cuda"),
            )

            # Create EOS labels: 1 for the last valid token, 0 otherwise
            eos_labels = torch.zeros_like(target_mask)
            eos_labels[
                torch.arange(target_mask.size(0)), (target_mask.sum(dim=1) - 1).long()
            ] = 1

            # Forward pass
            c_pred, t_pred, eos_pred = model(X, target_seq, target_mask)

            # Compute loss
            c_true = target_seq[:, :, :2]  # Extract true coefficients
            t_true = target_seq[:, :, 2:3]  # Extract true knots

            loss_c = criterion_c(c_pred, c_true)
            loss_t = criterion_t(t_pred, t_true)
            loss_eos = criterion_eos(eos_pred.squeeze(-1), eos_labels)

            # Apply the mask to the losses
            loss_c = loss_c * target_mask.unsqueeze(-1)
            loss_t = loss_t * target_mask.unsqueeze(-1)
            loss_eos = loss_eos * target_mask

            loss = loss_c.sum() + loss_t.sum() + loss_eos.sum()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/100], Loss: {running_loss/len(dataloader)}")

    print("Finished Training")


if __name__ == "__main__":
    main()
