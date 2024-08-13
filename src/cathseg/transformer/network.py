import math

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import ViT_B_16_Weights, vit_b_16

MAX_LEN = 20


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
    ):
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


class ViTEncoder(nn.Module):
    def __init__(self, n_channels, image_shape, output_dim=512, pretrained=True):
        super(ViTEncoder, self).__init__()

        # Load the pretrained ViT model
        if pretrained:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16(weights=None)

        # Freeze the parameters
        for param in self.vit.parameters():
            param.requires_grad = False

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

        # Project to the desired output dimension
        x = self.proj(x)

        return x


class ImageToSequenceTransformer(pl.LightningModule):
    def __init__(
        self,
        max_seq_len: int,
        img_size: tuple = (224, 224),
        d_model: int = 512,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(ImageToSequenceTransformer, self).__init__()

        self.encoder = ViTEncoder(3, img_size, d_model, pretrained=True)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

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

        # Output layers for coefficients `c` and knots `t` and end-of-sequence token `eos`
        self.fc_c = nn.Linear(d_model, 2)  # Predicting 2D control points
        self.fc_t = nn.Linear(d_model, 1)  # Predicting 1D knots
        self.fc_eos = nn.Linear(d_model, 1)  # Predicting end-of-sequence token

        # Embedding layer for target sequence
        self.target_embedding = nn.Linear(
            3, d_model
        )  # Concatenate c (2D) and t (1D) to form a 3D input for the next step

        # Loss functions
        self.criterion_c = nn.MSELoss(reduction="none")
        self.criterion_t = nn.MSELoss(reduction="none")
        self.criterion_eos = nn.BCELoss(reduction="none")

        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(max_seq_len)

        self.training_step_output = None

    def forward(self, x, target_seq, target_mask):
        # Feature extraction
        features = self.encoder(x)  # Output shape: (batch_size, d_model, H, W)

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

        tgt_key_padding_mask = target_mask == 0  # Shape: (batch_size, seq_len)

        # Needs to be moved here to match the device of the input tensors
        self.tgt_mask = self.tgt_mask.to(self.device)

        # Transformer Decoder with masked self-attention and key padding mask
        decoder_output = self.transformer_decoder(
            tgt=target_seq,
            memory=features,
            tgt_mask=self.tgt_mask,
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

    def _step(self, batch, batch_idx):
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

        self.training_step_output = dict(
            img=X[0],
            c_true=c_true[0],
            c_pred=c_pred[0],
            t_true=t_true[0],
            t_pred=t_pred[0],
            seq_len=target_mask.sum(dim=1)[0],
        )

        return loss_c.sum(), loss_t.sum(), loss_eos.sum(), loss

    def _log(self, loss_c, loss_t, loss_eos, loss, tag):
        self.log(f"{tag}/loss_c", loss_c.sum())
        self.log(f"{tag}/loss_t", loss_t.sum())
        self.log(f"{tag}/loss_eos", loss_eos.sum())
        self.log(f"{tag}/loss", loss)

    def training_step(self, batch, batch_idx):
        loss_c, loss_t, loss_eos, loss = self._step(batch, batch_idx)
        self._log(loss_c, loss_t, loss_eos, loss, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss_c, loss_t, loss_eos, loss = self._step(batch, batch_idx)
        self._log(loss_c, loss_t, loss_eos, loss, "val")
        return loss

    def configure_optimizers(self):
        optimizer = optim.NAdam(self.parameters(), lr=1e-4)
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


def unnormalize_img(img):
    import numpy as np

    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)[:, :, 0]
    return img


def plot_instance(instance):
    import numpy as np
    from scipy.interpolate import splev

    img, target_seq, target_mask = instance
    img, target_seq, target_mask = (
        img.cpu().detach().numpy(),
        target_seq.cpu().detach().numpy(),
        target_mask.cpu().detach().numpy(),
    )

    seq_len = target_mask.sum().astype(int)
    target_seq = target_seq[1:seq_len]

    t = target_seq[:, 0]
    t = np.concatenate([np.zeros((4,)), t])
    c = target_seq[:, 1:].T

    sample_idx = np.linspace(0, t[-1], 40)
    samples = splev(sample_idx, (t, c, 3))

    img = img.squeeze(0)
    # img = np.ones(img.shape).transpose(1, 2, 0)
    plt.imshow(img, cmap="gray")
    plt.scatter(c[0], c[1], c="r")
    plt.plot(samples[0], samples[1], c="b")
    plt.axis("off")
    plt.show()


def main():
    from pathlib import Path

    from guide3d.dataset.image.spline import Guide3D

    NUM_SAMPLES = 64
    X_SHAPE = (3, 1024, 1024)

    # dataset = DummyData(NUM_SAMPLES, X_SHAPE, MAX_LEN)
    dataset = Guide3D(root=Path.home() / "data/segment-real/")
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)

    model = ImageToSequenceTransformer(max_seq_len=MAX_LEN)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(100):  # Replace with more epochs as needed
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            img, target_seq, target_mask = batch
            plot_instance(tuple(x[0] for x in batch))
            continue
            exit()

            loss = model.training_step(batch, i)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/100], Loss: {running_loss/len(dataloader)}")

    print("Finished Training")


if __name__ == "__main__":
    main()
