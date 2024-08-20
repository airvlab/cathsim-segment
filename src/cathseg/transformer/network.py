import math

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import transforms

MAX_LEN = 20
N_CHANNELS = 1
IMAGE_SIZE = 1024

vit_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert image to PIL image
        # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize image to 224x224
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(N_CHANNELS, 1, 1)),
        transforms.Normalize(  # Normalize with mean and std
            mean=[0.5 for _ in range(N_CHANNELS)],
            std=[0.5 for _ in range(N_CHANNELS)],
        ),
    ]
)


def c_transform(c):
    max_val = IMAGE_SIZE
    min_val = -0.6102361976879171
    return (c - min_val) / (max_val - min_val)


def t_transform(t):
    return t / 1500


def c_untransform(c):
    max_val = IMAGE_SIZE
    min_val = -0.6102361976879171
    return c * (max_val + min_val) + min_val


def t_untransform(t):
    return t * 1500


def unnorm(img):
    img = img * 0.5 + 0.5
    return img


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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PositionalEncodingBFirst(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe[:, : x.size(1), :]  # Adjust indexing for batch-first input
        return self.dropout(x)


class ViTEncoder(nn.Module):
    def __init__(self, n_channels, image_shape, output_dim=512, pretrained=True):
        super(ViTEncoder, self).__init__()

        # Add avg pooling as first layer to ensure arbitrary sizes are mapped to ViT input size.
        self.initial_pooling = nn.AdaptiveAvgPool2d((224, 224))

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
            self.vit.conv_proj = nn.Conv2d(n_channels, self.vit.conv_proj.out_channels, kernel_size=16, stride=16)

        # Remove the classification head
        self.vit.heads = nn.Identity()

        # Add a projection layer to match the desired output dimension if specified
        self.output_dim = output_dim
        if output_dim != self.vit.hidden_dim:
            self.proj = nn.Linear(self.vit.hidden_dim, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        # Pass into pooling.
        x = self.initial_pooling(x)
        # Forward pass through the ViT model
        x = self.vit(x)

        # Project to the desired output dimension
        x = self.proj(x)

        return x


class ImageToSequenceTransformer(pl.LightningModule):
    def __init__(
        self,
        max_seq_len: int,
        n_channels: int = 3,
        img_size: tuple = (224, 224),
        d_model: int = 512,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(ImageToSequenceTransformer, self).__init__()
        self.save_hyperparameters()

        self.encoder = ViTEncoder(n_channels, img_size, d_model, pretrained=True)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Decoder-only Transformer with masked self-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output layers for coefficients `c` and knots `t` and end-of-sequence token `eos`
        self.fc_c = nn.Sequential(nn.Linear(d_model, 2), nn.Sigmoid())  # Predicting 2D control
        self.fc_t = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())  # Predicting 1D knots
        self.fc_eos = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())  # Predicting end-of-sequence token

        # Embedding layer for target sequence
        self.target_embedding = nn.Linear(
            3, d_model
        )  # Concatenate c (2D) and t (1D) to form a 3D input for the next step

        # Loss functions
        self.criterion_c = nn.MSELoss(reduction="none")
        self.criterion_t = nn.MSELoss(reduction="none")
        self.criterion_eos = nn.BCELoss(reduction="none")

        self.lambda_c = 10.0
        self.lambda_t = 10.0
        self.lambda_eos = 1.0

        self.training_step_output = None
        self.max_seq_len = max_seq_len

        self.init_token = torch.zeros(1, 3)  # (seq_len, dim)

    def forward(self, x, target_seq, target_mask):
        features = self.encoder(x)  #  (batch_size, d_model)

        features = features.unsqueeze(0)  # (1, batch_size, d_model), to match transformer input format

        # target sequence embedding and positional encoding
        target_seq = self.target_embedding(target_seq)  # (batch_size, seq_len, d_model)
        target_seq = self.pos_encoder(target_seq.permute(1, 0, 2))  # (seq_len, batch_size, d_model)

        tgt_key_padding_mask = target_mask.to(dtype=torch.float)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_seq.size(0)).to(device=target_seq.device)

        decoder_output = self.transformer_decoder(
            tgt=target_seq,
            memory=features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # output prediction
        c_pred = self.fc_c(decoder_output)  #  (seq_len, batch_size, 2)
        t_pred = self.fc_t(decoder_output)  #  (seq_len, batch_size, 1)
        eos_pred = self.fc_eos(decoder_output)  #  (seq_len, batch_size, 1)

        return (
            t_pred.permute(1, 0, 2),  # (batch_size, seq_len, 1)
            c_pred.permute(1, 0, 2),  # (batch_size, seq_len, 2)
            eos_pred.permute(1, 0, 2).squeeze(-1),  # (batch_size, seq_len)
        )

    def _step(self, batch, batch_idx):
        imgs, target_seq, target_mask = batch

        ## PREPARE TARGET DATA
        init_token = self.init_token.expand(target_seq.size(0), -1, -1).to(device=target_seq.device)
        target_seq = torch.cat([init_token, target_seq], 1)

        input_seq = target_seq[:, :-1, :]
        target_seq = target_seq[:, 1:, :]

        # Create EOS labels: 1 for the last valid token, 0 otherwise
        eos_labels = torch.zeros_like(target_mask)
        eos_labels[torch.arange(target_mask.size(0)), (target_mask.sum(dim=1) - 1).long()] = 1
        eos_labels = eos_labels.float()

        # Forward pass
        t_pred, c_pred, eos_pred = self(imgs, input_seq, target_mask)

        t_true = target_seq[:, :, 0:1]
        c_true = target_seq[:, :, 1:3]

        # Compute losses
        loss_t = self.criterion_t(t_pred, t_true)  # (batch_size, seq_len, 1)
        loss_c = self.criterion_c(c_pred, c_true)  # (batch_size, seq_len, 2)
        loss_eos = self.criterion_eos(eos_pred, eos_labels)  # (batch_size, seq_len)

        # Apply the mask to the losses
        loss_t = loss_t * target_mask.unsqueeze(-1)
        loss_c = loss_c * target_mask.unsqueeze(-1)
        loss_eos = loss_eos * target_mask

        # Compute the total loss as a weighted sum
        loss = self.lambda_t * loss_t.sum() + self.lambda_c * loss_c.sum() + self.lambda_eos * loss_eos.sum()

        if batch_idx == 0:
            self.training_step_output = [
                dict(
                    img=imgs[i],
                    t_true=t_true[i],
                    t_pred=t_pred[i],
                    c_true=c_true[i],
                    c_pred=c_pred[i],
                    seq_len=target_mask.sum(dim=1)[i],
                )
                for i in range(min(4, imgs.size(0)))
            ]

        return loss_c.sum(), loss_t.sum(), loss_eos.sum(), loss

    def _log(self, loss_c, loss_t, loss_eos, loss, tag):
        self.log(f"{tag}/loss_c", loss_c.sum())
        self.log(f"{tag}/loss_t", loss_t.sum())
        self.log(f"{tag}/loss_eos", loss_eos.sum())
        self.log(f"{tag}/loss", loss)

    def training_step(self, batch, batch_idx):
        loss_c, loss_t, loss_eos, loss = self._step(batch, batch_idx)
        self._log(loss_c, loss_t, loss_eos, loss, "train")

        # Gradient clipping to stablise training as we are using transformer.
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return loss

    def validation_step(self, batch, batch_idx):
        loss_c, loss_t, loss_eos, loss = self._step(batch, batch_idx)
        self._log(loss_c, loss_t, loss_eos, loss, "val")
        return loss

    def predict_step(self, batch):
        pass

    def test_step(self, batch, batch_idx):
        generated = self.inference_step(batch[0], batch_idx)
        loss = 0
        return loss

    def inference_step(self, X):
        self.eval()

        with torch.no_grad():
            X = X.unsqueeze(0)  # Add batch dimension
            features = self.encoder(X)  # (batch_size, d_model)

            generated_seq = torch.zeros(1, 1, 3).to(X.device)  # Start token or initial sequence

            for i in range(self.max_seq_len):
                # target sequence embedding and positional encoding
                target_seq = self.target_embedding(generated_seq)
                target_seq = self.pos_encoder(target_seq.permute(1, 0, 2))  # (seq_len, 1, d_model)

                decoder_output = self.transformer_decoder(
                    tgt=target_seq,
                    memory=features.unsqueeze(0),  # (1, batch_size, d_model)
                )  # (seq_len, 1, d_model)

                # Generate predictions for the entire sequence
                c_pred = self.fc_c(decoder_output)
                t_pred = self.fc_t(decoder_output)
                eos_pred = self.fc_eos(decoder_output)

                # Take the predictions corresponding to the last time step
                last_c_pred = c_pred[-1].unsqueeze(0)
                last_t_pred = t_pred[-1].unsqueeze(0)
                last_eos_pred = eos_pred[-1].unsqueeze(0)

                combined = torch.cat([last_t_pred, last_c_pred], dim=-1)  # (1, 1, 3)
                generated_seq = torch.cat([generated_seq, combined], dim=1)  # (1, seq_len, 3)

                if last_eos_pred.item() > 0.5 and i > 2:
                    break

            generated_seq = generated_seq[0, 1:, :]  # Remove the initial token
            t_pred = generated_seq[:, 0:1]  # (seq_len,)
            c_pred = generated_seq[:, 1:3]  # (seq_len, 2)

        self.train()

        return dict(t=t_pred, c=c_pred)

    def configure_optimizers(self):
        optimizer = optim.NAdam(self.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 70, eta_min = 1e-7)

        return [optimizer], [scheduler]


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
    target_seq = target_seq[:seq_len]

    t = target_seq[:, 0]
    t = np.concatenate([np.zeros((4,)), t])
    t = t_untransform(t)
    c = target_seq[:, 1:].T
    c = c_untransform(c)

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

    # dataset = DummyData(NUM_SAMPLES, X_SHAPE, MAX_LEN)
    dataset = Guide3D(
        root=Path.home() / "data/segment-real/",
        image_transform=vit_transform,
        c_transform=c_transform,
        t_transform=t_transform,
    )

    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=False)

    model = ImageToSequenceTransformer(max_seq_len=dataset.max_length, n_channels=N_CHANNELS, img_size=IMAGE_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(100):  # Replace with more epochs as needed
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            img, target_seq, target_mask = batch
            # plot_instance(tuple(x[0] for x in batch))

            # model.inference_step(img[0])
            # exit()

            loss = model.training_step(batch, i)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # if i == 3:
            #     exit()
            # exit()

        print(f"Epoch [{epoch+1}/100], Loss: {running_loss/len(dataloader)}")

    print("Finished Training")


if __name__ == "__main__":
    main()
