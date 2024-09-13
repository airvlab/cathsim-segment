import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.transforms import transforms
from x_transformers import Decoder, Encoder, TransformerWrapper, ViTransformerWrapper

#
# encoder = ViTransformerWrapper(image_size=256, patch_size=32, attn_layers=Encoder(dim=512, depth=6, heads=8))
#
# decoder = TransformerWrapper(
#     num_tokens=20000, max_seq_len=1024, attn_layers=Decoder(dim=512, depth=6, heads=8, cross_attend=True)
# )
#
# img = torch.randn(1, 3, 256, 256)
# caption = torch.randint(0, 20000, (1, 1024))
#
# encoded = encoder(img, return_embeddings=True)
# decoder(caption, context=encoded)  # (1, 1024, 20000)
# print(decoder(caption, context=encoded).shape)

MAX_LEN = 20
N_CHANNELS = 3
IMG_SIZE = 1024

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


class ImageToSequenceTransformer(pl.LightningModule):
    def __init__(
        self,
        max_seq_len: int,
        n_channels: int = 3,
        img_size: int = 224,
        d_model: int = 512,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(ImageToSequenceTransformer, self).__init__()
        assert max_seq_len > 2, "Sequence length must be greater than 2"

        self.encoder = ViTransformerWrapper(
            image_size=img_size, patch_size=32, attn_layers=Encoder(dim=d_model, depth=6, heads=8)
        )
        self.decoder = TransformerWrapper(
            num_tokens=1000,
            return_only_embed=True,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(dim=d_model, depth=6, heads=8, cross_attend=True),
        )

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

    def forward_backup(self, img, target_seq, target_mask):
        features = self.encoder(img)  #  (batch_size, d_model)

        features = features.unsqueeze(0)  # (1, batch_size, d_model), to match transformer input format

        # target sequence embedding and positional encoding
        target_seq = self.target_embedding(target_seq)  # (batch_size, seq_len, d_model)
        target_seq = self.pos_encoder(target_seq.permute(1, 0, 2))  # (seq_len, batch_size, d_model)

        tgt_key_padding_mask = target_mask == 0  # shape: (batch_size, seq_len)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_seq.size(0)).to(
            target_seq.device
        )  # (seq_len, seq_len)

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
            c_pred.permute(1, 0, 2),  # (batch_size, seq_len, 2)
            t_pred.permute(1, 0, 2),  # (batch_size, seq_len, 1)
            eos_pred.permute(1, 0, 2).squeeze(-1),  # (batch_size, seq_len)
        )

    def forward(self, img, target_seq, target_mask):
        features = self.encoder(img)  #  (batch_size, d_model)

        tgt_key_padding_mask = target_mask == 0  # shape: (batch_size, seq_len)

        decoder_output = self.decoder(
            target_seq,
            mask=tgt_key_padding_mask,
            context=features,
        )
        print(decoder_output.shape)

        # output prediction
        c_pred = self.fc_c(decoder_output)  #  (seq_len, batch_size, 2)
        t_pred = self.fc_t(decoder_output)  #  (seq_len, batch_size, 1)
        eos_pred = self.fc_eos(decoder_output)  #  (seq_len, batch_size, 1)

        exit()

        return (
            c_pred.permute(1, 0, 2),  # (batch_size, seq_len, 2)
            t_pred.permute(1, 0, 2),  # (batch_size, seq_len, 1)
            eos_pred.permute(1, 0, 2).squeeze(-1),  # (batch_size, seq_len)
        )

    def _step(self, batch, batch_idx):
        X, target_seq, target_mask = batch

        # Create EOS labels: 1 for the last valid token, 0 otherwise
        eos_labels = torch.zeros_like(target_mask)
        eos_labels[torch.arange(target_mask.size(0)), (target_mask.sum(dim=1) - 1).long()] = 1
        eos_labels = eos_labels.float()

        # Forward pass
        c_pred, t_pred, eos_pred = self(X, target_seq, target_mask)

        # Compute loss
        t_true = target_seq[:, :, 0:1]  # Extract true knots (batch_size, seq_len, 1)
        c_true = target_seq[:, :, 1:3]  # Extract true coefficients (batch_size, seq_len, 2)

        loss_t = self.criterion_t(t_pred, t_true)  # (batch_size, seq_len, 1)
        loss_c = self.criterion_c(c_pred, c_true)  # (batch_size, seq_len, 2)
        loss_eos = self.criterion_eos(eos_pred, eos_labels)  # (batch_size, seq_len)

        # Apply the mask to the losses
        loss_t = loss_t * target_mask.unsqueeze(-1)
        loss_c = loss_c * target_mask.unsqueeze(-1)
        loss_eos = loss_eos * target_mask

        loss = self.lambda_t * loss_t.sum() + self.lambda_c * loss_c.sum() + self.lambda_eos * loss_eos.sum()

        if batch_idx == 0:
            self.training_step_output = [
                dict(
                    img=X[i],
                    t_true=t_true[i],
                    t_pred=t_pred[i],
                    c_true=c_true[i],
                    c_pred=c_pred[i],
                    seq_len=target_mask.sum(dim=1)[i],
                )
                for i in range(min(4, X.size(0)))
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
        return loss

    def validation_step(self, batch, batch_idx):
        loss_c, loss_t, loss_eos, loss = self._step(batch, batch_idx)
        self._log(loss_c, loss_t, loss_eos, loss, "val")
        return loss

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
                target_seq = self.pos_encoder(target_seq.permute(1, 0, 2))

                decoder_output = self.transformer_decoder(
                    tgt=target_seq,
                    memory=features.unsqueeze(0),  # (1, batch_size, d_model)
                )  # (seq_len, 1, d_model)

                # Generate predictions for the entire sequence
                c_pred = torch.sigmoid(self.fc_c(decoder_output))
                t_pred = torch.sigmoid(self.fc_t(decoder_output))
                eos_pred = torch.sigmoid(self.fc_eos(decoder_output))

                # Take the predictions corresponding to the last time step
                last_c_pred = c_pred[-1].unsqueeze(0)
                last_t_pred = t_pred[-1].unsqueeze(0)
                last_eos_pred = eos_pred[-1].unsqueeze(0)

                combined = torch.cat([last_t_pred, last_c_pred], dim=-1)  # (1, 1, 3)
                generated_seq = torch.cat([generated_seq, combined], dim=1)  # (1, seq_len, 3)

                if last_eos_pred.item() > 0.5 and i > 2:
                    break

            generated_seq = generated_seq.squeeze(0)[1:]  # Remove the initial token
            t_pred = generated_seq[:, 0]  # (seq_len,)
            c_pred = generated_seq[:, 1:]  # (seq_len, 2)

        self.train()

        return dict(t=t_pred, c=c_pred)

    def configure_optimizers(self):
        optimizer = optim.NAdam(self.parameters(), lr=1e-5)
        return optimizer


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

    # dataset = DummyData(NUM_SAMPLES, X_SHAPE, MAX_LEN)
    dataset = Guide3D(root=Path.home() / "data/segment-real/", image_transform=vit_transform)
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)

    model = ImageToSequenceTransformer(max_seq_len=dataset.max_length, n_channels=N_CHANNELS, img_size=IMG_SIZE)
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

        print(f"Epoch [{epoch+1}/100], Loss: {running_loss/len(dataloader)}")

    print("Finished Training")


if __name__ == "__main__":
    main()
