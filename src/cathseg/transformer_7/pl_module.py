import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from cathseg.transformer_7.model import SplineTransformer as Model
from torchvision.transforms import transforms

MAX_LEN = 20
N_CHANNELS = 1
IMAGE_SIZE = 224

vit_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert image to PIL image
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize image to 224x224
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(N_CHANNELS, 1, 1)),
        transforms.Normalize(  # Normalize with mean and std
            mean=[0.5 for _ in range(N_CHANNELS)],
            std=[0.5 for _ in range(N_CHANNELS)],
        ),
    ]
)


def c_transform(c, c_min=0, c_max=1):
    c_max = IMAGE_SIZE
    return (c - c_min) / (c_max - c_min)


def t_transform(t):
    return t / 1500


def c_untransform(c, c_min=0, c_max=1):
    c_max = IMAGE_SIZE
    return c * (c_max + c_min) + c_min


def t_untransform(t):
    return t * 1500


def unnorm(img):
    img = img * 0.5 + 0.5
    return img


class SplineFormer(pl.LightningModule):
    def __init__(
        self,
        tgt_max_len: int,
        num_channels: int = 1,
        img_size: tuple = 1024,
        d_model: int = 512,
        patch_size: int = 32,
        num_heads: int = 8,
        num_layers_encoder: int = 6,
        num_layers_decoder: int = 6,
        dropout: float = 0.1,
        n_dim: int = 2,
    ):
        super(SplineFormer, self).__init__()
        self.save_hyperparameters()

        self.n_dim = n_dim
        self.tgt_max_len = tgt_max_len

        self.model = Model(
            image_size=img_size,
            num_channels=num_channels,
            patch_size=patch_size,
            d_model=d_model,
            num_layers_encoder=num_layers_encoder,
            num_layers_decoder=num_layers_decoder,
            dropout=dropout,
            num_heads=num_heads,
            tgt_max_len=tgt_max_len,
        )

        # Loss functions
        self.criterion_seq = nn.MSELoss(reduction="none")
        self.criterion_eos = nn.BCELoss(reduction="none")

        self.lambda_seq = 1.0
        self.lambda_eos = 1.0

        self.init_token = torch.zeros(1, 1 + self.n_dim)  # (seq_len, dim)

        self.training_step_output = []

    def forward(self, img, tgt, tgt_mask, tgt_pad_mask):
        seq_pred, eos_pred, memory, encoder_atts, decoder_atts = self.model(
            src=img, tgt=tgt, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask
        )
        return seq_pred, eos_pred.squeeze(-1), memory, encoder_atts, decoder_atts

    def _step(self, batch, batch_idx, val=False):
        imgs, tgt, tgt_pad_mask = batch

        init_token = self.init_token.expand(tgt.size(0), -1, -1).to(device=tgt.device)
        tgt = torch.cat([init_token, tgt], 1)

        tgt_input = tgt[:, :-1, :]
        tgt_output = tgt[:, 1:, :]

        # Create EOS labels: 1 for the last valid token, 0 otherwise
        eos_labels = torch.zeros_like(tgt_pad_mask)
        eos_labels[torch.arange(tgt_pad_mask.size(0)), (tgt_pad_mask.sum(dim=1) - 1).long()] = 1
        eos_labels = eos_labels.float()

        tgt_pad_mask = tgt_pad_mask.to(dtype=torch.float)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device=tgt_input.device)

        # Forward pass
        pred_seq, eos_pred, memory, encoder_atts, decoder_atts = self(imgs, tgt_input, tgt_mask, tgt_pad_mask)

        # Compute losses
        loss_seq = self.criterion_seq(pred_seq, tgt_output)
        loss_eos = self.criterion_eos(eos_pred, eos_labels)  # (batch_size, seq_len)

        # Apply the mask to the losses
        loss_seq = loss_seq * tgt_pad_mask.unsqueeze(-1)
        loss_eos = loss_eos * tgt_pad_mask

        # Compute the total loss as a weighted sum
        loss = self.lambda_seq * loss_seq.sum() + self.lambda_eos * loss_eos.sum()

        seq_lens = tgt_pad_mask.sum(dim=1)

        if val:
            if len(self.training_step_output) < 10:
                self.training_step_output.append(
                    dict(
                        img=imgs[0],
                        t_true=tgt_output[0, :, 0:1],
                        c_true=tgt_output[0, :, 1:3],
                        t_pred=pred_seq[0, :, 0:1],
                        c_pred=pred_seq[0, :, 1:3],
                        seq_len=seq_lens[0],
                    )
                )

        return loss_seq.sum(), loss_eos.sum(), loss

    def _log(self, loss_seq, loss_eos, loss, tag):
        self.log(f"{tag}/loss_seq", loss_seq.sum())
        self.log(f"{tag}/loss_eos", loss_eos.sum())
        self.log(f"{tag}/loss", loss)

    def training_step(self, batch, batch_idx):
        loss_seq, loss_eos, loss = self._step(batch, batch_idx)
        self._log(loss_seq, loss_eos, loss, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss_seq, loss_eos, loss = self._step(batch, batch_idx, val=True)
        self._log(loss_seq, loss_eos, loss, "val")
        return loss

    def inference_step(self, X):
        self.eval()

        with torch.no_grad():
            # Add batch dimension to the input tensor
            X = X.unsqueeze(0)  # (1, input_dim)

            # Initialize the generated sequence with start token (all zeros)
            generated_seq = torch.zeros(1, 1, 3).to(X.device)  # (1, 1, 3)

            for i in range(self.tgt_max_len):
                seq_pred, eos_pred, memory, encoder_atts, decoder_atts = self.model(src=X, tgt=generated_seq)

                # Take the last prediction for each component
                last_pt_pred = seq_pred[:, -1:, :]
                last_eos_pred = eos_pred[:, -1:, :]  # (1, 1)

                generated_seq = torch.cat([generated_seq, last_pt_pred], dim=1)  # (1, seq_len+1, 3)

                # Early stopping condition based on eos prediction
                if last_eos_pred.item() > 0.5 and i > 2:
                    break

            # Remove the initial start token and prepare the output
            generated_seq = generated_seq[:, 1:, :].squeeze(0)  # (seq_len, 3)
            t_pred = generated_seq[:, 0]  # (seq_len)
            c_pred = generated_seq[:, 1:3]  # (seq_len, 2)

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


def plot_instance(instance, dataset):
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
    t = t_untransform(t, None, None)
    c = target_seq[:, 1:].T
    c = c_untransform(c, dataset.c_min, dataset.c_max)

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
    from guide3d.dataset.image.spline import Guide3D

    # dataset = DummyData(NUM_SAMPLES, X_SHAPE, MAX_LEN)
    dataset = Guide3D(
        dataset_path="/tmp/guide3d/guide3d/",
        download=True,
        image_transform=vit_transform,
        c_transform=c_transform,
        t_transform=t_transform,
    )

    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=False)

    model = SplineFormer(tgt_max_len=dataset.max_length, num_channels=N_CHANNELS, img_size=IMAGE_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(100):  # Replace with more epochs as needed
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            img, target_seq, target_mask = batch
            # plot_instance(tuple(x[0] for x in batch), dataset)

            # model.inference_step(img[0])
            # exit()

            loss = model.training_step(batch, i)
            exit()

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
