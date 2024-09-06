import cathseg.splineformer.modules as modules
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cathseg.splineformer_w_tip_pred.model import SplineTransformer as Model
from torch import Tensor

BATCH_SIZE = 16
IMAGE_SIZE = 1024
NUM_CHANNELS = 1
PATCH_SIZE = 16
D_MODEL = 512


def img_untransform(img):
    img = img * 0.5 + 0.5
    return img


def c_untransform(c):
    return c * IMAGE_SIZE


def t_untransform(t):
    return t * IMAGE_SIZE


def c_transform(c):
    return c / 1024


def t_transform(t):
    return t / 1024


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
        self.img_size = img_size

        self.tip_predictor = modules.TipPredictor(num_channels=num_channels)

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
        self.criterion_tip = nn.MSELoss(reduction="none")
        self.criterion_seq = nn.MSELoss(reduction="none")
        self.criterion_eos = nn.BCELoss(reduction="none")

        self.lambda_tip = 10
        self.lambda_seq = 10
        self.lambda_eos = 1.0

        self.init_token = torch.zeros(1, 1 + self.n_dim)

        self.val_step_outputs = []

    def forward(self, img, tgt, tgt_mask, tgt_pad_mask):
        seq_pred, eos_pred, encoder_atts, decoder_atts = self.model(
            src=img, tgt=tgt, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask
        )
        return seq_pred, eos_pred.squeeze(-1), encoder_atts, decoder_atts

    def _compute_loss(self, pred_seq, eos_pred, tgt_output, eos_labels, tgt_pad_mask, tip_pred, tip_true):
        loss_tip = self.criterion_tip(tip_pred, tip_true)
        loss_seq = self.criterion_seq(pred_seq, tgt_output)
        loss_eos = self.criterion_eos(eos_pred, eos_labels)

        loss_seq = (loss_seq * tgt_pad_mask.unsqueeze(-1)).sum()
        loss_eos = (loss_eos * tgt_pad_mask).sum()
        loss_tip = loss_tip.sum()

        loss = self.lambda_seq * loss_seq + self.lambda_eos * loss_eos + self.lambda_tip * loss_tip
        return loss_seq, loss_eos, loss_tip, loss

    def _step(self, batch, batch_idx, val=False):
        imgs, tgt, tgt_pad_mask = batch

        seq_lens = tgt_pad_mask.sum(dim=1)
        tgt_pad_mask = tgt_pad_mask[:, 1:]

        tip_pred = self.tip_predictor(imgs)
        tip_true = tgt[:, 0, 1:3]

        tgt_input = tgt[:, :-1, :]
        tgt_output = tgt[:, 1:, :]

        eos_labels = torch.zeros_like(tgt_pad_mask)
        eos_labels[torch.arange(tgt_pad_mask.size(0)), (tgt_pad_mask.sum(dim=1) - 1).long()] = 1
        eos_labels = eos_labels.float()

        tgt_pad_mask = tgt_pad_mask.to(dtype=torch.float)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device=tgt_input.device)

        pred_seq, eos_pred, _, _ = self(imgs, tgt_input, tgt_mask, tgt_pad_mask)

        loss_seq, loss_eos, loss_tip, loss = self._compute_loss(
            pred_seq, eos_pred, tgt_output, eos_labels, tgt_pad_mask, tip_pred, tip_true
        )

        if val:
            if len(self.val_step_outputs) < 10:
                generated_seq, encoder_atts, decoder_atts = self.generate_sequence(imgs[0].unsqueeze(0))
                t_gen = generated_seq[0, :, 0:1]
                c_gen = generated_seq[0, :, 1:3]

                self.val_step_outputs.append(
                    dict(
                        img=imgs[0],
                        t_true=tgt[0, :, 0:1],
                        c_true=tgt[0, :, 1:3],
                        t_pred=pred_seq[0, :, 0:1],
                        c_pred=pred_seq[0, :, 1:3],
                        t_gen=t_gen,
                        c_gen=c_gen,
                        seq_len=seq_lens[0],
                    )
                )

        return loss_seq, loss_eos, loss_tip, loss

    def _log(self, loss_seq, loss_eos, loss_tip, loss, tag):
        self.log(f"{tag}/loss_seq", loss_seq * self.img_size)
        self.log(f"{tag}/loss_eos", loss_eos * self.img_size)
        self.log(f"{tag}/loss_tip", loss_tip * self.img_size)
        self.log(f"{tag}/loss", loss)

    def training_step(self, batch, batch_idx):
        loss_seq, loss_eos, loss_tip, loss = self._step(batch, batch_idx)
        self._log(loss_seq, loss_eos, loss_tip, loss, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss_seq, loss_eos, loss_tip, loss = self._step(batch, batch_idx, val=True)
        self._log(loss_seq, loss_eos, loss_tip, loss, "val")
        return loss

    def test_step(self, batch, batch_idx):
        imgs, tgt, tgt_pad_mask = batch
        batch_size, seq_len, _ = tgt.shape

        loss = 0
        for img in imgs:
            pred_seq, encoder_atts, decoder_atts = self.generate_sequence(img.unsqueeze(0))
            pred_size = pred_seq.size(1)
            pred_seq = F.pad(pred_seq, (0, 0, 0, seq_len - pred_size))
            loss += self.lambda_seq * (self.criterion_seq(pred_seq, tgt) * tgt_pad_mask.unsqueeze(-1)).sum()
        self.log("loss", loss)

        return loss

    def generate_sequence(self, src: Tensor, threshold: float = 0.5, output_attentions: bool = False):
        batch_size = src.size(0)
        assert batch_size == 1, "Only batch size 1 is supported"

        self.eval()

        with torch.no_grad():
            tip_pred = self.tip_predictor(src)
            generated_seq = torch.cat([torch.zeros(batch_size, 1, device=src.device), tip_pred], dim=-1).unsqueeze(0)

            for i in range(self.tgt_max_len):
                seq_pred, eos_pred, encoder_atts, decoder_atts = self.model(
                    src=src, tgt=generated_seq, output_attentions=output_attentions
                )
                generated_seq = torch.cat([generated_seq, seq_pred[:, -1:, :]], dim=1)
                if eos_pred[:, -1] > threshold and i > 2:
                    break
            generated_seq = generated_seq[:, 1:, :]

        return generated_seq, encoder_atts, decoder_atts

    def predict_step(self, batch, batch_idx):
        X, _, _ = batch
        return X, *self.generate_sequence(X, output_attentions=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer


def unnormalize_img(img):
    import numpy as np

    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)[:, :, 0]
    return img


def compute_metric(pred_seq, tgt, seq_len, step_size, metric: callable):
    import numppy as np
    from scipy.interpolate import splev

    seq_len = seq_len.detach().cpu().numpy().astype(int)

    t_true = tgt[:, :, 0:1]
    t_pred = pred_seq[:, :, 0:1]

    c_true = tgt[:, :, 1:3]
    c_pred = pred_seq[:, :, 1:3]

    c_pred = c_untransform(c_pred.detach().cpu().numpy())
    c_true = c_untransform(c_true.detach().cpu().numpy())[:seq_len]

    t_pred = t_untransform(t_pred.detach().cpu().numpy()[:seq_len]).flatten()
    t_true = t_untransform(t_true.detach().cpu().numpy()[:seq_len]).flatten()[:seq_len]

    t_pred = np.concatenate([np.zeros((4)), t_pred], axis=0)
    t_true = np.concatenate([np.zeros((4)), t_true], axis=0)

    last_t = min(t_pred[-1], t_true[-1])
    num_steps = int(last_t / step_size) + 1

    sample_idx = np.linspace(0, last_t, num_steps)

    sampled_true = splev(sample_idx, (t_true, c_true.T, 3))
    sampled_pred = splev(sample_idx, (t_true, c_true.T, 3))

    return metric(sampled_true, sampled_pred)


def main():
    from cathseg.dataset import Guide3DModule

    dm = Guide3DModule(batch_size=1, n_channels=NUM_CHANNELS, image_size=IMAGE_SIZE)
    dm.setup("fit")
    dl = dm.val_dataloader()
    print("Val batches:", len(dl))

    model = SplineFormer(
        tgt_max_len=Guide3DModule.max_seq_len,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
        img_size=IMAGE_SIZE,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(100):  # Replace with more epochs as needed
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dl):
            img, target_seq, target_mask = batch
            # loss = model.training_step(batch, i)
            # print(loss)
            # plot_instance(tuple(x[0] for x in batch), dataset)

            # loss = model.training_step(batch, i)
            # model.test_step(batch, i)
            model.generate_sequence(img)

            exit()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # if i == 3:
            #     exit()
            # exit()

        print(f"Epoch [{epoch+1}/100], Loss: {running_loss/len(dl)}")

    print("Finished Training")


if __name__ == "__main__":
    main()
