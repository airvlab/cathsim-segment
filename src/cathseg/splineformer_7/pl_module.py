import cathseg.utils as utils
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from cathseg.metrics import MyLossFn, compute_all_metrics
from cathseg.splineformer_7.model import SplineTransformer as Model
from torch import Tensor

BATCH_SIZE = 32
IMAGE_SIZE = 1024
NUM_CHANNELS = 1
PATCH_SIZE = 32
D_MODEL = 512

INSTANCE = 0


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
        num_layers_encoder: int = 10,
        num_layers_decoder: int = 10,
        dropout: float = 0.1,
        n_dim: int = 2,
    ):
        super(SplineFormer, self).__init__()
        self.save_hyperparameters()

        self.n_dim = n_dim
        self.tgt_max_len = tgt_max_len
        self.img_size = img_size

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

        self.criterion_knot = nn.MSELoss(reduction="none")
        self.criterion_coeff = nn.MSELoss(reduction="none")
        self.criterion_eos = nn.BCELoss(reduction="none")

        self.lambda_knot = 100.0
        self.lambda_coeff = 100.0
        self.lambda_eos = 1.0

        self.test_metrics = MyLossFn()

        self.val_step_outputs = []

    def forward(self, img, tgt, tgt_mask, tgt_pad_mask):
        seq_pred, eos_pred, encoder_atts, decoder_atts, memory = self.model(
            src=img, tgt=tgt, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask
        )
        return seq_pred, eos_pred.squeeze(-1), encoder_atts, decoder_atts, memory

    def _compute_loss(self, pred_seq, eos_pred, tgt_output, eos_labels, tgt_pad_mask):
        loss_knot = self.criterion_knot(pred_seq[:, :, 0:1], tgt_output[:, :, 0:1])
        loss_coeff = self.criterion_coeff(pred_seq[:, :, 1:3], tgt_output[:, :, 1:3])
        loss_eos = self.criterion_eos(eos_pred, eos_labels)

        loss_knot = (loss_knot * tgt_pad_mask.unsqueeze(-1)).sum() / tgt_pad_mask.sum()
        loss_coeff = (loss_coeff * tgt_pad_mask.unsqueeze(-1)).sum() / tgt_pad_mask.sum()
        loss_eos = (loss_eos * tgt_pad_mask[:, 1:]).sum() / tgt_pad_mask.sum()

        loss = self.lambda_knot * loss_knot + self.lambda_coeff * loss_coeff + self.lambda_eos * loss_eos

        return dict(
            knot=self.lambda_knot * loss_knot,
            coeff=self.lambda_coeff * loss_coeff,
            eos=loss_eos,
            total_loss=loss,
        )

    def _step(self, batch, batch_idx, val=False):
        imgs, tgt, tgt_pad_mask = batch

        seq_lens = tgt_pad_mask.sum(dim=1)

        tgt_input = tgt[:, :-1, :]

        eos_labels = torch.zeros_like(tgt_pad_mask)
        eos_labels[torch.arange(tgt_pad_mask.size(0)), (tgt_pad_mask.sum(dim=1) - 1).long()] = 1
        eos_labels = eos_labels.float()

        tgt_pad_mask = tgt_pad_mask.to(dtype=torch.float)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device=tgt_input.device)

        pred_seq, eos_pred, _, _, _ = self(imgs, tgt_input, tgt_mask, tgt_pad_mask[:, 1:])

        tip_pred = self.tip_predictor(imgs).unsqueeze(1)
        pred_seq = torch.cat([tip_pred, pred_seq], dim=1)

        losses = self._compute_loss(pred_seq, eos_pred, tgt, eos_labels[:, 1:], tgt_pad_mask)

        if val:
            if len(self.val_step_outputs) < 10:
                generated_seq, encoder_atts, decoder_atts, memory = self.generate_sequence(imgs[0].unsqueeze(0))
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

        return losses

    def _log(self, losses, tag):
        for k, v in losses.items():
            self.log(f"{tag}/{k}", v)

    def training_step(self, batch, batch_idx):
        losses = self._step(batch, batch_idx)
        self._log(losses, "train")
        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        losses = self._step(batch, batch_idx, val=True)
        self._log(losses, "val")
        return losses["total_loss"]

    def test_step(self, batch, batch_idx):
        imgs, tgt, tgt_pad_mask = batch
        batch_size, seq_len, _ = tgt.shape

        loss = 0
        generated_seq = torch.zeros_like(tgt)
        seq_lens = tgt_pad_mask.sum(1)
        pred_seq_lens = torch.zeros_like(seq_lens)
        for i, img in enumerate(imgs):
            pred_seq, encoder_atts, decoder_atts, meemory = self.generate_sequence(img.unsqueeze(0))
            pred_len = pred_seq.size(1)
            generated_seq[i, :pred_len, :] = pred_seq[0, :, :]
            pred_seq_lens[i] = pred_len

        generated_seq = torch.clip(generated_seq, 0, 1)
        print("target", tgt)
        print("pred:", generated_seq)
        exit()
        metrics = self.test_metrics(generated_seq, tgt, tgt_pad_mask)

        # plot_prediction(imgs, generated_seq, pred_seq_lens)

        losses = compute_metrics(generated_seq, pred_seq_lens, tgt, seq_lens)

        for k, v in losses.items():
            self.log(k, v)

        for k, v in metrics.items():
            self.log(k, v)

        self.log("loss", loss)

        return loss

    def generate_sequence(
        self, src: Tensor, threshold: float = 0.5, output_attentions: bool = False, output_memory: bool = False
    ):
        batch_size = src.size(0)
        assert batch_size == 1, "Only batch size 1 is supported"

        self.eval()

        with torch.no_grad():
            generated_seq = self.tip_predictor(src).unsqueeze(1)

            for i in range(self.tgt_max_len - 1):
                seq_pred, eos_pred, encoder_atts, decoder_atts, memory = self.model(
                    src=src, tgt=generated_seq, output_attentions=output_attentions, output_memory=output_memory
                )
                generated_seq = torch.cat([generated_seq, seq_pred[:, -1:, :]], dim=1)
                if eos_pred[:, -1] > threshold and i > 2:
                    break

        return generated_seq, encoder_atts, decoder_atts, memory

    def predict_step(self, batch, batch_idx):
        X, tgt, _ = batch
        self.eval()
        y_hat = self.generate_sequence(X, output_attentions=True)
        generated_seq, encoder_atts, decoder_atts, memory = y_hat
        img = X[0].cpu()
        generated_seq = generated_seq.cpu()
        encoder_atts = encoder_atts.cpu()
        decoder_atts = decoder_atts.cpu()

        decoder_atts = utils.process_attention_maps(
            decoder_atts,
            img_size=IMAGE_SIZE,
            channels=NUM_CHANNELS,
            patch_size=PATCH_SIZE,
            layer=-1,
            aggreg_func=lambda x: torch.max(x, dim=2)[0],
            discard_ratio=0.8,
        )
        utils.plot_attention_maps(generated_seq[0, 1:], decoder_atts[0], img[0])
        return X, *y_hat

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def plot_prediction(imgs, pred_seqs, pred_seq_lens):
    global INSTANCE

    import math

    import matplotlib.pyplot as plt
    from cathseg.dataset import sample_spline

    batch_size, seq_len, _ = pred_seqs.shape
    n_cols = 1
    n_rows = math.ceil(batch_size / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    if isinstance(axs, np.ndarray):
        axs = axs.ravel()
    else:
        axs = [axs]

    for idx in range(batch_size):
        img, seq, seq_len = imgs[idx], pred_seqs[idx], pred_seq_lens[idx]

        seq = (seq * 1024).to(int)

        t = seq[:seq_len, 0]
        t = torch.cat([torch.zeros(4).to(device=t.device), t])
        c = seq[:seq_len, 1:3]
        img = img[0].cpu().numpy()

        t = t.cpu().numpy()
        c = c.cpu().numpy()

        pts = sample_spline((t, c.T, 3), delta=10)

        axs[idx].imshow(img, cmap="gray")
        axs[idx].plot(c[:, 0], c[:, 1], "ro", markersize=1)
        axs[idx].plot(pts[:, 0], pts[:, 1], "b", linewidth=0.5)

    # Turn off axes for all subplots
    for ax in axs:
        ax.axis("off")

    # Remove any unused subplots
    for idx in range(batch_size, len(axs)):
        axs[idx].remove()

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(f"samples/{INSTANCE}.png")
    INSTANCE += 1
    plt.tight_layout()
    # plt.show()
    plt.close()


def plot_masks(mask_pred, mask_true):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(mask_pred.shape[0], 2)
    if mask_pred.shape[0] == 1:
        axes = axes[None, :]
    for i in range(mask_pred.shape[0]):
        axes[i, 0].imshow(mask_pred[i].squeeze(), cmap="gray")
        axes[i, 1].imshow(mask_true[i].squeeze(), cmap="gray")
    for ax in axes.ravel():
        ax.axis("off")
    plt.show()


def compute_metrics(pred, pred_seq_lens, tgt, seq_len):
    curves_params_true = get_params(tgt, seq_len)
    pts_true = sample_curves(curves_params_true, 20)
    mask_true = get_masks(pts_true, 1024)
    mask_true = torch.tensor(mask_true).unsqueeze(1).to(torch.float)

    curves_params_pred = get_params(pred, pred_seq_lens)
    pts_pred = sample_curves(curves_params_pred, 20)
    mask_pred = get_masks(pts_pred, 1024)
    mask_pred = torch.tensor(mask_pred).unsqueeze(1).to(torch.float)

    losses = compute_all_metrics(mask_true, mask_pred)
    return losses


def get_masks(batch_pts, img_size):
    import cv2

    masks = np.zeros((len(batch_pts), img_size, img_size), dtype=np.uint8)
    for i in range(len(batch_pts)):
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        pts = np.array(batch_pts[i]).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=2)
        masks[i] = mask

    return masks / 255


def get_params(params, seq_len):
    import numpy as np

    params = params.cpu().numpy()
    params = params * 1024

    ts = []
    cs = []

    for batch in range(params.shape[0]):
        t = np.concatenate([np.zeros(4), params[batch, : seq_len[batch], 0]], axis=-1)
        c = params[batch, : seq_len[batch], 1:3]
        ts.append(t)
        cs.append(c)

    return zip(ts, cs)


def sample_curves(curves, delta):
    import numpy as np
    from scipy.interpolate import splev

    pts_list = []
    for curve in curves:
        t, c = curve
        num_samples = int(t[-1] / delta) + 1
        samples = np.linspace(0, t[-1], num_samples)
        pts = splev(samples, (t, c.T, 3))
        pts_list.append(np.array(pts).T)
    return pts_list


def get_tcks(params):
    import numpy as np

    params = params.cpu().numpy()

    t = params[:, :, 0] * 1024
    t = np.concatenate([np.zeros((params.shape[0], 4)), t], axis=-1)

    c = params[:, :, 1:3] * 1024

    return t, c
