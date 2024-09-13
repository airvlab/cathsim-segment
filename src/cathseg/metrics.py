import lpips
import torch
import torch.nn as nn
from cathseg.custom_modules import BSpline
from piq import fsim
from torchmetrics.functional.classification import binary_jaccard_index


class BinaryMetrics:
    r"""Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative
    rate, as specificity/TPR is meaningless in multiclass cases.
    """

    def __init__(self, eps=1e-5, activation="0-1"):
        self.eps = eps
        self.activation = activation

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(
            -1,
        )
        target = gt.view(
            -1,
        ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)

        return pixel_acc, dice, precision, specificity, recall

    def __call__(self, y_true, y_pred):
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W)
        if self.activation in [None, "none"]:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            sigmoid_pred = nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)
        else:
            raise NotImplementedError("Not a supported activation!")

        assert activated_pred.shape[1] == 1, (
            "Predictions must contain only one channel" " when performing binary segmentation"
        )
        pixel_acc, dice, precision, specificity, recall = self._calculate_overlap_metrics(
            y_true.to(y_pred.device, dtype=torch.float), activated_pred
        )
        return [pixel_acc, dice, precision, specificity, recall]


loss_fn_alex = lpips.LPIPS(net="alex")
binary_metrics = BinaryMetrics()


def compute_all_metrics(y, y_hat):
    fsim_loss = fsim(y, y_hat, chromatic=False)
    pixel_acc, dice, precision, specificity, recall = binary_metrics(y, y_hat)
    y_hat = y_hat.to(device="cpu")
    y = y.to(device="cpu")
    jaccard = binary_jaccard_index(y_hat, y)
    losses = dict(
        fsim=fsim_loss,
        jaccard=jaccard,
        lpips=loss_fn_alex(y, y_hat),
        pixel_acc=pixel_acc,
        dice=dice,
        precision=precision,
        specificity=specificity,
        recall=recall,
    )
    for loss, loss_value in losses.items():
        losses[loss] = torch.round(loss_value, decimals=4).item()
    return losses


class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_seq, tgt_output):
        # Compute pairwise distances
        dist_pred_to_true = torch.cdist(pred_seq, tgt_output)  # (batch_size, seq_len_pred, seq_len_tgt)
        dist_true_to_pred = torch.cdist(tgt_output, pred_seq)  # (batch_size, seq_len_tgt, seq_len_pred)

        # For each point in pred_seq, find the minimum distance to tgt_output
        min_dist_pred_to_true = dist_pred_to_true.min(dim=2)[0]  # (batch_size, seq_len_pred)
        min_dist_true_to_pred = dist_true_to_pred.min(dim=2)[0]  # (batch_size, seq_len_tgt)

        # Average the distances (this is per sequence element, no padding involved yet)
        chamfer_loss_pred_to_true = min_dist_pred_to_true.mean(dim=1)  # (batch_size,)
        chamfer_loss_true_to_pred = min_dist_true_to_pred.mean(dim=1)  # (batch_size,)

        # Final Chamfer loss is the sum of both directions
        chamfer_loss = (chamfer_loss_pred_to_true + chamfer_loss_true_to_pred) / 2

        return chamfer_loss


class MyLossFn(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = 20 / 1024
        self.bspline = BSpline(3)
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred_seq, tgt, tgt_pad_mask):
        pred_seq = torch.clip(pred_seq, 0, 1)
        pts_pred, pts_pred_masks = self.bspline(
            coefficients=pred_seq[:, :, 1:3],
            knots=pred_seq[:, :, 0],
            masks=tgt_pad_mask,
            delta=self.delta,
            batched=True,
        )
        pts_tgt, pts_tgt_masks = self.bspline(
            coefficients=tgt[:, :, 1:3], knots=tgt[:, :, 0], masks=tgt_pad_mask, delta=self.delta, batched=True
        )

        spline_loss = self.mse(pts_pred, pts_tgt)
        spline_loss = spline_loss * pts_tgt_masks.unsqueeze(-1)

        mers = spline_loss.sum() / pts_tgt_masks.sum() * 1024
        mete = spline_loss[:, 0].sum() / spline_loss.shape[0] * 1024
        maxed = spline_loss.max(dim=1)[0].sum() / spline_loss.shape[0] * 1024
        return dict(mers=mers, mete=mete, maxed=maxed)
