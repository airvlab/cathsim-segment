import cv2
import guide3d.representations.curve as curve
import numpy as np
import pytorch_lightning as pl
from guide3d.dataset.image.bezier import visualize_bezier
from pytorch_lightning import Callback
from scipy.interpolate import splev

import wandb


def visualize_bezier(control_points, img):
    # Reconstruct the Bezier curve from the control points
    t_values = np.linspace(0, 1, 100)  # Parameter values for the smooth curve
    bezier_points = curve.bezier_curve(control_points, t_values)


def plot_images(img_true, img_pred, img_gen):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_true, cmap="gray")
    ax[1].imshow(img_pred, cmap="gray")
    ax[2].imshow(img_gen, cmap="gray")

    ax[0].set_title("GT")
    ax[1].set_title("Pred")
    ax[2].set_title("Inference")

    for a in ax:
        a.axis("off")

    plt.show()
    plt.close()


class ImageCallbackLoggerBezier(Callback):
    def __init__(self, img_untransform: callable, c_untransform: callable, t_untransform: callable):
        self.img_untransform = img_untransform
        self.c_untransform = c_untransform
        self.t_untransform = t_untransform

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        instances = pl_module.val_step_outputs
        instances = instances[: min(len(instances), 10)]

        table = wandb.Table(columns=["GT", "Preds", "Inference"])
        for instance in instances:
            img_true, img_pred, img_gen = self.make_images(instance)

            table.add_data(wandb.Image(img_true), wandb.Image(img_pred), wandb.Image(img_gen))
        trainer.logger.experiment.log({"img_samples": table})

        pl_module.train()
        pl_module.val_step_outputs = []

    def draw_points(self, img, c):
        img = img.copy()

        t_values = np.linspace(0, 1, 100)  # Parameter values for the smooth curve
        bezier_points = curve.bezier_curve(c, t_values)

        def in_bounds(x, y):
            return 0 <= x < img.shape[1] and 0 <= y < img.shape[0]

        for control_point in c.astype(np.int32):
            if not in_bounds(control_point[0], control_point[1]):
                continue
            img = cv2.circle(img, tuple(control_point), 2, 255, -1)

        points = np.array(bezier_points).astype(np.int32)  # sampled_c should be in (n_points, 2)
        points = points[None, ...]  # Shape it into (1, n_points, 2)

        img = cv2.polylines(img, points, isClosed=False, color=255, thickness=1)

        return img

    def make_images(self, instance):
        # Untransform the image to its original format (grayscale)
        img = self.img_untransform(instance["img"].detach().cpu().numpy())
        seq_len = instance["seq_len"].detach().cpu().numpy().astype(int)
        c_pred = self.c_untransform(instance["c_pred"].detach().cpu().numpy()[:seq_len])
        c_true = self.c_untransform(instance["c_true"].detach().cpu().numpy()[:seq_len])
        c_gen = self.c_untransform(instance["c_gen"].detach().cpu().numpy())

        img = img[0] * 255  # Scaling the grayscale image
        img = img.astype(np.uint8)

        img_true = self.draw_points(img, c_true)
        img_pred = self.draw_points(img, c_pred)
        img_gen = self.draw_points(img, c_gen)

        # plot_images(img_true, img_pred, img_gen)

        return [img_true, img_pred, img_gen]


class ImageCallbackLogger(Callback):
    def __init__(self, img_untransform: callable, c_untransform: callable, t_untransform: callable):
        self.img_untransform = img_untransform
        self.c_untransform = c_untransform
        self.t_untransform = t_untransform

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        instances = pl_module.val_step_outputs
        instances = instances[: min(len(instances), 10)]

        table = wandb.Table(columns=["GT", "Preds", "Inference"])
        for instance in instances:
            img_true, img_pred, img_gen = self.make_images(instance)

            table.add_data(wandb.Image(img_true), wandb.Image(img_pred), wandb.Image(img_gen))
        trainer.logger.experiment.log({"img_samples": table})

        pl_module.train()
        pl_module.val_step_outputs = []

    def draw_points(self, img, c, t):
        img = img.copy()

        def in_bounds(x, y):
            return 0 <= x < img.shape[1] and 0 <= y < img.shape[0]

        samples = np.linspace(0, t[-1], 50)
        sampled_c = splev(samples, (t, c.T, 3))

        for control_point in c.astype(np.int32):
            if not in_bounds(control_point[0], control_point[1]):
                continue
            img = cv2.circle(img, tuple(control_point), 2, 255, -1)

        points = np.array(sampled_c).T.astype(np.int32)  # sampled_c should be in (n_points, 2)
        points = points[None, ...]  # Shape it into (1, n_points, 2)

        img = cv2.polylines(img, points, isClosed=False, color=255, thickness=1)

        return img

    def make_images(self, instance):
        # Untransform the image to its original format (grayscale)
        img = self.img_untransform(instance["img"].detach().cpu().numpy())
        seq_len = instance["seq_len"].detach().cpu().numpy().astype(int)
        c_pred = self.c_untransform(instance["c_pred"].detach().cpu().numpy()[:seq_len])
        c_true = self.c_untransform(instance["c_true"].detach().cpu().numpy()[:seq_len])
        c_gen = self.c_untransform(instance["c_gen"].detach().cpu().numpy())
        t_pred = self.t_untransform(instance["t_pred"].detach().cpu().numpy()[:seq_len]).flatten()
        t_true = self.t_untransform(instance["t_true"].detach().cpu().numpy()[:seq_len]).flatten()
        t_gen = self.t_untransform(instance["t_gen"].detach().cpu().numpy()).flatten()

        # Extend t_pred, t_true, and t_gen to include zeros
        t_pred = np.concatenate([np.zeros((4)), t_pred], axis=0)
        t_true = np.concatenate([np.zeros((4)), t_true], axis=0)
        t_gen = np.concatenate([np.zeros((4)), t_gen], axis=0)

        img = img[0] * 255  # Scaling the grayscale image
        img = img.astype(np.uint8)

        img_true = self.draw_points(img, c_true, t_true)
        img_pred = self.draw_points(img, c_pred, t_pred)
        img_gen = self.draw_points(img, c_gen, t_gen)

        # plot_images(img_true, img_pred, img_gen)

        return [img_true, img_pred, img_gen]


class ImageCallbackLoggerPoints(Callback):
    def __init__(self, img_untransform: callable, c_untransform: callable, t_untransform: callable):
        self.img_untransform = img_untransform
        self.c_untransform = c_untransform
        self.t_untransform = t_untransform

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        instances = pl_module.val_step_outputs
        instances = instances[: min(len(instances), 10)]

        table = wandb.Table(columns=["GT", "Preds", "Inference"])
        for instance in instances:
            img_true, img_pred, img_gen = self.make_images(instance)

            table.add_data(wandb.Image(img_true), wandb.Image(img_pred), wandb.Image(img_gen))
        trainer.logger.experiment.log({"img_samples": table})

        pl_module.train()
        pl_module.val_step_outputs = []

    def draw_points(self, img, c):
        img = img.copy()

        def in_bounds(x, y):
            return 0 <= x < img.shape[1] and 0 <= y < img.shape[0]

        for control_point in c.astype(np.int32):
            if not in_bounds(control_point[0], control_point[1]):
                continue
            img = cv2.circle(img, tuple(control_point), 2, 255, -1)

        return img

    def make_images(self, instance):
        # Untransform the image to its original format (grayscale)
        img = self.img_untransform(instance["img"].detach().cpu().numpy())
        seq_len = instance["seq_len"].detach().cpu().numpy().astype(int)
        c_pred = self.c_untransform(instance["c_pred"].detach().cpu().numpy()[:seq_len])
        c_true = self.c_untransform(instance["c_true"].detach().cpu().numpy()[:seq_len])
        c_gen = self.c_untransform(instance["c_gen"].detach().cpu().numpy())

        img = img[0] * 255  # Scaling the grayscale image
        img = img.astype(np.uint8)

        img_true = self.draw_points(img, c_true)
        img_pred = self.draw_points(img, c_pred)
        img_gen = self.draw_points(img, c_gen)

        # plot_images(img_true, img_pred, img_gen)

        return [img_true, img_pred, img_gen]
