import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data


class DummyData(data.Dataset):
    def __init__(self, num_samples, X_shape, seq_len):
        self.num_samples = num_samples
        self.X_shape = X_shape
        self.point_dims = 2
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X = torch.randn(self.X_shape)

        # Random sequence length between 1 and seq_len (inclusive)
        seq_len = torch.randint(1, self.seq_len + 1, (1,)).item()

        # Create random sequences for c_seq and t_seq
        c_seq = torch.randn(seq_len, self.point_dims)
        t_seq = torch.randn(seq_len, 1)

        # Concatenate c_seq and t_seq along the last dimension
        target_seq = torch.cat([c_seq, t_seq], dim=-1)

        # Create the mask, initially all ones
        target_mask = torch.ones(self.seq_len)

        # Zero out the mask from the end of the current sequence length to seq_len
        target_mask[seq_len:] = 0

        # Pad target_seq to the right shape
        target_seq = F.pad(target_seq, (0, 0, 0, self.seq_len - seq_len))

        return X, target_seq, target_mask


def visualize_encoder_attention(image, atten, layer=None):
    batch_size, channels, height, width = image.shape

    print("Attention shape:", atten.shape)

    if layer is not None:
        atten = atten[:, layer, :, :]
    else:
        atten = atten.mean(dim=1)

    print("Selected layer attention shape:", atten.shape)

    # Interpolate to match the original image size
    attention_map = F.interpolate(
        atten.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
    ).squeeze(1)

    # Normalize the attention map for visualization
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Plot the original image and attention map overlay
    for i in range(batch_size):
        fig, ax = plt.subplots()
        ax.imshow(image[i].squeeze(), cmap="gray")  # Assuming grayscale input image
        ax.imshow(attention_map[i].cpu().detach().numpy(), cmap="jet", alpha=0.5)
        ax.axis("off")
        plt.show()


def process_attention_maps(
    decoder_attentions,
    img_size,
    channels,
    patch_size,
    layer=None,
    aggreg_func: callable = lambda x: torch.max(x, dim=2)[0],  # Max fusion by default
    discard_ratio: float = None,
):
    num_patches_per_dim = img_size // patch_size

    batch_size, num_layers, num_heads, num_points, num_patches = decoder_attentions.shape

    # Reshape attention to grid size (num_patches_per_dim x num_patches_per_dim)
    attention_map = decoder_attentions.view(
        batch_size, num_layers, num_heads, num_points, num_patches_per_dim, num_patches_per_dim
    )

    # Normalize each attention map across its spatial dimensions
    attention_map_min = (
        attention_map.view(batch_size, num_layers, num_heads, num_points, -1)
        .min(dim=4, keepdim=True)[0]
        .view(batch_size, num_layers, num_heads, num_points, 1, 1)
    )
    attention_map_max = (
        attention_map.view(batch_size, num_layers, num_heads, num_points, -1)
        .max(dim=4, keepdim=True)[0]
        .view(batch_size, num_layers, num_heads, num_points, 1, 1)
    )
    attention_map = (attention_map - attention_map_min) / (attention_map_max - attention_map_min + 1e-6)

    # Apply aggregation (max fusion or other function)
    decoder_attentions = aggreg_func(attention_map)

    # Select the specific layer after aggregation
    if layer is not None:
        attention_map = decoder_attentions[:, layer]
    else:
        attention_map = decoder_attentions.mean(dim=1)  # Default to averaging across layers if no layer specified

    # Discard the lowest pixels based on discard_ratio
    threshold = attention_map.max() * discard_ratio
    attention_map[attention_map < threshold] = 0

    # Upsample the attention map to match the original image size
    attention_map = F.interpolate(attention_map, size=(img_size, img_size), mode="bilinear", align_corners=False)

    return attention_map  # (batch_size, num_points, img_size, img_size)


def draw_points(img, c, t, control_pts_size: float = 10, line_thickness: int = 1):
    import cv2
    import numpy as np
    from scipy.interpolate import splev

    img = img.copy()

    def in_bounds(x, y):
        return 0 <= x < img.shape[1] and 0 <= y < img.shape[0]

    samples = np.linspace(0, t[-1], 50)
    sampled_c = splev(samples, (t, c.T, 3))

    for control_point in c.astype(np.int32):
        if not in_bounds(control_point[0], control_point[1]):
            continue
        img = cv2.circle(img, tuple(control_point), control_pts_size, 255, -1)

    points = np.array(sampled_c).T.astype(np.int32)  # sampled_c should be in (n_points, 2)
    points = points[None, ...]  # Shape it into (1, n_points, 2)

    img = cv2.polylines(img, points, isClosed=False, color=255, thickness=line_thickness)

    return img


def plot_attention_maps(gen, processed_attentions, img=None):
    num_points = len(gen)
    grid_cols = 5
    grid_rows = (num_points + grid_cols - 1) // grid_cols
    grid_rows = min(3, grid_rows)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, grid_rows * 3))
    axes = axes.flatten()

    if img is not None:
        img = img.squeeze(0).detach().cpu().numpy()
        gen = gen.squeeze(0)
        t = gen[:, 0:1].detach().cpu().numpy().flatten()
        t = np.concatenate([np.zeros((4)), t], axis=0)
        c = gen[:, 1:3].detach().cpu().numpy()
        t = t * 1024
        c = c * 1024
        img = img * 0.5 + 0.5
        img = img * 255
        img = img.astype(np.uint8)
        img = draw_points(img, c, t)[:, :, np.newaxis]
        img = np.concatenate([img, img, img], -1)

    for point_index in range(num_points):
        if img is None:
            axes[point_index].imshow(processed_attentions[point_index])
            continue

        axes[point_index].imshow(img)
        axes[point_index].imshow(processed_attentions[point_index], alpha=0.15, cmap="jet")
        if point_index % (grid_rows * grid_cols - 1) == 0 and point_index > 0:
            break
        continue

    for idx in range(len(axes)):
        axes[idx].axis("off")
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("attention_map.png", bbox_inches="tight")
    plt.show()


def get_latest_ckpt(ckpt_dir: str):
    import re
    from pathlib import Path

    # Convert the directory path to a Path object
    ckpt_path = Path(ckpt_dir)

    # Validate if the directory exists
    if not ckpt_path.is_dir():
        return f"Error: Directory '{ckpt_dir}' does not exist."

    # Regular expression to match the checkpoint files (e.g., epoch=154-step=31775.ckpt)
    ckpt_pattern = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt")

    # List all valid checkpoint files in the directory
    ckpt_info = []
    for file in ckpt_path.iterdir():  # Iterate over files in the directory
        if file.is_file() and file.suffix == ".ckpt":  # Ensure it's a .ckpt file
            match = ckpt_pattern.search(file.name)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                ckpt_info.append((epoch, step, file.name))

    # Check if any valid checkpoints were found
    if not ckpt_info:
        return "No valid checkpoint files found in the directory."

    # Sort by epoch and step (latest epoch and step come last)
    ckpt_info.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # Get the latest checkpoint file
    latest_ckpt = ckpt_info[0][2]

    return ckpt_path / latest_ckpt
