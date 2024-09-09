import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


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


def plot_attention_maps(gen, processed_attentions, img=None):
    num_points = len(gen)
    grid_cols = 5
    grid_rows = (num_points + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, grid_rows * 3))
    axes = axes.flatten()
    for point_index in range(num_points):
        if img is None:
            axes[point_index].imshow(processed_attentions[point_index])
            continue
        axes[point_index].imshow(img)
        axes[point_index].imshow(processed_attentions[point_index], alpha=0.5)
        continue

    for idx in range(len(axes)):
        axes[idx].axis("off")
    fig.savefig("attention_map.png")
    plt.show()


def preprocess_instance(ma_dict):
    for key in ma_dict:
        ma_dict[key] = ma_dict[key].detach().cpu().numpy()
    return ma_dict
