import matplotlib.pyplot as plt
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


def visualize_decoder_attention(image, decoder_attentions, layer=0, patch_size=32):
    batch_size, channels, height, width = image.shape
    num_patches_per_dim = height // patch_size  # Assuming square image
    num_points = decoder_attentions.shape[2]  # Number of points in the sequence

    # Determine the grid size for plotting (e.g., 4x5 for up to 20 points)
    grid_cols = 5
    grid_rows = (num_points + grid_cols - 1) // grid_cols  # Calculate rows based on the number of points

    # Create a figure with subplots for each point in the sequence
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, grid_rows * 3))

    # Flatten axes for easier indexing
    axes = axes.flatten()

    for point_index in range(num_points):
        # Select the attention map for the specified layer and point
        attention_map = decoder_attentions[:, layer, point_index, :]  # Shape: (batch_size, num_patches)

        # Reshape to (batch_size, num_patches_per_dim, num_patches_per_dim)
        attention_map = attention_map.reshape(batch_size, num_patches_per_dim, num_patches_per_dim)

        # Upsample to match the original image size
        attention_map = F.interpolate(
            attention_map.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)

        # Normalize the attention map for visualization
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # Overlay the attention map on the image
        for i in range(batch_size):
            axes[point_index].imshow(image[i].squeeze(), cmap="gray")  # Assuming grayscale input image
            axes[point_index].imshow(attention_map[i].cpu().detach().numpy(), cmap="jet", alpha=0.5)
            axes[point_index].axis("off")
            axes[point_index].set_title(f"Point {point_index + 1}")

    # Remove any empty subplots
    for idx in range(num_points, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()
