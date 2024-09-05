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
    img_size=1024,
    channels=1,
    layer=None,
    patch_size=32,
    discard_ratio=0.9,
    aggreg_func: callable = lambda x: torch.max(x, dim=2),
):
    num_patches_per_dim = img_size // patch_size

    batch_size, num_layers, num_heads, num_points, num_patches = decoder_attentions.shape

    decoder_attentions = aggreg_func(decoder_attentions)[0] # Default behaviour: average across heads.
    
    print("decoder attentions", decoder_attentions.shape)
    rollout_attn = torch.eye(num_points).unsqueeze(0).repeat(batch_size, 1, 1)
    print("rollout_attn", rollout_attn.shape)

    for curr_layer in range(num_layers):
        attention_layer = decoder_attentions[:, curr_layer]

        # Mask over least significant regions based on discard_ratio.
        #flattened_attn = attention_layer.view(batch_size, -1)
        _, idxs = attention_layer.topk(
                int(num_points * discard_ratio),
                -1,
                False
                )
        idxs = idxs[idxs != 0]
        attention_layer[:, :, idxs] = 0

        #attention_layer = torch.where(attention_layer >= threshold,
        #                              attention_layer,
        #                              torch.zeros_like(attention_layer))

        identity = torch.eye(num_points).unsqueeze(0)# Identity for skip connections.
        
        adjusted_attention = torch.cat([identity, attention_layer[:, :, num_points:]], dim = 2)
        
        attention_map = rollout_attn @ adjusted_attention

    if layer is not None:
        attention_map = decoder_attentions[:, layer]
    else:
        attention_map = decoder_attentions.mean(dim=1)

    # Reshape attention map to 2D patch layout for each batch (batch_size, num_points, H_patches, W_patches)
    attention_map = attention_map.view(batch_size, num_points, num_patches_per_dim, num_patches_per_dim)

    # Upsample the attention map to match the original image size
    attention_map = F.interpolate(attention_map, size=(img_size, img_size), mode="bilinear", align_corners=False)

    # Normalize attention maps across each item in the batch
    attention_map_min = (
        attention_map.view(batch_size, num_points, -1).min(dim=2, keepdim=True)[0].view(batch_size, num_points, 1, 1)
    )
    attention_map_max = (
        attention_map.view(batch_size, num_points, -1).max(dim=2, keepdim=True)[0].view(batch_size, num_points, 1, 1)
    )
    attention_map = (attention_map - attention_map_min) / (attention_map_max - attention_map_min + 1e-6)

    return attention_map  # (batch_size, num_points, img_size, img_size)


def plot_attention_maps(gen, processed_attentions, img=None):
    num_points = len(gen)
    grid_cols = 5
    grid_rows = (num_points + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, grid_rows * 3))
    axes = axes.flatten()
    for point_index in range(num_points):
        if img:
            axes[point_index].imshow(img)
            axes[point_index].imshow(processed_attentions[point_index], alpha=0.5)
            continue
        axes[point_index].imshow(processed_attentions[point_index])

    for idx in range(len(axes)):
        axes[idx].axis("off")
    plt.show()


def preprocess_instance(ma_dict):
    for key in ma_dict:
        ma_dict[key] = ma_dict[key].detach().cpu().numpy()
    return ma_dict
