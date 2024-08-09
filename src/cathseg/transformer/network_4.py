import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from guide3d.dataset.image.spline import Guide3D as Dataset
from torchvision import models, transforms

dataset_path = Path.home() / "data" / "segment-real"

image_transforms = transforms.Compose(
    [
        # transforms.Lambda(lambda x: x / 255.0),
        # transforms.Normalize((0.5,), (0.5,)),
        # gray to RGB
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ]
)


class ImageFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageFeatureExtractor, self).__init__()
        # Using ResNet as the feature extractor
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-2]
        )  # Remove last layers

    def forward(self, x):
        # Extract features
        features = self.resnet(x)  # (batch_size, 2048, H, W)
        features = features.view(
            features.size(0), features.size(1), -1
        )  # (batch_size, 2048, H*W)
        features = features.permute(0, 2, 1)  # (batch_size, H*W, 2048)
        return features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class ControlPointTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        dropout=0.1,
    ):
        super(ControlPointTransformerDecoder, self).__init__()

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Linear layer to project decoder output to 2D control points
        self.output_layer = nn.Linear(d_model, 2)  # 2D control points

    def forward(self, image_features, tgt_seq, tgt_mask=None, memory_mask=None):
        # Apply positional encoding to the target sequence
        tgt_seq = self.positional_encoding(tgt_seq)

        # Pass through the transformer decoder
        decoder_output = self.transformer_decoder(
            tgt_seq, image_features, tgt_mask=tgt_mask, memory_mask=memory_mask
        )

        # Project the decoder output to 2D control points
        predicted_points = self.output_layer(decoder_output)

        return predicted_points


class BSplineControlPointPredictor(nn.Module):
    def __init__(
        self, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length
    ):
        super(BSplineControlPointPredictor, self).__init__()
        self.image_feature_extractor = ImageFeatureExtractor()
        self.control_point_decoder = ControlPointTransformerDecoder(
            d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length
        )

    def forward(self, images, tgt_seq, tgt_mask=None, memory_mask=None):
        image_features = torch.rand((4, 256))

        # Predict the sequence of control points
        predicted_control_points = self.control_point_decoder(
            image_features, tgt_seq, tgt_mask, memory_mask
        )

        return predicted_control_points


def main():
    model_dimensionality = 256
    n_heads = 8
    num_decoder_layers = 6
    dim_feedforward = 512
    max_seq_length = 20
    batch_size = 4
    num_epochs = 10
    learning_rate = 0.001

    # Assuming you have a dataset class named Dataset
    train_dataset = Dataset(
        root=dataset_path,
        annotations_file="sphere_wo_reconstruct.json",
        split="train",
        with_mask=True,
        max_length=max_seq_length,
        image_transform=image_transforms,
    )

    # Create the Transformer Decoder Model
    model = BSplineControlPointPredictor(
        model_dimensionality,
        n_heads,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(max_seq_length)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            img, ts, cs, seq_lengths, masks = batch
            img = torch.rand((batch_size, 3, 224, 224))

            # Prepare the target sequence by concatenating ts and cs
            ts = ts.unsqueeze(-1)
            tgt = ts
            print(tgt.shape)

            # Assuming 'img' is the input image and should be processed by the model
            output = model(img, tgt, tgt_mask=tgt_mask)
            exit()

            # Compute loss
            loss = criterion(output, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:  # Print every 10 batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    print("Finished Training")


if __name__ == "__main__":
    main()
