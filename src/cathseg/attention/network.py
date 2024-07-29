import torch
import torchvision

from cathseg.attention.hyperparams import (
        ENC_LEARNING_RATE,
        ENC_MOMENTUM,
        DEC_LEARNING_RATE,
        DEC_MOMENTUM
        )
from torch import nn
from torchsummary import summary

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, enc_img_size):
        super(Encoder, self).__init__()
        
        self.enc_img_size = enc_img_size
        resnet = torchvision.models.resnet101(pretrained = True)

        # Remove linear and pool layers as we aren't doing classification.
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Resize image to fixed size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((enc_img_size,
                                                   enc_img_size))

        self.fine_tune(should_finetune = True)

    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(images)

    def fine_tune(self, should_finetune = True):
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = should_finetune

# AttentionNet also contains the encoder module.
class AttentionNet(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super(AttentionNet, self).__init__()

        self.encoder_attn = nn.Linear(encoder_dim, attn_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attn_dim)
        self.full_attn = nn.Linear(attn_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, enc_out, dec_hidden):
        attn1 = self.encoder_attn(enc_out)
        attn2 = self.decoder_attn(dec_hidden)
        attn = self.full_attn(
                self.relu(attn1 + attn2.unsqueeze(1))
                ).squeeze(2)
        alpha = self.softmax(attn)
        attn_weighted_enc = (enc_out * alpha.unsqueeze(2)).sum(dim = 1)

        return attn_weighted_enc, alpha
    
    def training_step(self, batch):
        x, y = batch

    def validation_step(self, batch):
        x, y = batch

    def test_step(self, batch):
        x, y = batch

    def configure_optimizers(self):
        enc_optimiser = torch.optim.Adam(f,
                                lr = ENC_LEARNING_RATE,
                                momentum = ENC_MOMENTUM
                )
        dec_optimiser = torch.optim.Adam(f,
                                         lr = DEC_LEARNING_RATE,
                                         momentum = DEC_MOMENTUM)

        return enc_optimiser, dec_optimiser

    def predict(self, x):
        return NotImplemented

if __name__ == "__main__":
    X = torch.rand((3, 1024, 1024))
    enc = Encoder(14)

    # First, we test the encoder.
    #encoded = enc(X)
    #print(encoded, encoded.shape)
    print(summary(enc, (3, 1024, 1024)))
