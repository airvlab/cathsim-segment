import torch
import torchvision
import pytorch_lightning as pl

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

        #self.fine_tune(should_finetune = True)

    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)

        return out

class DecoderWithAttentionEncoder(pl.LightningModule):
    def __init__(self):
        super(DecoderWithAttentionEncoder, self).__init__()
        self.model_enc = Encoder(24)
        #self.model_dec = AttentionNet()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, _ = batch # i.e. (2, 1024, 1024)
        # We now need shape (2, 3, 1024, 1024).
        x = x.view(2, 1, 1024, 1024).expand(-1, 3, -1, -1)
        
        # Push data through Encoder, then Attention Net. with Decoder.
        x = self.model_enc(x)
        #! = self.model_dec(!)

        #loss = self.loss(dec_x, x)

        #self.log("train_loss", loss)
        return torch.tensor(1.0, requires_grad = True)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.model_enc.parameters(),
                               lr = ENC_LEARNING_RATE,
                               momentum = ENC_MOMENTUM) 

# This class also contains the Decoder.
class AttentionNet(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super(AttentionNet, self).__init__()

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_dim = attn_dim

        self.attn_net = AttentionComp(enc_dim, dec_dim, attn_dim)

        self.dropout = nn.Dropout(p = 0.5)
        self.decoding = nn.LSTMCell(2 * enc_dim,
                                    dec_dim, bias = True)
        self.init_c = nn.Linear(enc_dim, dec_dim)
        self.init_h = nn.Linear(enc_dim, dec_dim)
        self.beta = nn.Linear(dec_dim, enc_dim)
        self.sigmoid = nn.Sigmoid()

    def init_h_state(self, enc_out):
        mean_enc_out = enc_out.mean(dim = 1)
        h = self.init_h(mean_enc_out)
        c = self.init_c(mean_enc_out)

        return h, c

    def forward(self, enc_out):
        # Initialise LSTM state.
        h, c = self.init_h_state(enc_out)
        print(h, c)

    def training_step(self, batch):
        x, y = batch

    def testing_step(self, batch):
        x, y = batch

    


# Attention component.
class AttentionComp(nn.Module):
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
    
if __name__ == "__main__":
    X = torch.rand((3, 1024, 1024))
    enc = Encoder(14)

    # First, we test the encoder.
    #encoded = enc(X)
    #print(encoded, encoded.shape)
    print(summary(enc, (3, 1024, 1024)))
