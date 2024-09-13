import torch
from x_transformers import Decoder, Encoder, TransformerWrapper, ViTransformerWrapper

encoder = ViTransformerWrapper(image_size=256, patch_size=32, attn_layers=Encoder(dim=512, depth=6, heads=8))

decoder = TransformerWrapper(
    num_tokens=20000, max_seq_len=1024, attn_layers=Decoder(dim=512, depth=6, heads=8, cross_attend=True)
)

img = torch.randn(1, 3, 256, 256)
caption = torch.randint(0, 20000, (2, 1024))
word_embeddings = torch.randn(1, 1024, 512)
print(caption.shape)

encoded = encoder(img, return_embeddings=True)
out = decoder(None, prepend_embeddings=word_embeddings, context=encoded)  # (1, 1024, 20000)
print(out.shape)
