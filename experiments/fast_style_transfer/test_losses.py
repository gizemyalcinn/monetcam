import torch
from utils import device, load_image
from losses import (load_vgg, get_features, gram_matrix,
                    content_loss, style_loss, STYLE_LAYERS)

content = load_image("images/pelicans.jpeg")
style = load_image("images/waterlily.jpg")

vgg = load_vgg(device)

with torch.no_grad():
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

for name in STYLE_LAYERS:
    c = content_features[name]
    s = style_features[name]
    print(f"{name}: içerik {tuple(c.shape)} → Gram {tuple(gram_matrix(c).shape)} | "
          f"stil {tuple(s.shape)} → Gram {tuple(gram_matrix(s).shape)}")

style_grams = {name: gram_matrix(style_features[name]) for name in STYLE_LAYERS}

print("İçerik kaybı (fotoğraf vs kendisi):", content_loss(content_features, content_features).item())
print("Stil kaybı (fotoğraf vs Monet):   ", style_loss(content_features, style_grams).item())
print("Stil kaybı (Monet vs kendisi):    ", style_loss(style_features, style_grams).item())