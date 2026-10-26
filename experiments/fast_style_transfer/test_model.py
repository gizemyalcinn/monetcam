import torch
from utils import device, load_image
from model import TransformerNet

model = TransformerNet().to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parametre sayısı: {n_params:,}")

for path in ["images/pelicans.jpeg", "images/waterlily.jpg"]:
    x = load_image(path)
    with torch.no_grad():
        y = model(x)
    print(f"{path}: girdi {tuple(x.shape)} → çıktı {tuple(y.shape)}")