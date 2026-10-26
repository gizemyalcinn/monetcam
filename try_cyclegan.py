"""Hazır CycleGAN Monet modelini farklı çözünürlüklerde dener.
Kullanım: python try_cyclegan.py images/pelicans.jpeg"""
import sys, time
from pathlib import Path
import torch
from utils import device, load_image, save_image
from cyclegan import load_cyclegan

src = sys.argv[1] if len(sys.argv) > 1 else "images/pelicans.jpeg"
model = load_cyclegan("checkpoints/cyclegan/style_monet.pth", device)
Path("outputs").mkdir(exist_ok=True)
for size in [512, 768, 896]:
    x = load_image(src, max_size=size)
    _, _, H, W = x.shape
    x = x[:, :, :H // 4 * 4, :W // 4 * 4]
    t = time.time()
    with torch.no_grad():
        y = model(x)
    out = f"outputs/cyclegan_{Path(src).stem}_{size}.jpg"
    save_image(y, out)
    print(f"{out}  ({time.time()-t:.2f} sn)")
