"""Modeli 20 parçaya bölerek tflite'a çevirir (tek dosya, her parça ayrı imza).
tek parça halinde GPU ekranı birkaç saniye donduruyordu."""
import torch, numpy as np, litert_torch
from torch import nn
src = open("export_cyclegan_gpu.py", encoding="utf-8").read()
exec(src[:src.index("CKPT =")])           # ReflectPad, InstanceNorm, ConvTransposeAsConv, gpu_friendly
from cyclegan import load_cyclegan

CKPT = "checkpoints/cyclegan/G_bold_final.pth"
full = load_cyclegan(CKPT).eval()          # referans (değiştirilmemiş)
m = load_cyclegan(CKPT); gpu_friendly(m.g); m.eval()
L = list(m.g.model)
# indeksler: 0-9 kodlayıcı, 10-18 artık bloklar, 19-21 büyütme 1, 22-24 büyütme 2, 25-27 son katman


class Enc(nn.Module):
    def __init__(s): super().__init__(); s.l = nn.Sequential(*L[0:10])
    def forward(s, x): return s.l(x * 2 - 1)

class ResA(nn.Module):
    def __init__(s, blk): super().__init__(); s.l = nn.Sequential(*list(blk.conv_block)[0:4])
    def forward(s, h): return s.l(h)

class ResB(nn.Module):
    def __init__(s, blk): super().__init__(); s.l = nn.Sequential(*list(blk.conv_block)[4:7])
    def forward(s, y, h): return h + s.l(y)

class Dec1(nn.Module):
    def __init__(s): super().__init__(); s.l = nn.Sequential(*L[19:22])
    def forward(s, h): return s.l(h)

class Dec2(nn.Module):
    def __init__(s): super().__init__(); s.l = nn.Sequential(*L[22:28])
    def forward(s, h): return torch.clamp((s.l(h) + 1) / 2, 0.0, 1.0)


def cl(mod, nargs):   # girdi/çıktıları kanal-sonda (NHWC) yap
    return litert_torch.to_channel_last_io(mod.eval(), args=list(range(nargs)), outputs=[0])

H, W = 896, 672
img = torch.rand(1, H, W, 3)
f4 = torch.rand(1, H // 4, W // 4, 256)
f4b = torch.rand(1, H // 4, W // 4, 256)
f2 = torch.rand(1, H // 2, W // 2, 128)

conv = litert_torch.signature("enc", cl(Enc(), 1), (img,))
for i in range(9):
    blk = L[10 + i]
    conv = conv.signature(f"r{i}a", cl(ResA(blk), 1), (f4,))
    # aynı tensörü iki kez verince iki girdiyi tek sanıyor, ayrı tensör lazım
    conv = conv.signature(f"r{i}b", cl(ResB(blk), 2), (f4, f4b))
conv = conv.signature("dec1", cl(Dec1(), 1), (f4,))
conv = conv.signature("dec2", cl(Dec2(), 1), (f2,))
em = conv.convert()
em.export("monet_cyclegan_seg_896x672.tflite")

# kontrol: parçaları sırayla çalıştır, referansla karşılaştır
x = torch.rand(1, 3, H, W)
with torch.no_grad():
    ref = full(x).clamp(0, 1).permute(0, 2, 3, 1).numpy()
xh = x.permute(0, 2, 3, 1).numpy()
h = em(xh, signature_name="enc")
for i in range(9):
    y = em(h, signature_name=f"r{i}a")
    h = em(y, h, signature_name=f"r{i}b")
h = em(h, signature_name="dec1")
out = em(h, signature_name="dec2")
print("parçalı tflite farkı:", float(np.abs(out - ref).max()))
