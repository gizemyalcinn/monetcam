"""Modeli telefon için LiteRT (.tflite) formatına çevirir.
Linux gerekir (Colab/Kaggle/WSL): pip install litert-torch
Kullanım: python export_tflite.py [checkpoint] [çıktı.tflite]

Modelin çıktısına, telefonda ayrıca kod gerektirmeyen iki düzeltme gömülür:
  1) Karanlık bölge renk koruması: koyu yerlerde renk tonu orijinal fotoğraftan alınır
     (model koyu kahverengiyi yeşile boyayamaz).
  2) Gren temizliği: renk kanalları hafifçe bulanıklaştırılır (renkli noktacık gren gider),
     parlaklık ise sadece düz bölgelerde (gökyüzü, duvar) yumuşatılır; detaylı bölgeler aynen kalır.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from model import load_transformer


def blur(t, k):
    return F.avg_pool2d(t, k, 1, k // 2, count_include_pad=False)


def ycc(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y, (b - y) * 0.564, (r - y) * 0.713


def rgb(y, cb, cr):
    r = y + 1.403 * cr
    b = y + 1.773 * cb
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return torch.clamp(torch.cat([r, g, b], 1), 0.0, 1.0)


class MonetPhone(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = torch.clamp(self.net(x), 0.0, 1.0)
        y0, cb0, cr0 = ycc(x)
        y1, cb1, cr1 = ycc(out)
        # 1) karanlık bölge renk koruması
        k = blur(torch.clamp((y0 - 0.15) / 0.30, 0.0, 1.0), 9)
        gain = torch.clamp(y1 / (y0 + 0.04), 0.5, 2.0)
        cb = k * cb1 + (1 - k) * cb0 * gain
        cr = k * cr1 + (1 - k) * cr0 * gain
        # 2a) renkli gren: renk kanallarını yumuşat
        cb = blur(cb, 7)
        cr = blur(cr, 7)
        # 2b) düz bölgelerde parlaklık greni
        var = torch.clamp(blur(y0 * y0, 7) - blur(y0, 7) ** 2, min=0.0)
        flat = blur(torch.clamp(torch.sqrt(var + 1e-8) / 0.04, 0.0, 1.0), 9)  # 0 düz, 1 detaylı
        y = flat * y1 + (1 - flat) * blur(y1, 7)
        return rgb(y, cb, cr)


if __name__ == "__main__":
    import litert_torch
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/bridge_sw15e9_long_epoch_5.pth"
    out = sys.argv[2] if len(sys.argv) > 2 else "monet_v6_896x672.tflite"
    m = MonetPhone(load_transformer(ckpt)).eval()
    em = litert_torch.convert(m, (torch.rand(1, 3, 896, 672),))
    em.export(out)
    x = torch.rand(1, 3, 896, 672)
    with torch.no_grad():
        ref = m(x).numpy()
    print("Kaydedildi:", out, "| en büyük fark:", float(np.abs(ref - em(x)).max()))
