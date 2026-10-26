"""CycleGAN'ı telefonun GPU'sunda çalışabilecek şekilde TFLite'a çevirir.

Eski dışa aktarımda (export_cyclegan_tflite.py) iki şey GPU'da derlenmiyordu:
  1) ReflectionPad2d, dosyada GATHER_ND + tamsayı indeks hesaplarına dönüşüyordu -> GPU desteklemiyor.
  2) Girdi/çıktı NCHW olduğu için modelin her yerine ~110 TRANSPOSE ekleniyordu.
Burada:
  - Yansıtmalı dolgu, dilimleme + birleştirme (SLICE + CONCATENATION) ile birebir aynı şekilde yapılıyor.
  - InstanceNorm, GPU'nun bildiği MEAN/MUL/RSQRT işlemleriyle açıkça yazılıyor.
  - Girdi ve çıktı kanal-sonda (NHWC: 1x896x672x3) oluyor; Android tarafı da böyle yazıp okuyor.
  - ConvTranspose2d yine pad + conv + pixel_shuffle ile (export_cyclegan_tflite.py'deki gibi).
"""
import torch, numpy as np, litert_torch
import torch.nn.functional as F
from torch import nn
from cyclegan import load_cyclegan


class ReflectPad(nn.Module):
    """nn.ReflectionPad2d(p) ile birebir aynı; sadece dilim + birleştirme kullanır."""
    def __init__(self, p):
        super().__init__(); self.p = p
    def forward(self, x):
        p = self.p
        W = x.shape[3]
        left = [x[:, :, :, i:i + 1] for i in range(p, 0, -1)]
        right = [x[:, :, :, W - 1 - i:W - i] for i in range(1, p + 1)]
        x = torch.cat(left + [x] + right, dim=3)
        H = x.shape[2]
        top = [x[:, :, i:i + 1, :] for i in range(p, 0, -1)]
        bottom = [x[:, :, H - 1 - i:H - i, :] for i in range(1, p + 1)]
        return torch.cat(top + [x] + bottom, dim=2)


class InstanceNorm(nn.Module):
    """nn.InstanceNorm2d(affine=False) ile aynı, ama GPU'nun yarım hassasiyetinde (fp16) taşmayan sürüm.

    fp16 en fazla 65504'ü tutabiliyor. Bu ağda InstanceNorm'a giren değerler 30.000'e çıkıyor ve
    varyans için kareleri alınınca taşıyor -> resim siyah/bozuk çıkıyordu. Burada:
      - Ortalama iki adımda alınıyor (önce satır boyunca /W, sonra sütun boyunca /H): ara toplamlar küçük kalır.
      - Her kanal kendi en büyük mutlak değerine (s) bölünüyor: d/s her zaman -1..1 arasında,
        karesi de 0..1 arasında; asla taşmaz.
      - Sonuç: d/s * rsqrt(var(d/s) + eps/s²) = d * rsqrt(var(d) + eps)  -> matematiksel olarak birebir aynı."""
    def __init__(self, eps=1e-5):
        super().__init__(); self.eps = eps
    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        mean = (x * (1.0 / W)).sum(dim=3, keepdim=True)
        mean = (mean * (1.0 / H)).sum(dim=2, keepdim=True)
        d = x - mean
        s = d.abs().amax(dim=3, keepdim=True).amax(dim=2, keepdim=True) + 1e-3
        inv_s = 1.0 / s
        dn = d * inv_s
        var = (dn * dn * (1.0 / W)).sum(dim=3, keepdim=True)
        var = (var * (1.0 / H)).sum(dim=2, keepdim=True)
        return dn * torch.rsqrt(var + self.eps * inv_s * inv_s)


class ConvTransposeAsConv(nn.Module):
    """ConvTranspose2d(k=3, s=2, p=1, op=1) ile birebir aynı.
    Çıktının her 2x2 bloğundaki 4 piksel (a=satır, b=sütun), girdinin 2x2 komşuluğundan
    normal bir evrişimle hesaplanır (4 alt-çekirdek). Sonra bu 4 parça yerlerine dizilir.
    pixel_shuffle 6 boyutlu RESHAPE/TRANSPOSE'a dönüşüp GPU'da derlenmediği için dizme işi
    kanal-sonda (NHWC) sadece 4 boyutlu dilim + birleştirme + reshape ile yapılıyor."""
    def __init__(self, ct):
        super().__init__()
        W = ct.weight.detach()            # (in, out, 3, 3)
        cin, cout = W.shape[0], W.shape[1]
        K = torch.zeros(2, 2, cout, cin, 2, 2)     # kanal sırası: [a][b][c]
        taps = {0: [(0, 1)], 1: [(0, 2), (1, 0)]}
        for a_ in (0, 1):
            for b_ in (0, 1):
                for dy, ky in taps[a_]:
                    for dx, kx in taps[b_]:
                        K[a_, b_, :, :, dy, dx] = W[:, :, ky, kx].t()
        self.cout = cout
        self.weight = nn.Parameter(K.reshape(4 * cout, cin, 2, 2), requires_grad=False)
        self.bias = nn.Parameter(ct.bias.detach().repeat(4), requires_grad=False)
    def forward(self, x):
        C = self.cout
        y = F.conv2d(F.pad(x, (0, 1, 0, 1)), self.weight, self.bias)   # N, 4C, H, W
        N, _, H, W = y.shape
        t = y.permute(0, 2, 3, 1)                                      # N, H, W, [a][b][c]
        rows = [t[..., a * 2 * C:(a + 1) * 2 * C].reshape(N, H, 2 * W, C) for a in (0, 1)]
        out = torch.cat(rows, dim=2).reshape(N, 2 * H, 2 * W, C)         # satırlar araya girer
        return out.permute(0, 3, 1, 2)


CONV_SCALE = 1.0 / 16   # InstanceNorm'dan önceki evrişimlerin çıktısı bu kadar küçültülür (fp16 taşmasın)


def gpu_friendly(module):
    """Modeldeki GPU'da sorun çıkaran katmanları eşdeğerleriyle değiştirir (yerinde).

    Ayrıca InstanceNorm'dan hemen önce gelen her evrişimin ağırlık ve bias'ını CONV_SCALE ile çarpar.
    InstanceNorm ölçekten bağımsız olduğu için (IN(c*y) = IN(y), eps de c² ile ölçeklenince birebir)
    sonuç değişmez; ama evrişim çıktıları ~30.000 yerine ~2.000 olur ve fp16'da rahatça sığar."""
    children = list(module.named_children())
    for i, (name, child) in enumerate(children):
        if isinstance(child, nn.ReflectionPad2d):
            setattr(module, name, ReflectPad(child.padding[0]))
        elif isinstance(child, nn.InstanceNorm2d):
            prev = getattr(module, children[i - 1][0]) if i > 0 else None
            eps = child.eps
            if isinstance(prev, (nn.Conv2d, ConvTransposeAsConv)):
                with torch.no_grad():
                    prev.weight.mul_(CONV_SCALE)
                    prev.bias.mul_(CONV_SCALE)
                eps = eps * CONV_SCALE ** 2
            setattr(module, name, InstanceNorm(eps))
        elif isinstance(child, nn.ConvTranspose2d):
            setattr(module, name, ConvTransposeAsConv(child))
        else:
            gpu_friendly(child)


class Clamp(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return torch.clamp(s.m(x), 0.0, 1.0)


CKPT = "checkpoints/cyclegan/G_bold_final.pth"
ref = Clamp(load_cyclegan(CKPT)).eval()
m = load_cyclegan(CKPT)
gpu_friendly(m.g)
mm = Clamp(m).eval()

x = torch.rand(1, 3, 128, 96)
with torch.no_grad():
    print("PyTorch eşdeğerlik farkı:", float((mm(x) - ref(x)).abs().max()))

nhwc = litert_torch.to_channel_last_io(mm, args=[0], outputs=[0])
em = litert_torch.convert(nhwc, (torch.rand(1, 896, 672, 3),))
em.export("monet_cyclegan_gpu_896x672.tflite")

x = torch.rand(1, 3, 896, 672)
with torch.no_grad():
    t = ref(x).permute(0, 2, 3, 1).numpy()
print("tflite farkı:", float(np.abs(t - em(x.permute(0, 2, 3, 1))).max()))
