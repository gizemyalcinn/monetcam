import torch, numpy as np, litert_torch
import torch.nn.functional as F
from torch import nn
from cyclegan import load_cyclegan

class ConvTransposeAsConv(nn.Module):
    """ConvTranspose2d(k=3, stride=2, padding=1, output_padding=1) ile birebir aynı sonuç.
    Çıktının her 2x2 bloğundaki 4 pikselin her biri, girdinin 2x2 komşuluğundan normal bir
    evrişimle hesaplanır (4 alt-çekirdek); sonra pixel_shuffle bunları yerine dizer.
    Dönüştürücü bu işlemleri destekliyor ve dosyaya büyük sabitler gömülmüyor."""
    def __init__(self, ct):
        super().__init__()
        W = ct.weight.detach()            # (in, out, 3, 3)
        cin, cout = W.shape[0], W.shape[1]
        K = torch.zeros(cout, 2, 2, cin, 2, 2)
        # a=0 (çift satır): ky=1 @ dy=0 | a=1 (tek satır): ky=2 @ dy=0, ky=0 @ dy=1
        taps = {0: [(0, 1)], 1: [(0, 2), (1, 0)]}
        for a_ in (0, 1):
            for b_ in (0, 1):
                for dy, ky in taps[a_]:
                    for dx, kx in taps[b_]:
                        K[:, a_, b_, :, dy, dx] = W[:, :, ky, kx].t()
        self.weight = nn.Parameter(K.reshape(cout * 4, cin, 2, 2), requires_grad=False)
        self.bias = nn.Parameter(ct.bias.detach().repeat_interleave(4), requires_grad=False)
    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1))
        return F.pixel_shuffle(F.conv2d(x, self.weight, self.bias), 2)

m = load_cyclegan("checkpoints/cyclegan/G_bold_final.pth")
ref = load_cyclegan("checkpoints/cyclegan/G_bold_final.pth")
seq = m.g.model
for i, layer in enumerate(seq):
    if isinstance(layer, nn.ConvTranspose2d):
        seq[i] = ConvTransposeAsConv(layer)
x = torch.rand(1, 3, 128, 96)
with torch.no_grad():
    print("ConvTranspose eşdeğerlik farkı:", float((m(x) - ref(x)).abs().max()))

class Clamp(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return torch.clamp(s.m(x), 0.0, 1.0)
mm = Clamp(m).eval()
em = litert_torch.convert(mm, (torch.rand(1, 3, 896, 672),))
em.export("monet_cyclegan_896x672.tflite")
x = torch.rand(1, 3, 896, 672)
with torch.no_grad(): t = mm(x).numpy()
print("tflite farkı:", float(np.abs(t - em(x)).max()))
