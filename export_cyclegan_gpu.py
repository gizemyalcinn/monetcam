"""CycleGAN'ı telefon GPU'sunda çalışacak şekilde tflite'a çevirir.
reflection pad, instance norm ve conv transpose GPU'nun desteklediği işlemlerle yeniden yazıldı.
girdi/çıktı NHWC."""
import torch, numpy as np, litert_torch
import torch.nn.functional as F
from torch import nn
from cyclegan import load_cyclegan


class ReflectPad(nn.Module):
    """ReflectionPad2d ile aynı, sadece slice + concat."""
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
    """InstanceNorm2d (affine yok). fp16'da taşmasın diye kanalı önce max değerine bölüyor."""
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
    """ConvTranspose2d(k3, s2, p1, op1) ile aynı. 4 alt çekirdekli conv, sonra parçalar yerine dizilir."""
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
        out = torch.cat(rows, dim=2).reshape(N, 2 * H, 2 * W, C)         
        return out.permute(0, 3, 1, 2)


CONV_SCALE = 1.0 / 16   # norm öncesi convlar küçültülüyor, fp16 taşmasın


def gpu_friendly(module):
    """GPU'da sorun çıkaran katmanları eşdeğerleriyle değiştirir.
    norm öncesi convlar CONV_SCALE ile çarpılıyor, IN ölçekten bağımsız olduğu için sonuç değişmiyor."""
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
