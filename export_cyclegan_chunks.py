"""CycleGAN'ı GPU'da ekranı takılmadan çalıştırmak için ~90 küçük parçaya bölerek TFLite'a çevirir.

Neden: GPU bir işlemi bitirmeden ekranı çizemiyor. Model tek parçayken ekran 6 sn donuyordu,
20 parçaya bölününce her parça ~0,3 sn sürüyordu (halka kesik kesik). Burada her katmanın çıkış
kanalları 4 gruba bölünüyor: InstanceNorm her kanalı kendi başına normalize ettiği için her grup
bağımsız hesaplanabiliyor ve sonuç birebir aynı kalıyor. Her parça ~0,05-0,07 sn sürüyor.

Tensörler K=4 parça hâlinde taşınır (her biri NHWC, kanalların dörtte biri). Parçalar (imzalar):
  e1            : resim                    -> a1 (896x672x64)
  e2_j  (j<2)   : a1                       -> a2'nin j. yarısı (448x336x64)
  e3_k  (k<4)   : a2_0, a2_1               -> h_k (224x168x64)
  r{i}a_k       : h_0..h_3                 -> y_k
  r{i}b_k       : y_0..y_3, h_k            -> yeni h_k
  d1_k          : h_0..h_3                 -> u_k (448x336x32)
  d2_k          : u_0..u_3                 -> v_k (896x672x16)
  fin           : v_0..v_3                 -> resim (0..1)
Katman değişiklikleri (GPU uyumu) export_cyclegan_gpu.py'deki ile aynı.
"""
import torch, numpy as np, litert_torch
from torch import nn
import torch.nn.functional as F
src = open("export_cyclegan_gpu.py", encoding="utf-8").read()
exec(src[:src.index("CKPT =")])           # ReflectPad, InstanceNorm, ConvTransposeAsConv, gpu_friendly
from cyclegan import load_cyclegan

K = 4
CKPT = "checkpoints/cyclegan/G_bold_final.pth"
full = load_cyclegan(CKPT).eval()
m = load_cyclegan(CKPT); gpu_friendly(m.g); m.eval()
L = list(m.g.model)


def conv_part(conv, sl):
    """Bir Conv2d'nin sadece sl aralığındaki çıkış kanallarını hesaplayan kopyası."""
    c = nn.Conv2d(conv.in_channels, sl.stop - sl.start, conv.kernel_size, conv.stride, conv.padding)
    c.weight.data = conv.weight.data[sl].clone(); c.bias.data = conv.bias.data[sl].clone()
    return c


def ct_part(ct, sl):
    """ConvTransposeAsConv'un sadece sl aralığındaki çıkış kanallarını hesaplayan kopyası."""
    C = ct.cout
    w = ct.weight.data.reshape(4, C, *ct.weight.shape[1:])[:, sl].reshape(-1, *ct.weight.shape[1:])
    b = ct.bias.data.reshape(4, C)[:, sl].reshape(-1)
    p = ConvTransposeAsConv.__new__(ConvTransposeAsConv); nn.Module.__init__(p)
    p.cout = sl.stop - sl.start
    p.weight = nn.Parameter(w.clone(), requires_grad=False); p.bias = nn.Parameter(b.clone(), requires_grad=False)
    return p


def chunks(n, k):
    s = n // k
    return [slice(i * s, (i + 1) * s) for i in range(k)]


class Cat(nn.Module):
    """Birden çok parçayı kanal boyunca birleştirip bir alt modüle verir."""
    def __init__(s, body, scale_in=False):
        super().__init__(); s.body = body; s.scale_in = scale_in
    def forward(s, *xs):
        x = torch.cat(xs, dim=1) if len(xs) > 1 else xs[0]
        return s.body(x * 2 - 1 if s.scale_in else x)


class ResB(nn.Module):
    def __init__(s, body):
        super().__init__(); s.body = body
    def forward(s, y0, y1, y2, y3, h):
        return h + s.body(torch.cat([y0, y1, y2, y3], dim=1))


class Fin(nn.Module):
    def __init__(s, body):
        super().__init__(); s.body = body
    def forward(s, *xs):
        return torch.clamp((s.body(torch.cat(xs, dim=1)) + 1) / 2, 0.0, 1.0)


def IN(n, ref):  # aynı eps ile yeni InstanceNorm
    return InstanceNorm(ref.eps)

parts = {}   # imza -> (modül, girdi sayısı)
pad0, c0, n0 = L[0], L[1], L[2]
parts["e1"] = (Cat(nn.Sequential(pad0, c0, n0, nn.ReLU()), scale_in=True), ["img"])
for j, sl in enumerate(chunks(128, 2)):
    parts[f"e2_{j}"] = (Cat(nn.Sequential(conv_part(L[4], sl), IN(0, L[5]), nn.ReLU())), ["e1"])
for k, sl in enumerate(chunks(256, K)):
    parts[f"e3_{k}"] = (Cat(nn.Sequential(conv_part(L[7], sl), IN(0, L[8]), nn.ReLU())), ["e2_0", "e2_1"])
hn = [f"e3_{k}" for k in range(K)]
for i in range(9):
    cb = list(L[10 + i].conv_block)   # pad, conv, IN, relu, pad, conv, IN
    yn = []
    for k, sl in enumerate(chunks(256, K)):
        parts[f"r{i}a_{k}"] = (Cat(nn.Sequential(cb[0], conv_part(cb[1], sl), IN(0, cb[2]), nn.ReLU())), hn)
        yn.append(f"r{i}a_{k}")
    new_h = []
    for k, sl in enumerate(chunks(256, K)):
        parts[f"r{i}b_{k}"] = (ResB(nn.Sequential(cb[4], conv_part(cb[5], sl), IN(0, cb[6]))), yn + [hn[k]])
        new_h.append(f"r{i}b_{k}")
    hn = new_h
for k, sl in enumerate(chunks(128, K)):
    parts[f"d1_{k}"] = (Cat(nn.Sequential(ct_part(L[19], sl), IN(0, L[20]), nn.ReLU())), hn)
un = [f"d1_{k}" for k in range(K)]
for k, sl in enumerate(chunks(64, K)):
    parts[f"d2_{k}"] = (Cat(nn.Sequential(ct_part(L[22], sl), IN(0, L[23]), nn.ReLU())), un)
vn = [f"d2_{k}" for k in range(K)]
parts["fin"] = (Fin(nn.Sequential(L[25], L[26], L[27])), vn)

H, W = 896, 672
shapes = {"img": (1, 3, H, W)}

def run_torch(x):
    t = {"img": x}
    with torch.no_grad():
        for name, (mod, ins) in parts.items():
            t[name] = mod(*[t[n] for n in ins])
    return t

if __name__ == "__main__":
    x = torch.rand(1, 3, 128, 96)
    with torch.no_grad():
        ref = full(x).clamp(0, 1)
    print("PyTorch parçalı fark:", float((run_torch(x)["fin"] - ref).abs().max()), flush=True)

    # örnek girdiler: her biri ayrı tensör (aynı tensörü iki kez vermek girdileri birleştiriyor)
    # tensör boyutlarını küçük bir girdiyle bul, tam boyuta ölçekle (bellek tasarrufu)
    small = run_torch(torch.rand(1, 3, 128, 96))
    def full_shape(n):
        b, c, h, w = small[n].shape
        return (1, h * H // 128, w * W // 96, c)
    conv = None
    for name, (mod, ins) in parts.items():
        sample = tuple(torch.rand(full_shape(n)) for n in ins)
        mod_cl = litert_torch.to_channel_last_io(mod.eval(), args=list(range(len(ins))), outputs=[0])
        conv = litert_torch.signature(name, mod_cl, sample) if conv is None else conv.signature(name, mod_cl, sample)
    em = conv.convert()
    em.export("monet_cyclegan_chunk_896x672.tflite")

    # Doğrulama: parçalar sırayla TFLite'ta
    x = torch.rand(1, 3, H, W)
    with torch.no_grad():
        ref = full(x).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    tt = {"img": x.permute(0, 2, 3, 1).numpy()}
    last_use = {}
    for idx, (name, (mod, ins)) in enumerate(parts.items()):
        for n in ins: last_use[n] = idx
    for idx, (name, (mod, ins)) in enumerate(parts.items()):
        tt[name] = em(*[tt[n] for n in ins], signature_name=name)
        for n in ins:
            if last_use[n] == idx: del tt[n]     # artık gerekmeyen ara sonuçları bırak
    print("parçalı tflite farkı:", float(np.abs(tt["fin"] - ref).max()))
    # Kotlin için plan: imza ve girdileri
    with open("plan.txt", "w") as f:
        for name, (mod, ins) in parts.items():
            f.write(name + ":" + ",".join(ins) + "\n")
    print("imza sayısı:", len(parts))
