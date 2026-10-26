"""CycleGAN üreticisi (generator).

CycleGAN makalesinin resmi kodundaki ResnetGenerator'ın aynısı; böylece yazarların
yayımladığı hazır ağırlıklar (ör. style_monet.pth) doğrudan yüklenebilir.

Fast Style Transfer modelimizle (model.py) benzer bir iskelet:
  encoder (küçült) -> 9 residual blok -> decoder (büyüt)
Farkları: daha geniş (kanal sayısı 64 -> 256), 5 yerine 9 blok,
büyütmede transpoze evrişim ve çıkışta tanh (değerler -1..1 aralığında).
"""
import torch
from torch import nn


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, ngf=64, n_blocks=9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        for i in range(2):                                   # 2 kez küçült
            m = 2 ** i
            layers += [nn.Conv2d(ngf * m, ngf * m * 2, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(ngf * m * 2), nn.ReLU(True)]
        layers += [ResnetBlock(ngf * 4) for _ in range(n_blocks)]
        for i in range(2):                                   # 2 kez büyüt
            m = 2 ** (2 - i)
            layers += [nn.ConvTranspose2d(ngf * m, ngf * m // 2, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(ngf * m // 2), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MonetCycleGAN(nn.Module):
    """Girdi ve çıktıyı bizim uygulamanın beklediği 0..1 aralığına çeviren sarmalayıcı.
    CycleGAN -1..1 aralığında çalışır; bu sayede Android tarafında hiçbir şey değişmez."""
    def __init__(self, generator):
        super().__init__()
        self.g = generator

    def forward(self, x):
        y = self.g(x * 2 - 1)
        return (y + 1) / 2


def load_cyclegan(path, device="cpu"):
    g = ResnetGenerator()
    state = torch.load(path, map_location=device)
    # Eski PyTorch sürümleriyle kaydedilmiş dosyalarda InstanceNorm için fazladan
    # (kullanılmayan) kayıtlar bulunabiliyor; modelde olmayan anahtarları atıyoruz.
    own = g.state_dict()
    state = {k: v for k, v in state.items() if k in own}
    missing = [k for k in own if k not in state]
    if missing:
        raise RuntimeError(f"Eksik ağırlıklar: {missing[:5]}")
    g.load_state_dict(state)
    return MonetCycleGAN(g).to(device).eval()


if __name__ == "__main__":
    g = ResnetGenerator()
    n = sum(p.numel() for p in g.parameters())
    print(f"Parametre sayısı: {n:,}  (~{n*4/1e6:.0f} MB)")
    print("Örnek anahtarlar:", list(g.state_dict())[:4])
