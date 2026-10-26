import torch
from torch import nn


def make_norm(kind, ch):
    """Normalizasyon katmanı.
    "batch":    BatchNorm. İstatistikler eğitim boyunca öğrenilir, kullanımda sabittir.
                Model tek bir noktayla istatistiği manipüle edemez -> "damla" lekeleri oluşmaz.
                Telefona dönüştürülürken evrişimin içine katlanır (daha hızlı).
    "instance": InstanceNorm. Her görüntüyü kendi içinde normalleştirir.
                Eski modeller (v1, v2) bununla eğitildi."""
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "instance":
        return nn.InstanceNorm2d(ch, affine=True)
    raise ValueError(f"Bilinmeyen normalizasyon: {kind}")


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)

    def forward(self, x):
        return self.conv(self.pad(x))


class ResidualBlock(nn.Module):
    def __init__(self, ch, norm):
        super().__init__()
        self.conv1 = ConvLayer(ch, ch, 3, 1)
        self.in1 = make_norm(norm, ch)
        self.conv2 = ConvLayer(ch, ch, 3, 1)
        self.in2 = make_norm(norm, ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + x


class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ConvLayer(in_ch, out_ch, kernel_size, 1)

    def forward(self, x):
        return self.conv(self.up(x))


class TransformerNet(nn.Module):
    def __init__(self, norm="batch"):
        super().__init__()
        self.norm = norm
        self.encoder = nn.Sequential(
            ConvLayer(3, 32, 9, 1),   make_norm(norm, 32),  nn.ReLU(),
            ConvLayer(32, 64, 3, 2),  make_norm(norm, 64),  nn.ReLU(),
            ConvLayer(64, 128, 3, 2), make_norm(norm, 128), nn.ReLU(),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(128, norm) for _ in range(5)])
        self.decoder = nn.Sequential(
            UpsampleConv(128, 64, 3), make_norm(norm, 64), nn.ReLU(),
            UpsampleConv(64, 32, 3),  make_norm(norm, 32), nn.ReLU(),
            ConvLayer(32, 3, 9, 1),
        )

    def forward(self, x):
        return self.decoder(self.residuals(self.encoder(x)))


def load_transformer(path, device="cpu"):
    """Checkpoint'i yükler. BatchNorm mu InstanceNorm mu olduğunu dosyadan kendisi anlar:
    BatchNorm modellerinde 'running_mean' kayıtları bulunur."""
    state = torch.load(path, map_location=device)
    norm = "batch" if any(k.endswith("running_mean") for k in state) else "instance"
    model = TransformerNet(norm=norm).to(device)
    model.load_state_dict(state)
    return model.eval()
