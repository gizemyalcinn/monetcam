"""CycleGAN eğitimi: fotoğraf <-> Monet.

İki üretici (generator) ve iki ayırt edici (discriminator) birlikte eğitilir:
  G  : fotoğraf -> Monet   (telefona gidecek olan bu)
  F  : Monet -> fotoğraf   (sadece eğitimde, "döngü tutarlılığı" için)
  D_M: "bu gerçek bir Monet mi?"
  D_P: "bu gerçek bir fotoğraf mı?"

Kayıplar:
  GAN kaybı     : G'nin ürettikleri D_M'yi kandırabilmeli -> Monet gibi görünmeli
  Döngü kaybı   : F(G(foto)) ≈ foto -> içerik (pelikan, ağaç...) korunmalı
  Kimlik kaybı  : G(Monet) ≈ Monet -> renkleri gereksiz yere değiştirmesin

G, CycleGAN yazarlarının hazır style_monet modelinden başlar. Hazır model "nazik"
sonuç verdiği için döngü ve kimlik ağırlıklarını düşürüyoruz: model fotoğrafa daha az
sadık kalıp daha cesur boyar. G ilk adımlarda dondurulur; bu sırada F ve D'ler
sıfırdan ısınır, böylece rastgele başlayan F, hazır G'yi bozamaz.
"""
import os, random, time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from cyclegan import ResnetGenerator
from utils import load_image, save_image

ON_KAGGLE = Path("/kaggle/input").exists()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_NAME = "cyclegan_monet_bold"
PRETRAINED_G = "checkpoints/cyclegan/style_monet.pth"
TEST_IMAGES = ["images/pelicans.jpeg", "images/ormanya.jpeg", "images/deniz.png"]
IMAGE_SIZE = 256
TOTAL_STEPS = 20000
WARMUP_STEPS = 1500        # bu kadar adım G donuk kalır
LR_G = 5e-5                # hazır G'yi yavaşça değiştir
LR_OTHER = 2e-4            # F ve D'ler sıfırdan öğreniyor
LAMBDA_CYCLE = 5.0         # orijinal: 10 -> daha cesur dönüşüm
LAMBDA_IDENTITY = 0.5      # orijinal: 5 -> renkleri daha özgür bırak
LOG_EVERY = 100
SAMPLE_EVERY = 1000
SAVE_G_EVERY = 2000
SAVE_EVERY = 500


# ---------------------------------------------------------------- veri
def find_dirs():
    """Monet ve fotoğraf klasörlerini bulur. monet2photo setinde trainA=Monet, trainB=fotoğraf;
    Kaggle yarışma setinde monet_jpg / photo_jpg. Her iki düzen de desteklenir."""
    roots = ["/kaggle/input"] if ON_KAGGLE else ["data"]
    found = {}
    for root in roots:
        for r, dirs, files in os.walk(root):
            for d in dirs:
                found.setdefault(d, os.path.join(r, d))
            dirs[:] = [d for d in dirs if d not in ("val2017", "test2017", "train2017", "annotations")]
    if "trainA" in found and "trainB" in found:
        return found["trainA"], found["trainB"]
    if "monet_jpg" in found and "photo_jpg" in found:
        return found["monet_jpg"], found["photo_jpg"]
    raise SystemExit(f"Monet/fotoğraf klasörleri bulunamadı. Bulunanlar: {sorted(found)[:30]}")


class Folder(Dataset):
    def __init__(self, folder):
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self.t = transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.12)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),   # CycleGAN -1..1 aralığında çalışır
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.t(Image.open(self.paths[i]).convert("RGB"))


def forever(loader):
    while True:
        for b in loader:
            yield b


# ---------------------------------------------------------------- ayırt edici
class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN: görüntünün tamamına tek bir not vermek yerine, her küçük
    parçaya ayrı ayrı "gerçek mi sahte mi" notu verir. Doku/fırça dokusunu yakalamakta iyidir."""
    def __init__(self, ndf=64):
        super().__init__()
        def block(i, o, s, norm=True):
            layers = [nn.Conv2d(i, o, 4, s, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers
        self.model = nn.Sequential(
            *block(3, ndf, 2, norm=False),
            *block(ndf, ndf * 2, 2),
            *block(ndf * 2, ndf * 4, 2),
            *block(ndf * 4, ndf * 8, 1),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


class ImagePool:
    """Ayırt ediciye sadece son üretilen görüntüyü değil, geçmişten rastgele görüntüleri de
    gösterir. Eğitimi kararlı hâle getiren standart bir CycleGAN hilesi."""
    def __init__(self, size=50):
        self.size, self.items = size, []

    def query(self, x):
        x = x.detach()
        if len(self.items) < self.size:
            self.items.append(x)
            return x
        if random.random() < 0.5:
            i = random.randrange(self.size)
            old, self.items[i] = self.items[i], x
            return old
        return x


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def set_grad(nets, flag):
    for n in nets:
        for p in n.parameters():
            p.requires_grad_(flag)


def save_ckpt(path, step, nets, opts):
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step,
                "nets": {k: v.state_dict() for k, v in nets.items()},
                "opts": {k: v.state_dict() for k, v in opts.items()}}, tmp)
    tmp.replace(path)


def find_previous_checkpoint():
    if not ON_KAGGLE:
        return None
    for r, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if not d.endswith("2017") and d not in ("trainA", "trainB", "testA", "testB", "monet_jpg", "photo_jpg")]
        if "latest.pth" in files and r.replace("\\", "/").endswith(f"runs/{RUN_NAME}"):
            return Path(r) / "latest.pth"
    return None


# ---------------------------------------------------------------- eğitim
def main():
    run = Path("runs") / RUN_NAME
    (run / "samples").mkdir(parents=True, exist_ok=True)
    ckpt = run / "latest.pth"

    monet_dir, photo_dir = find_dirs()
    monet, photo = Folder(monet_dir), Folder(photo_dir)
    print(f"Monet: {len(monet)} tablo ({monet_dir})")
    print(f"Fotoğraf: {len(photo)} fotoğraf ({photo_dir})")
    nw = 2 if ON_KAGGLE else 0
    it_m = forever(DataLoader(monet, batch_size=1, shuffle=True, num_workers=nw, drop_last=True))
    it_p = forever(DataLoader(photo, batch_size=1, shuffle=True, num_workers=nw, drop_last=True))

    G, F = ResnetGenerator().to(device), ResnetGenerator().to(device)
    D_M, D_P = PatchDiscriminator().to(device), PatchDiscriminator().to(device)
    for n in (F, D_M, D_P):
        n.apply(init_weights)
    nets = {"G": G, "F": F, "D_M": D_M, "D_P": D_P}

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_F = torch.optim.Adam(F.parameters(), lr=LR_OTHER, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(list(D_M.parameters()) + list(D_P.parameters()), lr=LR_OTHER, betas=(0.5, 0.999))
    opts = {"G": opt_G, "F": opt_F, "D": opt_D}

    step = 0
    if not ckpt.exists():
        prev = find_previous_checkpoint()
        if prev is not None:
            ckpt.write_bytes(prev.read_bytes())
            print(f"Önceki çalıştırmanın checkpoint'i bulundu: {prev}")
    if ckpt.exists():
        c = torch.load(ckpt, map_location=device)
        for k, n in nets.items():
            n.load_state_dict(c["nets"][k])
        for k, o in opts.items():
            o.load_state_dict(c["opts"][k])
        step = c["step"]
        print(f"Kaldığı yerden devam ediliyor: adım {step}")
    else:
        state = torch.load(PRETRAINED_G, map_location=device)
        own = G.state_dict()
        G.load_state_dict({k: v for k, v in state.items() if k in own})
        print(f"Hazır G yüklendi: {PRETRAINED_G}")

    tests = [p for p in TEST_IMAGES if Path(p).exists()]
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    pool_M, pool_P = ImagePool(), ImagePool()
    real_label = lambda t: torch.ones_like(t)
    fake_label = lambda t: torch.zeros_like(t)

    t0, s0 = time.time(), step
    while step < TOTAL_STEPS:
        real_P = next(it_p).to(device)   # fotoğraf
        real_M = next(it_m).to(device)   # Monet
        warm = step < WARMUP_STEPS

        # --- üreticiler
        set_grad([D_M, D_P], False)
        set_grad([G], not warm)
        fake_M = G(real_P)
        fake_P = F(real_M)
        pm, pp = D_M(fake_M), D_P(fake_P)
        loss_gan = mse(pm, real_label(pm)) + mse(pp, real_label(pp))
        loss_cyc = l1(F(fake_M), real_P) + l1(G(fake_P), real_M)
        loss_id = l1(G(real_M), real_M) + l1(F(real_P), real_P)
        loss_GF = loss_gan + LAMBDA_CYCLE * loss_cyc + LAMBDA_IDENTITY * loss_id
        opt_G.zero_grad(); opt_F.zero_grad()
        loss_GF.backward()
        if not warm:
            opt_G.step()
        opt_F.step()

        # --- ayırt ediciler
        set_grad([D_M, D_P], True)
        fm, fp = pool_M.query(fake_M), pool_P.query(fake_P)
        rm, fmo, rp, fpo = D_M(real_M), D_M(fm), D_P(real_P), D_P(fp)
        loss_D = 0.5 * (mse(rm, real_label(rm)) + mse(fmo, fake_label(fmo))) \
               + 0.5 * (mse(rp, real_label(rp)) + mse(fpo, fake_label(fpo)))
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        step += 1
        if step % LOG_EVERY == 0:
            per = (time.time() - t0) / (step - s0)
            print(f"adım {step}/{TOTAL_STEPS}{' (ısınma)' if warm else ''} | GAN {loss_gan.item():.3f} | "
                  f"döngü {loss_cyc.item():.3f} | kimlik {loss_id.item():.3f} | D {loss_D.item():.3f} | "
                  f"{per:.2f} sn/adım | kalan ~{(TOTAL_STEPS - step) * per / 60:.0f} dk", flush=True)

        if step % SAMPLE_EVERY == 0:
            G.eval()
            with torch.no_grad():
                for p in tests:
                    x = load_image(p, max_size=512).to(device)
                    _, _, H, W = x.shape
                    x = x[:, :, :H // 4 * 4, :W // 4 * 4]
                    y = (G(x * 2 - 1) + 1) / 2
                    save_image(y, run / "samples" / f"{Path(p).stem}_{step:05d}.jpg")
            G.train()
        if step % SAVE_G_EVERY == 0:
            torch.save(G.state_dict(), run / f"G_{step:05d}.pth")
        if step % SAVE_EVERY == 0:
            save_ckpt(ckpt, step, nets, opts)

    save_ckpt(ckpt, step, nets, opts)
    torch.save(G.state_dict(), run / "G_final.pth")
    print("Eğitim bitti.")


if __name__ == "__main__":
    main()
