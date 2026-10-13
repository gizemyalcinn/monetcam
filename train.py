import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from utils import device, load_image, save_image
from losses import (load_vgg, get_features, gram_matrix,
                    content_loss, style_loss, tv_loss, STYLE_LAYERS)
from model import TransformerNet

# Veri nerede? Ortama göre kendiliğinden bulunur:
#  - Kaggle: eklenen COCO veri seti /kaggle/input altında bir yerde durur
#  - Colab: /content/data (hücreyle indirilen)
#  - Bilgisayar: proje klasöründeki data/
ON_KAGGLE = Path("/kaggle/input").exists()


SKIP_DIRS = {"train2017", "val2017", "test2017", "annotations", "unlabeled2017"}


def walk_kaggle_input():
    """/kaggle/input altını gezer ama fotoğraf klasörlerinin İÇİNE girmez.
    (Her birinde on binlerce dosya var; içlerini taramak dakikalar sürer.)"""
    for root, dirs, files in os.walk("/kaggle/input"):
        yield root, dirs, files
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]


def find_data_dirs():
    if ON_KAGGLE:
        found = {}
        for root, dirs, _ in walk_kaggle_input():
            for d in dirs:
                if d in ("train2017", "val2017", "test2017") and d not in found:
                    found[d] = os.path.join(root, d)
        return [found[n] for n in ("train2017", "val2017", "test2017") if n in found]
    root = "/content/data" if Path("/content").exists() else "data"
    return [f"{root}/val2017", f"{root}/test2017"]


def find_previous_checkpoint(run_name):
    """Kaggle'da her çalıştırma temiz başlar. Önceki çalıştırmanın çıktısı
    girdi olarak eklendiyse, onun latest.pth dosyasını bulup oradan devam ederiz."""
    if not ON_KAGGLE:
        return None
    for root, _, files in walk_kaggle_input():
        if "latest.pth" in files and root.replace("\\", "/").endswith(f"runs/{run_name}"):
            return Path(root) / "latest.pth"
    return None


DATA_DIRS = find_data_dirs()
MAX_IMAGES = 60000   # Kaggle'da train2017 de bulunursa ~120 bin fotoğraf olur; bu kadarı yeterli
NUM_WORKERS = 4 if ON_KAGGLE else 2

RUN_NAME = "bridge_512_relu22_sw4e10_tv"
STYLE_IMAGE = "images/bridge.jpg"
TEST_IMAGE = "images/pelicans.jpeg"
INIT_WEIGHTS = None
IMAGE_SIZE = 512
BATCH_SIZE = 4
TOTAL_STEPS = 10000
LR = 5e-4
CONTENT_WEIGHT = 1e5
STYLE_WEIGHT = 4e10     # relu2_2 içeriği güçlendirdi; stili de dengelemek için artırıldı
TV_WEIGHT = 2e5         # pürüzsüzlük kaybı: yatay çizgili gren gibi ince desenlere karşı
LOG_EVERY = 50
SAMPLE_EVERY = 500
SAVE_EVERY = 500
USE_AMP = device.type == "cuda"   # karma hassasiyet: T4'te belirgin hızlanma
MAX_GRAD_NORM = 5.0              # ani sıçramaları (çöküşü) önlemek için gradyan kırpma


class ImageDataset(Dataset):
    def __init__(self, folders, size):
        self.paths = []
        for folder in folders:
            self.paths += sorted(Path(folder).glob("*.jpg"))
        self.paths = self.paths[:MAX_IMAGES]
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


def save_checkpoint(path, model, optimizer, step):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step}, tmp)
    tmp.replace(path)


def main():
    run_dir = Path("runs") / RUN_NAME
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "latest.pth"

    dataset = ImageDataset(DATA_DIRS, IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, drop_last=True)
    print("Veri klasörleri:", DATA_DIRS)
    print(f"{len(dataset)} görüntü, epoch başına {len(loader)} adım")
    if len(dataset) == 0:
        raise SystemExit("Hiç fotoğraf bulunamadı: COCO veri setinin eklendiğinden emin ol.")

    vgg = load_vgg(device)
    model = TransformerNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    step = 0
    if not ckpt_path.exists():
        prev = find_previous_checkpoint(RUN_NAME)
        if prev is not None:
            ckpt_path.write_bytes(prev.read_bytes())
            print(f"Önceki çalıştırmanın checkpoint'i bulundu: {prev}")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"Kaldığı yerden devam ediliyor: adım {step}")
    elif INIT_WEIGHTS:
        model.load_state_dict(torch.load(INIT_WEIGHTS, map_location=device))
        print(f"Başlangıç ağırlıkları yüklendi: {INIT_WEIGHTS}")

    style = load_image(STYLE_IMAGE)
    with torch.no_grad():
        style_features = get_features(style, vgg)
        style_grams = {name: gram_matrix(style_features[name]) for name in STYLE_LAYERS}

    test_image = load_image(TEST_IMAGE, max_size=896)

    model.train()
    start = time.time()
    start_step = step
    while step < TOTAL_STEPS:
        for batch in loader:
            batch = batch.to(device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                output = model(batch)
                with torch.no_grad():
                    content_features = get_features(batch, vgg)
                output_features = get_features(output, vgg)

            # Kayıplar float32'de: Gram değerleri çok küçük, float16'da sıfıra yuvarlanabilir
            output_features = {k: v.float() for k, v in output_features.items()}
            content_features = {k: v.float() for k, v in content_features.items()}
            c_loss = CONTENT_WEIGHT * content_loss(output_features, content_features)
            s_loss = STYLE_WEIGHT * style_loss(output_features, style_grams)
            t_loss = TV_WEIGHT * tv_loss(output.float())
            loss = c_loss + s_loss + t_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if step % LOG_EVERY == 0:
                elapsed = time.time() - start
                per_step = elapsed / (step - start_step)
                remaining = (TOTAL_STEPS - step) * per_step / 60
                print(f"adım {step}/{TOTAL_STEPS} | içerik {c_loss.item():.1f} | "
                      f"stil {s_loss.item():.1f} | tv {t_loss.item():.1f} | {per_step:.2f} sn/adım | "
                      f"kalan ~{remaining:.0f} dk")

            if step % SAMPLE_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    save_image(model(test_image), run_dir / "samples" / f"step_{step:05d}.jpg")
                model.train()
                torch.save(model.state_dict(), run_dir / f"model_{step:05d}.pth")

            if step % SAVE_EVERY == 0:
                save_checkpoint(ckpt_path, model, optimizer, step)

            if step >= TOTAL_STEPS:
                break

    save_checkpoint(ckpt_path, model, optimizer, step)
    torch.save(model.state_dict(), run_dir / "final.pth")
    print("Eğitim bitti.")


if __name__ == "__main__":
    main()