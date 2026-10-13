import argparse
from pathlib import Path

import torch
from utils import device, load_image, save_image
from model import TransformerNet


def main():
    parser = argparse.ArgumentParser(description="Fotoğrafı Monet tarzına çevir")
    parser.add_argument("input", help="girdi fotoğrafı")
    parser.add_argument("output", help="çıktı dosyası")
    parser.add_argument("--checkpoint",
                        default="checkpoints/bridge_sw15e9_long_epoch_5.pth")
    parser.add_argument("--size", type=int, default=512,
                        help="uzun kenarın piksel boyutu")
    args = parser.parse_args()

    model = TransformerNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    x = load_image(args.input, max_size=args.size)
    _, _, H, W = x.shape
    x = x[:, :, :H // 4 * 4, :W // 4 * 4]

    with torch.no_grad():
        y = model(x)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_image(y, args.output)
    print(f"{args.input} → {args.output} ({x.shape[3]}×{x.shape[2]})")


if __name__ == "__main__":
    main()