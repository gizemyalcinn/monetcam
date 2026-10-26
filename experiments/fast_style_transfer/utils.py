import torch
from PIL import Image, ImageOps
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(path, max_size=512):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    scale = max_size / max(image.size)
    new_size = (round(image.height * scale), round(image.width * scale))

    transform = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)

def save_image(tensor, path):
    image = tensor.detach().squeeze(0).clamp(0, 1).cpu()
    transforms.ToPILImage()(image).save(path)