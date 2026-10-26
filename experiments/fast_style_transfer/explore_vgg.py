import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(path, max_size = 512):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    scale = max_size / max(image.size)
    new_size = (round(image.height * scale), round(image.width * scale))

    transform = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

content = load_image("images/pelicans.jpeg")
style = load_image("images/waterlily.jpg")
print("İçerik:", tuple(content.shape))
print("Stil:", tuple(style.shape))

vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

for p in vgg.parameters():
    p.requires_grad_(False)

x = content
with torch.no_grad():
    for i, layer in enumerate(vgg):
        x = layer(x)
        if isinstance(layer, torch.nn.ReLU):
            print(f"Katman {i:2d}: {tuple(x.shape)}")
