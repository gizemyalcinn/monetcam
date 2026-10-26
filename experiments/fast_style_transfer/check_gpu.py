import torch
#kurulu pytorch sürümü
print("PyTorch Sürümü:", torch.__version__)
#GPU görülüyor mu?
print("CUDA kullanılabilir mi:", torch.cuda.is_available() )

if torch.cuda.is_available():
    print("GPU:",torch.cuda.get_device_name(0))

    device = torch.device("cuda")
    a = torch.rand(1000, 1000).to(device)
    b = torch.rand(1000, 1000).to(device)
    c = a @ b
    print("Sonucun cihazı:", c.device)
else:
    print("GPU görülmüyor, kurulumda bir sorun var.")
