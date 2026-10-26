import torch

def gram_matrix_slow(x):
    C, H, W = x.shape
    G = torch.zeros(C, C)
    for i in range(C):
        for j in range(C):
            G[i, j] = (x[i] * x[j]).mean()
    return G

def gram_matrix(x):
    C, H, W = x.shape
    F = x.view(C, H * W )
    G = F @ F.t()
    return G / (H * W)

x = torch.rand(3, 4, 4)
G_slow = gram_matrix_slow(x)
G_fast = gram_matrix(x)

print(G_fast)
print("Boyut:", G_fast.shape)
print("İki yöntem aynı mı:",torch.allclose(G_fast, G_slow))
print("Simetrik mi:", torch.allclose(G_fast, G_fast.t()))