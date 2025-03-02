import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

def test_model(device, num_classes, current_in_channels):
    # モデルのパラメータ
    image_size = 224
    patch_size = 16
    dim = 768
    depth = 12
    n_heads = 64
    mlp_dim = 3072

    # モデルのインスタンス化
    model = ViT(image_size=image_size, patch_size=patch_size, n_classes=num_classes, dim=dim, 
                depth=depth, n_heads=n_heads, channels=current_in_channels, mlp_dim=mlp_dim)

    # モデルを指定されたデバイスに移動
    model = model.to(device)

    return model

class Patching(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=patch_size, pw=patch_size)

    def forward(self, x):
        return self.net(x)

class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        super().__init__()
        self.net = nn.Linear(patch_dim, dim)

    def forward(self, x):
        return self.net(x)

class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
    
    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        return x + self.pos_embedding

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim_heads = dim // n_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.split_into_heads = Rearrange("b n (h d) -> b h n d", h=n_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.concat = Rearrange("b h n d -> b n (h d)", h=n_heads)

    def forward(self, x):
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        q, k, v = self.split_into_heads(q), self.split_into_heads(k), self.split_into_heads(v)

        attention_weight = self.softmax(torch.matmul(q, k.transpose(-1, -2)) * (self.dim_heads ** -0.5))
        output = torch.matmul(attention_weight, v)
        return self.concat(output)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(dim, n_heads, mlp_dim) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLPHead(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, n_classes, dim, depth, n_heads, channels=3, mlp_dim=256):
        super().__init__()

        n_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patching = Patching(patch_size)
        self.projection = LinearProjection(patch_dim, dim)
        self.embedding = Embedding(dim, n_patches)
        self.encoder = TransformerEncoder(dim, n_heads, mlp_dim, depth)
        self.head = MLPHead(dim, n_classes)

    def forward(self, img):
        x = self.patching(img)
        x = self.projection(x)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.head(x[:, 0])
        return x
