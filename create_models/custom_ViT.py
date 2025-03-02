import torch.nn as nn

# パラメータ設定
patch_size = 16
embed_dim = 384
num_layers = 32
num_heads = 64

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        # x: [seq_length, batch_size, embed_dim]
        x2 = self.norm1(x)
        x2, _ = self.attn(x2, x2, x2)
        x = x + x2
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class FlattenAndTranspose(nn.Module):
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).transpose(1, 2)
        # x: [batch_size, num_patches, embed_dim]
        return x

class ExtractClsToken(nn.Module):
    def forward(self, x):
        # x: [seq_length, batch_size, embed_dim]
        return x[:, 0]

def custom_ViT(device, num_classes, current_in_channels):
    layers = []

    # パッチエンベッドを追加
    layers.append(nn.Conv2d(current_in_channels, embed_dim, kernel_size=patch_size, stride=patch_size))
    
    # フラット化と転置
    layers.append(FlattenAndTranspose())
    
    # トランスフォーマーブロックを追加
    for _ in range(num_layers):
        layers.append(TransformerEncoderLayer(embed_dim, num_heads))
    
    # 最後の LayerNorm を追加
    layers.append(nn.LayerNorm(embed_dim))
    
    # [CLS]トークンを利用するカスタムレイヤー
    layers.append(ExtractClsToken())
    
    # 最後の全結合層を追加
    layers.append(nn.Linear(embed_dim, num_classes))
    
    # モデルの作成
    model = nn.Sequential(*layers).to(device)

    return model

# import torch.nn as nn

# # パラメータ設定
# patch_size = 16
# embed_dim = 384
# num_layers = 12
# num_heads = 8

# def custom_ViT(device, num_classes, current_in_channels):
#     layers = []

#     # パッチエンベッドを追加
#     layers.append(nn.Conv2d(current_in_channels, embed_dim, kernel_size=patch_size, stride=patch_size))
#     current_in_channels = embed_dim
    
#     # フラット化
#     layers.append(nn.Flatten(2))  # [batch_size, embed_dim, num_patches]
#     layers.append(nn.Identity())

#     # トランスフォーマーブロックを追加
#     for _ in range(num_layers):
#         # 形状を変換して LayerNorm を適用
#         layers.append(nn.LayerNorm(embed_dim))
        
#         # MultiheadAttention の適用
#         layers.append(nn.Linear(embed_dim, embed_dim))  # この Linear は形状変換
#         layers.append(nn.Identity())
#         layers.append(nn.MultiheadAttention(embed_dim, num_heads=num_heads))
#         layers.append(nn.Identity())
#         layers.append(nn.Linear(embed_dim, embed_dim))
        
#         # 後続処理
#         layers.append(nn.LayerNorm(embed_dim))
#         layers.append(nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.GELU(),
#             nn.Linear(embed_dim * 4, embed_dim)
#         ))
#         layers.append(nn.Identity())

#     # 最後の LayerNorm を追加
#     layers.append(nn.LayerNorm(embed_dim))

#     # 最後の全結合層を追加
#     layers.append(nn.Flatten())  # [batch_size, embed_dim * num_patches]
#     layers.append(nn.Linear(embed_dim * (current_in_channels // patch_size) ** 2, num_classes))
    
#     # モデルの作成
#     model = nn.Sequential(*layers).to(device)

#     return model
