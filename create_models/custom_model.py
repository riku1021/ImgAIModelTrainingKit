import torch.nn as nn

# カーネルサイズ＆畳み込み層＆全結合層を指定
kernel_size = 3
# フィルタ数 / 層数 / パディング / プーリング
layer_configs = [
    (64, 1, 1, False),
    (64, 1, 1, True),
    (128, 1, 1, False),
    (128, 1, 1, True),
    (256, 1, 1, False),
    (256, 1, 1, True),
    (512, 1, 1, False),
    (512, 1, 1, True),
    # (1024, 1, 1, False),
    # (1024, 1, 1, True),
    # (2048, 2, 1, False),
    # (2048, 1, 1, True),
]
fc_layer_units = []

# 残差ブロックを定義するクラス
class ResidualBlock(nn.Module):
    def __init__(self, residual, skip):
        super(ResidualBlock, self).__init__()
        self.residual = residual
        self.skip = skip

    def forward(self, x):
        return self.residual(x) + self.skip(x)

# 残差ブロックを追加する関数
def add_residual_block(layers, in_channels, out_channels, kernel_size, padding, num_layers):
    for _ in range(num_layers):
        residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, out_channels),
            nn.PReLU(),
        )

        if in_channels != out_channels:
            skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            skip = nn.Identity()

        layers.append(ResidualBlock(residual, skip))
        in_channels = out_channels

    return out_channels

# 全結合層を追加する関数
def add_fc_layers(layers, fc_layer_units, current_in_channels):
    for out_channels in fc_layer_units:
        layers.append(nn.Linear(current_in_channels, out_channels))
        layers.append(nn.PReLU())
        current_in_channels = out_channels
    return current_in_channels

# モデル構造を定義する関数
def custom_model(device, num_classes, current_in_channels):
    layers = []
    
    # 畳み込み層の追加
    for out_channels, num_layers, padding, use_pooling in layer_configs:
        current_in_channels = add_residual_block(layers, current_in_channels, out_channels, kernel_size, padding, num_layers)
        if use_pooling:
            layers.append(nn.MaxPool2d(2, 2))
        layers.append(nn.Dropout(0.1))
    
    # 全結合層の追加
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())
    current_in_channels = add_fc_layers(layers, fc_layer_units, current_in_channels)
    
    # 最後の層を追加
    layers.append(nn.Linear(current_in_channels, num_classes))
    
    # モデルの作成
    model = nn.Sequential(*layers).to(device)

    return model
