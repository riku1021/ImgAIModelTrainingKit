import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights

def tuning_resnet101(device, num_classes):
    # 事前学習済みのResNet101モデルをロード
    model = models.resnet101(weights=ResNet101_Weights.DEFAULT)

    # 最終層を置換
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # モデルをデバイスに転送
    model = model.to(device)
    
    return model
