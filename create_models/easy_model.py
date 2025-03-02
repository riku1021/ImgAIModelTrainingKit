import torch
import torch.nn as nn
from torch.optim import RAdam

# モデルを初期化する関数
def easy_model(device, num_classes, in_channels):
    kernel_size = 3
    padding = 1 
    model = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size, padding=padding),
        nn.BatchNorm2d(64),
        nn.PReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.2),
        
        nn.Conv2d(64, 128, kernel_size, padding=padding),
        nn.BatchNorm2d(128),
        nn.PReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.2),
        
        nn.Conv2d(128, 256, kernel_size, padding=padding),
        nn.BatchNorm2d(256),
        nn.PReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.2),
        
        nn.Conv2d(256, 512, kernel_size, padding=padding),
        nn.BatchNorm2d(512),
        nn.PReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.2),

        nn.Conv2d(512, 1024, kernel_size, padding=padding),
        nn.BatchNorm2d(512),
        nn.PReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.2),
        
        nn.AdaptiveAvgPool2d((1, 1)),
        
        nn.Flatten(),
        
        nn.Linear(512, 1024),
        nn.PReLU(),
        nn.Linear(1024, 512),
        nn.PReLU(),
        nn.Linear(512, num_classes)
    ).to(device)
    
    return model
