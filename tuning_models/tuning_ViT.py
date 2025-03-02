import timm
import torch.nn as nn

def tuning_ViT(device, num_classes):
    # ViTモデルの作成
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    # model = timm.create_model('vit_huge_patch14_224.orig_in21k', pretrained=True, num_classes=num_classes)
    model.dropout = nn.Dropout(p=0.5)

    # モデルをデバイスに転送
    model.to(device)
    
    return model
