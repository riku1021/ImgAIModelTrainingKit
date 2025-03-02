import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageEnhance

# データ拡張を行う関数
def get_data_transforms(augmentation_params, img_size):
    pil_transform_list = []
    tensor_transform_list = []

    # 回転（ランダムに±30度回転）
    if augmentation_params.get('do_rotation', False):
        pil_transform_list.append(transforms.RandomRotation(degrees=30))

    # 平行移動（ランダムに上下左右に10%移動）
    if augmentation_params.get('do_translation', False):
        pil_transform_list.append(transforms.RandomAffine(0, translate=(0.1, 0.1)))

    # スケーリング（ランダムに0.8から1.2倍の範囲でサイズ変更し、再度クロップ）
    if augmentation_params.get('do_scaling', False):
        pil_transform_list.append(transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.2)))

    # 水平反転
    if augmentation_params.get('do_flipping', False):
        pil_transform_list.append(transforms.RandomHorizontalFlip())

    # 上下反転
    if augmentation_params.get('do_vertical_flipping', False):
        pil_transform_list.append(transforms.RandomVerticalFlip())

    # クロップ（ランダムにクロップし、指定サイズにリサイズ）
    if augmentation_params.get('do_cropping', False):
        pil_transform_list.append(transforms.RandomCrop(size=img_size, padding=4))

    # ズーム（ランダムに0.8から1.2倍の範囲でズーム）
    if augmentation_params.get('do_zooming', False):
        pil_transform_list.append(transforms.RandomAffine(0, scale=(0.8, 1.2)))

    # 明るさ調整
    if augmentation_params.get('do_brightness', False):
        pil_transform_list.append(transforms.ColorJitter(brightness=0.2))

    # コントラスト調整
    if augmentation_params.get('do_contrast', False):
        pil_transform_list.append(transforms.ColorJitter(contrast=0.2))

    # 彩度調整
    if augmentation_params.get('do_saturation', False):
        pil_transform_list.append(transforms.ColorJitter(saturation=0.2))

    # 色相変化
    if augmentation_params.get('do_hue', False):
        pil_transform_list.append(transforms.ColorJitter(hue=0.1))
    
    # シャープネス調整
    if augmentation_params.get('do_sharpness', False):
        pil_transform_list.append(AdjustSharpness(1.5))

    # ノイズ追加
    if augmentation_params.get('do_noise', False):
        tensor_transform_list.append(transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1))

    # カットアウト
    if augmentation_params.get('do_cutout', False):
        tensor_transform_list.append(transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3)))

    # リフレクションパディング
    if augmentation_params.get('do_reflection_padding', False):
        pil_transform_list.append(transforms.RandomApply([transforms.Pad(padding=4, padding_mode='reflect')]))

    # グレースケール
    if augmentation_params.get('do_grayscale', False):
        pil_transform_list.append(transforms.RandomGrayscale(p=1.0))

    return transforms.Compose(pil_transform_list + [
        transforms.ToTensor()
    ] + tensor_transform_list)

# シャープネス調整クラスの定義
class AdjustSharpness:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(self.factor)
        return img

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).cuda()

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
