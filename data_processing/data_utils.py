import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from data_processing.augmentations import get_data_transforms

# クラスごとにラベル付けを行う関数
def get_labels(base_dir):
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# クラスごとのファイル数を算出する関数
def count_images_per_class(base_dir):
    class_counts = {}
    labels = get_labels(base_dir)

    for label in labels:
        image_files = os.listdir(os.path.join(base_dir, label))
        class_counts[label] = len(image_files)

    total_images = sum(class_counts.values())
    return class_counts, total_images

# データセットの詳細を表示する関数
def display_class_counts(class_counts, total_images):
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    df.loc['Total'] = ['Total', total_images]
    return df

# 平均値＆標準偏差を算出する関数
def calculate_mean_std(base_dir, img_size):
    all_images = []
    labels = get_labels(base_dir)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    for label in labels:
        image_files = os.listdir(os.path.join(base_dir, label))
        for image_file in image_files:
            image_path = os.path.join(base_dir, label, image_file)
            image = transform(Image.open(image_path).convert("RGB"))
            all_images.append(image)

    all_images = torch.stack(all_images)
    mean = all_images.mean(dim=[0, 2, 3])
    std = all_images.std(dim=[0, 2, 3])

    return mean, std

# データセットを準備する関数
def prepare_data(base_dir):
    labels = get_labels(base_dir)
    df = pd.DataFrame(
        [(os.path.join(base_dir, label, img), idx) for idx, label in enumerate(labels) for img in os.listdir(os.path.join(base_dir, label))],
        columns=["images", "labels"]
    )
    return df, len(labels)

# データローダーを設定する関数
def setup_dataloaders(base_dir, df, img_size, batch_size, test_size, augmentation_params):
    mean, std = calculate_mean_std(base_dir, img_size=(img_size, img_size))

    common_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=mean, std=std)
    ]

    # データ拡張の適用
    train_transform = transforms.Compose([
        get_data_transforms(augmentation_params, img_size)
    ] + common_transforms)

    val_transform = transforms.Compose([transforms.ToTensor()] + common_transforms)

    train, val = train_test_split(df.values, test_size=test_size, random_state=42)

    train_loader = DataLoader(PipeDS(train, train_transform), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(PipeDS(val, val_transform), batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader

# データセットの前処理を行うクラス
class PipeDS(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = Image.open(img).convert("RGB")
        return self.transform(img), label
