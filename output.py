# main.py

import torch
import torch.nn as nn
from torch.optim import RAdam
from torch.optim.lr_scheduler import StepLR
from model_selection.model_selector import select_model
from model_selection.model_info import model_info 
from data_processing.data_utils import prepare_data, setup_dataloaders, count_images_per_class, display_class_counts
from training.train import train_and_evaluate
from saving.model_save import save_model_as_onnx
from saving.result_save import table_save
from evaluation.postprocess import plot_and_evaluate_results

def main():
    # 基本設定
    dataset = "donut4"
    model_name = "custom_CNN"
    ai_server = False
    use_fp16 = True
    base_dir = f"../data/{dataset}"
    result_dir = "result"

    # パラメータ設定
    num_epochs = 300
    img_size = 512
    batch_size = 16
    test_size = 0.2
    in_channels = 3

    # EarlyStoppingの設定
    early_stopping_config = {
        'early_stopping': False,     # EarlyStoppingの使用指定
        'min_epochs': 30,           # EarlyStoppingの判定開始エポック数
        'patience': 10,             # 連続増加許容回数
    }

    # データ拡張の制御用辞書
    augmentation_params = {
        # 'do_rotation': True,            # 回転
        # 'do_translation': True,         # 平行移動
        # 'do_scaling': True,             # スケーリング
        # 'do_flipping': True,            # 水平反転
        # # 'do_vertical_flipping': True,   # 上下反転
        # 'do_cropping': True,            # クロップ
        # 'do_zooming': True,             # ズーム
        # 'do_brightness': True,          # 明るさ調整
        # 'do_contrast': True,            # コントラスト調整
        # 'do_saturation': True,          # 彩度調整
        # 'do_hue': True,                 # 色相変化
        # 'do_sharpness': True,           # シャープネス調整
        # 'do_noise': True,               # ノイズ追加
        # 'do_cutout': True,              # カットアウト
        # 'do_reflection_padding': True,  # リフレクションパディング
        # 'do_mixup': True,               # Mixup
        # 'do_cutmix': True,              # CutMix
    }

    # クラスごとのデータ数と合計データ数を取得
    class_counts, total_images = count_images_per_class(base_dir)
    count_df = display_class_counts(class_counts, total_images)
    table_save(count_df, "dataset_info.png", result_dir)
    if not ai_server:
        print(f"\n{count_df}\n")

    # データの準備
    df, num_classes = prepare_data(base_dir)
    class_names = list(class_counts.keys())
    train_loader, val_loader = setup_dataloaders(base_dir, df, img_size, batch_size, test_size, augmentation_params)

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not ai_server:
        print(f'Using device: {device}\n')

    # モデルの初期化＆定義
    model = select_model(model_name, device, num_classes, in_channels)
    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.parameters(), weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # モデル構造の表示
    if not ai_server:
        model_info(model, in_channels, img_size)

    # モデルの訓練＆評価
    train_acc, train_loss, val_acc, val_loss, best_model_state, actual_epochs = train_and_evaluate(
        model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs, ai_server,
        augmentation_params, early_stopping_config, use_fp16
    )

    # ベストモデルをロード
    model.load_state_dict(best_model_state)

    # ONNXファイルで保存
    onnx_file_path = "best_model.onnx"
    save_model_as_onnx(model, onnx_file_path, (in_channels, img_size, img_size), device, result_dir, ai_server)

    # 結果のプロットと評価
    plot_and_evaluate_results(model, train_acc, val_acc, train_loss, val_loss, val_loader, device, class_names, actual_epochs, result_dir, ai_server)

if __name__ == "__main__":
    main()




# output.py





# data_processing\augmentations.py

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





# data_processing\data_utils.py

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




# evaluation\plot.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import pandas as pd

# 正答率をプロットする関数
def plot_accuracy(train_accuracy, val_acc, num_epochs, ai_server):
    epochs = range(1, num_epochs + 1)

    fig = plt.figure(figsize=(7, 6))
    plt.plot(epochs, train_accuracy, marker='o', label='Train')
    plt.plot(epochs, val_acc, marker='o', label='Validation')
    plt.title(f'Accuracy over epochs    max acc:{max(val_acc):.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    if not ai_server:
        plt.show()

    return fig

# 誤差をプロットする関数
def plot_loss(train_loss, val_loss, num_epochs, ai_server):
    epochs = range(1, num_epochs + 1)

    fig = plt.figure(figsize=(7, 6))
    plt.plot(epochs, train_loss, marker='o', label='Train')
    plt.plot(epochs, val_loss, marker='o', label='Validation')
    plt.title(f'Loss over epochs    max loss:{min(val_loss):.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if not ai_server:
        plt.show()

    return fig

# 混同行列をプロットする関数
def plot_confusion_matrix(model, val_loader, device, class_names, ai_server):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if not ai_server:
        plt.show()
    
    return fig

# 評価指標をプロットする関数
def evaluate_result(model, val_loader, device, class_names, ai_server):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # クラスごとの正答率を算出
    cm = confusion_matrix(all_labels, all_preds)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # zero_divisionパラメータを追加してclassification_reportを作成
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()

    # サポートをint型に変換
    df['support'] = df['support'].astype(int)

    # 正解率の行を除外
    df = df.drop(['accuracy'], axis=0)

    # 再現率、適合率、F値の列をfloat64型に変換
    df[['precision', 'recall', 'f1-score']] = df[['precision', 'recall', 'f1-score']].astype(float)

    # 小数第4位まで表示
    for col in ['precision', 'recall', 'f1-score']:
        df[col] = df[col].map(lambda x: '{:.4f}'.format(x))
    
    # クラスごとの正答率を追加
    class_accuracy_df = pd.DataFrame(class_accuracy, index=class_names, columns=['accuracy'])
    class_accuracy_df['accuracy'] = class_accuracy_df['accuracy'].map(lambda x: '{:.4f}'.format(x))
    
    # マージ
    df = df.join(class_accuracy_df)

    if not ai_server:
        print(f"{df}\n")
    
    return df




# evaluation\postprocess.py

from evaluation.plot import plot_accuracy, plot_loss, plot_confusion_matrix, evaluate_result
from saving.result_save import table_save, save_figure

def plot_and_evaluate_results(model, train_acc, val_acc, train_loss, val_loss, val_loader, device, class_names, num_epochs, result_dir, ai_server):
    # 評価指標の表示
    evaluate_df = evaluate_result(model, val_loader, device, class_names, ai_server)
    table_save(evaluate_df, "evaluate_data.png", result_dir)

    # 正答率のプロット
    fig_acc = plot_accuracy(train_acc, val_acc, num_epochs, ai_server)
    save_figure(fig_acc, "accuracy_plot.png", result_dir)

    # 誤差のプロット
    fig_loss = plot_loss(train_loss, val_loss, num_epochs, ai_server)
    save_figure(fig_loss, "loss_plot.png", result_dir)

    # 混同行列のプロット
    fig_matrix = plot_confusion_matrix(model, val_loader, device, class_names, ai_server)
    save_figure(fig_matrix, "matrix_plot.png", result_dir)




# model_selection\labelSmoothingLoss.py

import torch.nn.functional as F
import torch.nn as nn
import torch


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_label = one_hot * confidence + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prob = F.log_softmax(pred, dim=1)
        loss = - (smooth_label * log_prob).sum(dim=1).mean()
        return loss




# model_selection\model_info.py

from torchinfo import summary

# モデル構造を表示する関数
def model_info(model, in_channels, img_size):
    # モデルの概要を取得
    summary_str = str(summary(model, input_size=(1, in_channels, img_size, img_size), verbose=0))

    # フィルタリングするキーワードのリスト
    keywords_to_filter = ['Identity', 'Sequential', 'GroupNorm', 'LayerNorm','BatchNorm2d', 'ReLU', 'PReLU', 'Dropout']

    # 概要を行ごとに分割し、不要な行をフィルタリング
    filtered_lines = [line for line in summary_str.split('\n') if not any(keyword in line for keyword in keywords_to_filter)]

    # フィルタリングされた概要を結合して表示
    filtered_summary = '\n'.join(filtered_lines)
    print(f"{filtered_summary}\n")

    return




# model_selection\model_selector.py

from create_models.custom_model import custom_model
from create_models.easy_model import easy_model
from create_models.custom_CNN import custom_CNN
from create_models.vgg16 import vgg16
from create_models.resnet101 import resnet101
from create_models.custom_ViT import custom_ViT
from tuning_models.tuning_resnet101 import tuning_resnet101
from tuning_models.tuning_ViT import tuning_ViT
from create_models.test_model import test_model
def select_model(model_name, device, num_classes, in_channels):

    # グローバルスコープからインポートされたモデル関数を取得
    all_model_functions = {name: func for name, func in globals().items() if callable(func)}

    # モデル名をリストに分類
    standard_model_names = []
    finetuned_model_names = []

    # チューニングモデルかを判定
    for name in all_model_functions:
        if name.startswith('tuning'):
            finetuned_model_names.append(name)
        else:
            standard_model_names.append(name)

    # モデルがない場合のエラー出力
    if model_name not in standard_model_names + finetuned_model_names:
        raise ValueError(f"Unknown model name: {model_name}")

    model_function = globals()[model_name]

    if model_name in standard_model_names:
        return model_function(device, num_classes, in_channels)
    elif model_name in finetuned_model_names:
        return model_function(device, num_classes)




# saving\model_save.py

import torch
import os

# モデルをONNX形式で保存
def save_model_as_onnx(model, file_path, input_size, device, result_dir, ai_server):
    model.eval()
    dummy_input = torch.randn(1, *input_size).to(device)
    output_path = os.path.join(result_dir, file_path)
    os.makedirs(result_dir, exist_ok=True)
    torch.onnx.export(model, dummy_input, output_path, export_params=True, opset_version=14,
                      do_constant_folding=True, input_names=['input'], output_names=['output'])
    if not ai_server:
        print(f"Model saved to {output_path}")




# saving\result_save.py

import os
import matplotlib.pyplot as plt

# ディレクトリを作成する関数
def create_directory(directory_path):
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path)

# テーブルを画像として保存する関数
def table_save(df, output_path, result_dir):
    create_directory(result_dir)

    num_rows, num_cols = df.shape
    fig_width = num_cols * 1.5 + 2
    fig_height = num_rows * 0.4 + 2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    
    # テーブルを作成
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center',
        cellLoc='center',
        bbox=[0.1, 0, 0.8, 1]
    )
    
    # セルの幅をカスタマイズ
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    col_widths = [0.25] * (num_cols - 1)
    col_widths.insert(0, 1.5)
    for i, width in enumerate(col_widths):
        table.auto_set_column_width(i)
        for key, cell in table.get_celld().items():
            if key[1] == i:
                cell.set_width(width)

    table.scale(1.2, 1.2)

    plt.savefig(os.path.join(result_dir, output_path))
    plt.close(fig)

# 画像を保存する関数
def save_figure(fig, output_path, result_dir):
    create_directory(result_dir)
    fig.savefig(os.path.join(result_dir, output_path))
    plt.close(fig)




# training\train.py

import numpy as np
import time
import torch
from datetime import datetime, timedelta
from torch.cuda.amp import autocast, GradScaler
from data_processing.augmentations import mixup_data, cutmix_data, mixup_criterion, cutmix_criterion

# 学習＆評価を行う関数
def train_and_evaluate(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs, ai_server, augmentation_params, early_stopping_config, use_fp16):
    train_acc, train_loss, val_acc, val_loss = [], [], [], []
    best_val_loss = 0.0
    best_model_state = None
    actual_epochs = 0

    # FP16モード用のGradScalerを初期化
    scaler = GradScaler() if use_fp16 else None

    # EarlyStoppingの初期化
    early_stopping = EarlyStopping(
        patience=early_stopping_config['patience'],
        min_epochs=early_stopping_config['min_epochs']
    ) if early_stopping_config == {} or early_stopping_config['early_stopping'] else None

    total_start_time = time.time()

    def calculate_metrics(loader, is_train=True):
        model.train() if is_train else model.eval()
        running_loss, correct, total = 0.0, 0, 0
        batch_counter = 0

        for data, target in loader:
            data, target = data.to(device), target.to(device)

            if is_train:
                optimizer.zero_grad()

                data, target_a, target_b, lam, augmentation_type = apply_augmentation(data, target, augmentation_params, batch_counter)
                batch_counter += 1

                with autocast(enabled=use_fp16):
                    loss, output = compute_loss(model, data, target_a, target_b, lam, augmentation_type, criterion)

                if use_fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                with torch.no_grad():
                    with autocast(enabled=use_fp16):
                        output = model(data)
                        loss = criterion(output, target)

            running_loss += loss.item()
            correct += (output.argmax(dim=1) == target).sum().item()
            total += target.size(0)

            del data, target, output, loss
            torch.cuda.empty_cache()

        return running_loss / len(loader), correct / total

    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()

        tr_loss, tr_acc = calculate_metrics(train_loader, is_train=True)
        vl_loss, vl_acc = calculate_metrics(val_loader, is_train=False)

        epoch_elapsed_time = time.time() - epoch_start_time
        if epoch == 1:
            estimated_total_time = epoch_elapsed_time * num_epochs
            if not ai_server:
                print(f"Estimated total training time: {format_time(estimated_total_time)}")
                print(f"Estimated end time: {format_end_time(estimated_total_time)}")

        train_acc.append(tr_acc)
        train_loss.append(tr_loss)
        val_acc.append(vl_acc)
        val_loss.append(vl_loss)
        actual_epochs = epoch

        if not ai_server:
            print(f'Epoch: [{epoch}/{num_epochs}], Train Loss: {tr_loss:.4f}, Train Accuracy: {tr_acc:.4f}, Val Loss: {vl_loss:.4f}, Val Accuracy: {vl_acc:.4f}')

        # EarlyStoppingの判定
        if early_stopping and early_stopping(epoch, vl_loss):
            if not ai_server:
                print("Early stopping !")
            break

        # ベストモデルを記憶
        if vl_loss > best_val_loss:
            best_val_loss = vl_loss
            best_model_state = model.state_dict()

        # # 学習率スケジューラをステップ
        # scheduler.step()

    total_elapsed_time = time.time() - total_start_time
    if not ai_server:
        print(f"\nTotal elapsed time: {format_time(total_elapsed_time)}")
        print(f'Average epoch accuracy: {np.mean(val_acc): .4f}')
        print(f'Best accuracy: {max(val_acc): .4f}\n')

    return train_acc, train_loss, val_acc, val_loss, best_model_state, actual_epochs

class EarlyStopping:
    def __init__(self, patience, min_epochs):
        self.patience = patience
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, epoch, val_loss):
        if epoch < self.min_epochs:
            return False

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

# 実行時間を予測＆表示する関数
def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes} minutes {remaining_seconds} seconds"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = int(seconds % 60)
        return f"{hours} hours {minutes} minutes {remaining_seconds} seconds"

# 終了時刻を予測＆表示する関数
def format_end_time(seconds):
    estimated_end_time = datetime.now() + timedelta(seconds=seconds)
    return estimated_end_time.strftime("%H:%M\n")

def apply_augmentation(data, target, augmentation_params, batch_counter):
    if augmentation_params.get('do_mixup', False) and augmentation_params.get('do_cutmix', False):
        if batch_counter % 2 == 0:
            data, target_a, target_b, lam = mixup_data(data, target)
            return data, target_a, target_b, lam, 'mixup'
        else:
            data, target_a, target_b, lam = cutmix_data(data, target)
            return data, target_a, target_b, lam, 'cutmix'
    elif augmentation_params.get('do_mixup', False):
        data, target_a, target_b, lam = mixup_data(data, target)
        return data, target_a, target_b, lam, 'mixup'
    elif augmentation_params.get('do_cutmix', False):
        data, target_a, target_b, lam = cutmix_data(data, target)
        return data, target_a, target_b, lam, 'cutmix'
    else:
        return data, target, target, 1.0, 'none'

def compute_loss(model, data, target_a, target_b, lam, augmentation_type, criterion):
    output = model(data)
    if augmentation_type == 'mixup':
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
    elif augmentation_type == 'cutmix':
        loss = cutmix_criterion(criterion, output, target_a, target_b, lam)
    else:
        loss = criterion(output, target_a)
    return loss, output



