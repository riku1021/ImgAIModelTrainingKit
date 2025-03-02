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
    dataset = "soil"
    model_name = "custom_CNN"
    ai_server = False
    use_fp16 = True
    base_dir = f"../data/{dataset}"
    result_dir = "result"

    # パラメータ設定
    num_epochs = 50
    img_size = 128
    batch_size = 10
    test_size = 0.8
    in_channels = 3

    # EarlyStoppingの設定
    early_stopping_config = {
        'early_stopping': False,     # EarlyStoppingの使用指定
        'min_epochs': 50,           # EarlyStoppingの判定開始エポック数
        'patience': 20,             # 連続増加許容回数
    }

    # データ拡張の制御用辞書
    augmentation_params = {
        'do_rotation': True,            # 回転
        'do_translation': True,         # 平行移動
        # 'do_scaling': True,             # スケーリング
        'do_flipping': True,            # 水平反転
        'do_vertical_flipping': True,   # 上下反転
        # 'do_cropping': True,            # クロップ
        'do_zooming': True,             # ズーム
        'do_brightness': True,          # 明るさ調整
        'do_contrast': True,            # コントラスト調整
        'do_saturation': True,          # 彩度調整
        'do_hue': True,                 # 色相変化
        'do_sharpness': True,           # シャープネス調整
        'do_noise': True,               # ノイズ追加
        'do_cutout': True,              # カットアウト
        'do_reflection_padding': True,  # リフレクションパディング
        'do_grayscale': True,           # グレースケール
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
