# 画像分類モデル生成プログラム

## 概要

このプロジェクトは、PyTorch を用いた画像分類タスク向けの学習、評価、保存を行うサンプル実装です。

- **特徴:**  
  - カスタム CNN モデルの選択と初期化  
  - データ前処理と拡張（多彩なデータ拡張手法の利用）  
  - 学習プロセスの可視化・評価  
  - 最良モデルの ONNX 形式へのエクスポート  

## 特徴

- **柔軟なモデル選択:** `select_model` を用いて複数のモデルから選択可能  
- **多彩なデータ拡張:** 回転、平行移動、反転、ズーム、明るさ・コントラスト調整などをサポート  
- **学習プロセスの追跡:** EarlyStopping、学習率スケジューラ、トレーニング/評価結果の可視化  
- **ONNX 形式でのモデル保存:** 推論用途に向けた最適化済みモデル出力

## ディレクトリ構成

```plaintext
プロジェクトルート/
├── data/                   # データセット（例: soil データセット）
├── model_selection/        # モデル選択関連のコード
│   ├── model_selector.py
│   └── model_info.py
├── data_processing/        # データ準備・前処理関連のコード
│   └── data_utils.py
├── training/               # 学習処理関連のコード
│   └── train.py
├── saving/                 # モデルや結果の保存処理
│   ├── model_save.py
│   └── result_save.py
├── evaluation/             # 学習結果の評価と可視化
│   └── postprocess.py
├── main.py                 # エントリーポイント（上記コード）
└── README.md               # 本ドキュメント
```

## 依存関係

- Python 3.x
- PyTorch
- [その他必要なライブラリ] (例: numpy, matplotlib など)

依存関係は `requirements.txt` にまとめることを推奨します。

## インストール方法

1. リポジトリをクローンする

   ```bash
   git clone https://github.com/your_username/your_project.git
   cd your_project
   ```

2. 仮想環境の作成（任意）

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 依存パッケージのインストール

   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. **データセットの配置:**  
   `data/` フォルダに対象データセット（例: soil データセット）を配置してください。

2. **学習の実行:**  

   ```bash
   python main.py
   ```

   - 各種パラメータ（エポック数、画像サイズ、バッチサイズ、データ拡張設定など）は `main.py` 内で設定可能です。

3. **結果の確認:**  
   学習完了後、結果フォルダ（例: `result/`）に評価結果、グラフ、ONNX 形式のモデルが保存されます。

## プロジェクトの詳細

### データ前処理
- `prepare_data`: データフレーム作成、クラス数の算出  
- `setup_dataloaders`: 学習および検証用の DataLoader をセットアップ  
- `count_images_per_class`: クラス毎の画像数の集計と可視化

### モデル構築
- `select_model`: モデルの選択と初期化  
- `model_info`: モデル構造の表示

### 学習と評価
- `train_and_evaluate`: モデルの訓練、検証、EarlyStopping の適用  
- `plot_and_evaluate_results`: 学習過程の可視化および評価結果のプロット

### モデルの保存
- `save_model_as_onnx`: 学習済みモデルを ONNX 形式でエクスポート  
- `table_save`: データセット情報などのテーブル形式結果を保存

## カスタマイズ方法

- **パラメータ変更:**  
  各種パラメータ（エポック数、データ拡張、EarlyStopping 設定など）は `main.py` 内の該当セクションを編集することで変更できます。

- **モデルの追加:**  
  新たなモデルを `model_selection/` に追加し、`select_model` 関数内で条件分岐を追加することで対応可能です。
