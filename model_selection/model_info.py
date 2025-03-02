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
