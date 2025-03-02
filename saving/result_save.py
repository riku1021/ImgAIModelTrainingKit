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
