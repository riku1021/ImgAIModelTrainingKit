import os

# 1. 統合ファイルのパス
combined_file_path = 'output.py'

# 2. 統合ファイルを読み込み
with open(combined_file_path, 'r', encoding='utf-8') as combined_file:
    combined_contents = combined_file.readlines()

# 3. 分割ポイントの識別
split_files = {}
current_file = None

for line in combined_contents:
    if line.startswith("# ") and line.endswith(".py\n"):
        # ファイル名をコメントから取得
        current_file = line[2:].strip()
        split_files[current_file] = []
    elif current_file:
        split_files[current_file].append(line)

# 4. 分割されたファイルの保存
output_folder_path = 'split_files'
os.makedirs(output_folder_path, exist_ok=True)

for file_name, contents in split_files.items():
    # 各ファイルの最初の要素が空行の場合、削除
    if contents and contents[0] == '\n':
        contents = contents[1:]
    output_file_path = os.path.join(output_folder_path, file_name)
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("".join(contents))

print(f"ファイルは {output_folder_path} に分割されて保存されました。")
