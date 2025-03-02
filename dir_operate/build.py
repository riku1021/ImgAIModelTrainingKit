import os

def get_specific_python_files(folder_path, exclude_files, target_dirs):
    """
    指定されたフォルダ内の特定のディレクトリのPythonファイルを収集し、除外リストに含まれていないものを返す。
    """
    py_files = []
    for root, dirs, files in os.walk(folder_path):
        # 指定されたディレクトリ内かルートディレクトリの場合のみファイルを収集
        if os.path.basename(root) in target_dirs or root == folder_path:
            for file in files:
                if file.endswith('.py') and file not in exclude_files:
                    py_files.append(os.path.join(root, file))
    return py_files

def combine_python_files(py_files, output_file_path):
    """
    指定されたPythonファイルを結合し、出力ファイルに保存する。
    """
    combined_contents = []
    for py_file in py_files:
        with open(py_file, 'r', encoding='utf-8') as file:
            relative_path = os.path.relpath(py_file, os.path.dirname(output_file_path))
            combined_contents.append(f"# {relative_path}\n")
            combined_contents.append(file.read())
            combined_contents.append("\n\n")

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(combined_contents))

def main():
    # このファイルがいる場所の親のディレクトリのパスを取得
    folder_path = os.path.dirname(os.path.dirname(__file__))
    
    # 除外するファイルのリスト
    exclude_files = [os.path.basename(__file__)]
    
    # 対象とするディレクトリのリスト
    target_dirs = ['data_processing', 'model_selection', 'training', 'evaluation', 'saving']

    # 特定のディレクトリ内のPythonファイルを取得（除外ファイルを除く）
    py_files = get_specific_python_files(folder_path, exclude_files, target_dirs)
    
    # 結合した内容を一つのファイルに保存
    output_file_path = os.path.join(folder_path, 'output.py')
    
    # 既存のoutput.pyの内容を削除
    if os.path.exists(output_file_path):
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write('')

    # 結合した内容を書き込み
    combine_python_files(py_files, output_file_path)
    
    print(f"結合されたファイルは {output_file_path} に保存されました！")

if __name__ == "__main__":
    main()
