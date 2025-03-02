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
