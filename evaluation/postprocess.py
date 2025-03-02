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
