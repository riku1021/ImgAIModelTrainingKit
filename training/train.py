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
