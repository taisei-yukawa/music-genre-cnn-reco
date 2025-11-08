import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

# ---------------------------
# 乱数シード固定
# ---------------------------
random_seed = 77  # 42 -> 17 -> 43 -> 29 -> 52 -> 53 -> 54 -> 55 -> 56 -> 57 -> 58 -> 59 -> 60 -> 61 -> 
# 62 -> 63 -> 64 -> 65 -> 66 -> 67 -> 68 -> 69 -> 70 -> 71 -> 72 -> 73 -> 74 -> 75 -> 76 -> 77

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# ---------------------------
# パス・ラベル
# ---------------------------
dataset_folder = r"C:\Yukawa\Lab\Dataset\9genre_music100"

genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'rock', "pop"]

# ---------------------------
# CQT 読み込み & クリップ抽出
# ---------------------------
def load_cqt_features(file_path):
    cqt = np.load(os.path.join(file_path, 'cqt.npy'))
    return cqt  # (freq, time)

def extract_clip_cqt(cqt, sr=22050, hop_length=512, clip_duration=5, start_frame=None):
    """
    cqt: (F, T)
    5秒相当(デフォルト)のフレームを抽出。全長が短い場合はそのまま返す（評価では batch=1 なので可）。
    """
    frames_per_second = sr / hop_length  # ~43.07
    total_frames = cqt.shape[1]
    frames_clip = int(round(frames_per_second * clip_duration))

    if total_frames > frames_clip:
        if start_frame is None:
            start_frame = np.random.randint(0, total_frames - frames_clip + 1)
        end_frame = start_frame + frames_clip
        return cqt[:, start_frame:end_frame]
    else:
        return cqt  # 短い場合はそのまま

# ---------------------------
# データセット
# ---------------------------
class MusicGenreDataset(Dataset):
    """
    train=True  : ランダム5秒切り出しを返す（バッチ学習のため固定長）
    train=False : フル長CQTを返す（検証/テストでセグメント抽出や注意プーリングを上位で実施）
    """
    def __init__(self, dataset_folder, genres, sr=22050, hop_length=512, train=True):
        self.dataset_folder = dataset_folder
        self.genres = genres
        self.sr = sr
        self.hop_length = hop_length
        self.train = train
        self.track_paths = []
        self.labels = []

        for genre in genres:
            genre_folder = os.path.join(dataset_folder, genre)
            if os.path.exists(genre_folder):
                for track_id in os.listdir(genre_folder):
                    track_folder = os.path.join(genre_folder, track_id)
                    if os.path.isdir(track_folder):
                        self.track_paths.append(track_folder)
                        self.labels.append(genres.index(genre))

    def __len__(self):
        return len(self.track_paths)

    def __getitem__(self, idx):
        track_folder = self.track_paths[idx]
        cqt = load_cqt_features(track_folder)

        if self.train:
            cqt_clip = extract_clip_cqt(cqt, sr=self.sr, hop_length=self.hop_length, clip_duration=5)
            x = np.expand_dims(cqt_clip, axis=0)  # (1, F, Tclip)
        else:
            # 評価時は “フル長” を返す（上位でセグメント抽出やAttentionで集約）
            x = np.expand_dims(cqt, axis=0)  # (1, F, Tfull)

        label = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ---------------------------
# デバイス & ハイパラ
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 1e-4
dropout_rate = 0.2
batch_size = 16
num_epochs = 200
early_stopping_patience = 200

# ---------------------------
# データローダ
# ---------------------------
full_dataset = MusicGenreDataset(dataset_folder, genres, train=True)  # 長さ取得用
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(random_seed)

# 注意：train/test でクラス挙動を変えるため、別インスタンスを作る
train_dataset_all = MusicGenreDataset(dataset_folder, genres, train=True)
test_dataset_all  = MusicGenreDataset(dataset_folder, genres, train=False)

indices = torch.randperm(len(full_dataset), generator=generator).tolist()
train_indices = indices[:train_size]
test_indices  = indices[train_size:]

# Subset を手軽に作る
from torch.utils.data import Subset
train_dataset = Subset(train_dataset_all, train_indices)
test_dataset  = Subset(test_dataset_all,  test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)

# ---------------------------
# Attention Pooling
# ---------------------------
class TemporalAttentionPooling(nn.Module):
    """
    入力: x (B, C, T)  -> 重み α_t を学習して 時間重み付き和 (B, C) を返す
    """
    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.W = nn.Linear(in_channels, hidden, bias=True)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):  # x: (B, C, T)
        # (B, T, C)
        x_t = x.transpose(1, 2)
        H = torch.tanh(self.W(x_t))      # (B, T, hidden)
        e = self.v(H).squeeze(-1)        # (B, T)
        a = torch.softmax(e, dim=1)      # (B, T)
        # 重み付き和: (B, 1, T) @ (B, T, C) -> (B, 1, C) -> (B, C)
        context = torch.bmm(a.unsqueeze(1), x_t).squeeze(1)
        return context, a  # （aは可視化用に返す）

# ---------------------------
# 改良CNN（Attention付き）
# ---------------------------
class ImprovedMusicGenreCNN(nn.Module):
    def __init__(self, filter_sizes, num_classes):
        super(ImprovedMusicGenreCNN, self).__init__()
        # Conv は BN 直前なので bias=False、端情報保持のため padding 付与
        self.conv1 = nn.Conv2d(1, filter_sizes[0], kernel_size=(3, 9), stride=(1, 2), padding=(1, 4), bias=False)
        self.bn1   = nn.BatchNorm2d(filter_sizes[0])

        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=(3, 9), stride=(1, 2), padding=(1, 4), bias=False)
        self.bn2   = nn.BatchNorm2d(filter_sizes[1])

        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=(3, 9), stride=(1, 2), padding=(1, 4), bias=False)
        self.bn3   = nn.BatchNorm2d(filter_sizes[2])

        self.conv4 = nn.Conv2d(filter_sizes[2], filter_sizes[3], kernel_size=(3, 9), stride=(1, 2), padding=(1, 4), bias=False)
        self.bn4   = nn.BatchNorm2d(filter_sizes[3])

        self.conv5 = nn.Conv2d(filter_sizes[3], filter_sizes[4], kernel_size=(3, 9), stride=(1, 2), padding=(1, 4), bias=False)
        self.bn5   = nn.BatchNorm2d(filter_sizes[4]) #終わったら戻す

        # self.conv1 = nn.Conv2d(1, filter_sizes[0], kernel_size=(3, 8), stride=(1, 2), padding=(1, 4), bias=False)
        # self.bn1   = nn.BatchNorm2d(filter_sizes[0])

        # self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=(3, 8), stride=(1, 2), padding=(1, 4), bias=False)
        # self.bn2   = nn.BatchNorm2d(filter_sizes[1])

        # self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=(3, 8), stride=(1, 2), padding=(1, 4), bias=False)
        # self.bn3   = nn.BatchNorm2d(filter_sizes[2])

        # self.conv4 = nn.Conv2d(filter_sizes[2], filter_sizes[3], kernel_size=(3, 8), stride=(1, 2), padding=(1, 4), bias=False)
        # self.bn4   = nn.BatchNorm2d(filter_sizes[3])

        # self.conv5 = nn.Conv2d(filter_sizes[3], filter_sizes[4], kernel_size=(3, 8), stride=(1, 2), padding=(1, 4), bias=False)
        # self.bn5   = nn.BatchNorm2d(filter_sizes[4])
        

        self.gelu  = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        # Attention Pooling（周波数平均 → (B, C, T) → Attentionで (B, C)）
        self.attn_pool = TemporalAttentionPooling(in_channels=filter_sizes[4], hidden=128)

        # FC 前後にも正規化を追加（安定化）
        self.fc1    = nn.Linear(filter_sizes[4], 256, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2    = nn.Linear(256, 64, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc3    = nn.Linear(64, num_classes)  

        # デバッグ用（最後のアテンションを保持）
        self.last_attention = None

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.gelu(self.bn2(self.conv2(x)))
        x = self.gelu(self.bn3(self.conv3(x)))
        x = self.gelu(self.bn4(self.conv4(x)))
        x = self.gelu(self.bn5(self.conv5(x)))  # -> (B, C, F', T')

        # 周波数方向平均（F' を潰し、時間T'は保持）
        x = x.mean(dim=2)  # (B, C, T')

        # 時間方向 Attention Pooling -> (B, C)
        x, a = self.attn_pool(x)
        self.last_attention = a  # （必要なら参照）

        # FC
        x = self.gelu(self.fc1(x))
        x = self.bn_fc1(x)
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.bn_fc2(x)
        x = self.fc3(x)  # (B, num_classes)
        return x

# ---------------------------
# 補助: セグメント平均logitsで推論
# ---------------------------
def logits_from_segments(model, full_cqt_np, num_segments=5, sr=22050, hop_length=512, clip_duration=5):
    """
    full_cqt_np: (F, Tfull) の numpy
    num_segments 回だけ 5秒クリップをサンプルし、logits を平均して返す
    """
    seg_logits = []
    total_frames = full_cqt_np.shape[1]
    frames_per_second = sr / hop_length
    frames_clip = int(round(frames_per_second * clip_duration))

    for _ in range(num_segments):
        if total_frames > frames_clip:
            start_frame = np.random.randint(0, total_frames - frames_clip + 1)
            cqt_clip = full_cqt_np[:, start_frame:start_frame+frames_clip]
        else:
            cqt_clip = full_cqt_np  # 短ければそのまま

        cqt_t = torch.tensor(cqt_clip, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,F,Tc)
        with torch.no_grad():
            logits = model(cqt_t)  # (1, num_classes)
        seg_logits.append(logits)

    logits_mean = torch.stack(seg_logits, dim=0).mean(dim=0)  # (1, num_classes)
    return logits_mean

# ---------------------------
# 学習 & 評価
# ---------------------------
def train_and_evaluate(model_class, filter_sizes):
    model_name = f"filters_{'_'.join(map(str, filter_sizes))}"
    model = model_class(filter_sizes, num_classes=len(genres)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )

    # 保存ディレクトリ
    base_dir = r"C:\Yukawa\Lab\CNN\GeLU_Attention_result_seed59_earlystop77"
    train_csv_dir     = os.path.join(base_dir, "Train_csvData")
    loss_curve_dir    = os.path.join(base_dir, "Loss_Curve")
    learning_rate_dir = os.path.join(base_dir, "Learning_Rate")
    heat_map_dir      = os.path.join(base_dir, "heat_map")
    model_save_dir    = os.path.join(base_dir, "Models")
    os.makedirs(train_csv_dir, exist_ok=True)
    os.makedirs(loss_curve_dir, exist_ok=True)
    os.makedirs(learning_rate_dir, exist_ok=True)
    os.makedirs(heat_map_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    train_losses = []
    valid_losses = []
    training_results = []
    learning_rates = []
    min_valid_loss = float('inf')
    early_stopping_counter = 0

    # ---- エポックループ ----
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: (B, 1, F, Tclip)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # (B, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- 検証（セグメント平均logitsで損失/予測）----
        model.eval()
        valid_loss = 0.0
        all_valid_preds = []
        all_valid_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                # test_loader は (B=1, 1, F, Tfull) の “フル長”
                labels = labels.to(device)
                full_cqt = inputs[0].squeeze(0).cpu().numpy()  # (F, Tfull)

                logits_mean = logits_from_segments(
                    model, full_cqt, num_segments=5, sr=22050, hop_length=512, clip_duration=5
                )  # (1, num_classes)

                valid_loss += criterion(logits_mean, labels).item()
                final_pred = logits_mean.argmax(dim=1).item()

                all_valid_preds.append(final_pred)
                all_valid_labels.append(labels.item())

        avg_valid_loss = valid_loss / len(test_loader)
        valid_losses.append(avg_valid_loss)
        valid_accuracy = accuracy_score(all_valid_labels, all_valid_preds) * 100
        valid_f1 = f1_score(all_valid_labels, all_valid_preds, average='weighted', zero_division=0) * 100
        valid_precision = precision_score(all_valid_labels, all_valid_preds, average='weighted', zero_division=0) * 100
        valid_recall = recall_score(all_valid_labels, all_valid_preds, average='weighted', zero_division=0) * 100

        # ベストモデル保存（valid_loss ベース）
        if avg_valid_loss < min_valid_loss:
            min_valid_loss = avg_valid_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'{model_name}_best_model.pth'))
        else:
            early_stopping_counter += 1

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, '
              f'Valid Loss: {avg_valid_loss:.4f},  '
              f'min_valid_loss: {min_valid_loss:.4f},  '
              f'Accuracy: {valid_accuracy:.2f}%, '
              f'Precision: {valid_precision:.2f}%, '
              f'Recall: {valid_recall:.2f}%, '
              f'F1 Score: {valid_f1:.2f}%', flush=True)

        scheduler.step(avg_valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        training_results.append([epoch + 1, avg_train_loss, avg_valid_loss, min_valid_loss,
                                 valid_accuracy, valid_precision, valid_recall, valid_f1])

        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # ---- テスト評価（ベストで）----
    model.load_state_dict(torch.load(os.path.join(model_save_dir, f'{model_name}_best_model.pth')))
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.to(device)
            full_cqt = inputs[0].squeeze(0).cpu().numpy()  # (F, Tfull)

            logits_mean = logits_from_segments(
                model, full_cqt, num_segments=10, sr=22050, hop_length=512, clip_duration=5
            )  # (1, num_classes)

            pred = logits_mean.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(labels.item())
            total += 1
            if pred == labels.item():
                correct += 1

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    recall    = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    f1        = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100

    print(f'{model_name} - Accuracy: {accuracy:.2f}%')
    print(f'{model_name} - Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%')

    # ---- 混同行列 ----
    cm = confusion_matrix(all_labels, all_preds)
    class_names = genres

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}%')
    heat_map_dir = os.path.join(base_dir, "heat_map")
    plt.savefig(os.path.join(heat_map_dir, f'{model_name}_ConfusionMatrix.png'))
    plt.close()

    # ---- レポート ----
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)

    # ---- 学習率推移 ----
    plt.figure()
    plt.plot(range(1, len(learning_rates) + 1), learning_rates)
    plt.title(f'Learning Rate Over Epochs - Accuracy: {accuracy:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    learning_rate_dir = os.path.join(base_dir, "Learning_Rate")
    plt.savefig(os.path.join(learning_rate_dir, f'{model_name}_LearningRate.png'))
    plt.close()

    # ---- 損失曲線 ----
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss - {model_name} - Accuracy: {accuracy:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_curve_dir = os.path.join(base_dir, "Loss_Curve")
    plt.savefig(os.path.join(loss_curve_dir, f'{model_name}_LossCurve.png'))
    plt.close()

    # ---- CSV 保存 ----
    df = pd.DataFrame(training_results, columns=[
        'epoch', 'train_loss', 'valid_loss', 'min_valid_loss', 'accuracy', 'precision', 'recall', 'f1_score'
    ])
    train_csv_dir = os.path.join(base_dir, "Train_csvData")
    df.to_csv(os.path.join(train_csv_dir, f'{model_name}_Train_Data.csv'), index=False, quoting=csv.QUOTE_NONE)

# ---------------------------
# 試すフィルタサイズ
# ---------------------------
filter_sizes_list = [
    # [32, 64, 128, 256, 512],
    [64, 72, 144, 288, 576],
    # [32, 79, 158, 316, 632]
]

print("Filter Sizes to Train:", filter_sizes_list)

# ---------------------------
# 実行
# ---------------------------
for filter_sizes in filter_sizes_list:
    print(f"\nTraining with filter sizes: {filter_sizes}")
    train_and_evaluate(ImprovedMusicGenreCNN, filter_sizes)
