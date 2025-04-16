import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import os

# -------------------------------
# CONFIGURATION
# -------------------------------
csv_path = "horse_imu_data.csv"           # Labeled CSV
unlabeled_csv_path = "horse_unlabeled.csv"  # === NEW === Unlabeled CSV for prediction
label_column = "label"
window_size = 200
stride = 50
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
hidden_size = 128
num_layers = 2
val_split = 0.2  # === NEW ===

# -------------------------------
# 1. WINDOWING + ENCODING
# -------------------------------
def window_data(df, window_size=200, stride=50, label_col='label', require_label=True):
    if require_label:
        features = df.drop(columns=[label_col]).values
        labels = df[label_col].values
    else:
        features = df.values
        labels = None

    X, y = [], []
    for start in range(0, len(features) - window_size + 1, stride):
        end = start + window_size
        window = features[start:end]
        if require_label:
            window_labels = labels[start:end]
            label = Counter(window_labels).most_common(1)[0][0]
            y.append(label)
        X.append(window)

    return np.array(X), np.array(y) if require_label else None

def encode_labels(labels):
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    return y_encoded, le

df = pd.read_csv(csv_path)
X, y_raw = window_data(df, window_size, stride, label_column)
y, label_encoder = encode_labels(y_raw)
num_classes = len(label_encoder.classes_)

# -------------------------------
# 2. TORCH DATASET
# -------------------------------
class IMUDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

dataset = IMUDataset(X, y)
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])  # === NEW ===

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size)

# -------------------------------
# 3. MODEL
# -------------------------------
class HorseMotionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(HorseMotionBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = h_lstm[:, -1, :]
        return self.fc(out)

input_size = X.shape[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HorseMotionBiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# -------------------------------
# 4. TRAINING
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_dl:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch+1} | Train Loss: {running_loss/len(train_dl):.4f} | "
          f"Val Loss: {val_loss/len(val_dl):.4f} | Acc: {correct/total:.2%}")

# -------------------------------
# 5. INFERENCE ON UNMARKED DATA
# -------------------------------
def predict_unlabeled_csv(csv_path, window_size=200, stride=50):
    df_unlabeled = pd.read_csv(csv_path)
    X_unlabeled, _ = window_data(df_unlabeled, window_size, stride, label_col=None, require_label=False)
    ds = IMUDataset(X_unlabeled)
    dl = DataLoader(ds, batch_size=batch_size)

    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in dl:
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred_indices = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(pred_indices)

    class_names = label_encoder.inverse_transform(predictions)
    print("\n Predicted Labels:")
    for i, label in enumerate(class_names):
        print(f"Window {i}: {label}")

    # Optional: save to CSV
    output_df = pd.DataFrame({
        "window_id": range(len(class_names)),
        "predicted_label": class_names
    })
    output_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to 'predictions.csv'")

# === Run prediction ===
predict_unlabeled_csv(unlabeled_csv_path, window_size, stride)
