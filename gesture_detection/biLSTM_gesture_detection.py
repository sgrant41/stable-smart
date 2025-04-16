import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ---------- 1. Dataset Class ----------
class IMUDataset(Dataset):
    def __init__(self, data, labels, sequence_length=50):
        self.sequence_length = sequence_length
        self.data, self.labels = self.create_sequences(data, labels)

    def create_sequences(self, data, labels):
        sequences = []
        seq_labels = []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            lbl = labels[i + self.sequence_length - 1]  # label for the end of the sequence
            sequences.append(seq)
            seq_labels.append(lbl)
        return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(seq_labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ---------- 2. BiLSTM Model ----------
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 2)  # binary classification

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

# ---------- 3. Load & Preprocess Data ----------
def load_imu_data(csv_path, sequence_length=50):
    df = pd.read_csv(csv_path)

    # Map annotations: 'marked' -> 1, 'unmarked' -> 0
    df["Annotation"] = df["Annotation"].map({"Unmarked": 0, "Annotated": 1})

    # Separate features and labels
    df_features = df.drop(columns=[df.columns[0], "Annotation"])
    X = df_features.values
    y = df["Annotation"].values

    print("Columns used for input features:", df_features.columns.tolist())

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    train_dataset = IMUDataset(X_train, y_train, sequence_length)
    test_dataset = IMUDataset(X_test, y_test, sequence_length)

    return train_dataset, test_dataset, X.shape[1]


# ---------- 4. Train Function ----------
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# ---------- 5. Evaluation ----------
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# ---------- 6. Main ----------
if __name__ == "__main__":
    csv_file = "/Users/madelinelombard/Documents/stable-smart/gesture_detection/training_data/pawing_imu_data_2025-04-15_21-45-23.csv" # <-- Your CSV path
    sequence_len = 100
    hidden_dim = 64
    batch_size = 64
    num_epochs = 15

    train_ds, test_ds, input_dim = load_imu_data(csv_file, sequence_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = BiLSTMModel(input_size=input_dim, hidden_size=hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader, criterion, optimizer, num_epochs)
    evaluate_model(model, test_loader)
