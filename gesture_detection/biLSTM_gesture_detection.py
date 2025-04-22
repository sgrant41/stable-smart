import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. Dataset Class ----------
class IMUDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.sequence_length = sequence_length
        self.X_seq = []
        self.y_seq = []

        # Slide window over data to create sequences
        for i in range(len(X) - sequence_length):
            self.X_seq.append(X[i:i+sequence_length])
            self.y_seq.append(y[i+sequence_length-1])  # label for the last frame in sequence

        self.X_seq = np.array(self.X_seq)
        self.y_seq = np.array(self.y_seq)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return torch.tensor(self.X_seq[idx], dtype=torch.float32), torch.tensor(self.y_seq[idx], dtype=torch.long)

# ---------- 2. BiLSTM Model ----------
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # num_classes = 3

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ---------- 3. Load & Preprocess Data ----------
def load_imu_data(csv_path, sequence_length=50):
    df = pd.read_csv(csv_path)

    # Drop device name if it's still included
    if not np.issubdtype(df[df.columns[0]].dtype, np.number):
        df = df.drop(columns=[df.columns[0]])

    # Encode "Label" column: stand=0, walk=1, lay=2
    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"])

    # Separate features and labels
    X = df.drop(columns=["Label"]).values
    y = df["Label"].values

    # Normalize input features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    train_dataset = IMUDataset(X_train, y_train, sequence_length)
    test_dataset = IMUDataset(X_test, y_test, sequence_length)

    # Return also the number of input features and class names
    return train_dataset, test_dataset, X.shape[1], label_encoder.classes_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        model.eval()


# ---------- 5. Evaluation ----------
def evaluate_model(model, test_loader, class_names):
    model.eval()
    correct = 0
    total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()   

# ---------- 6. Main ----------
if __name__ == "__main__":
    csv_file = "/Users/madelinelombard/Documents/stable-smart/gesture_detection/training_data/3-18-sheldon.csv" # <-- Your CSV path
    sequence_len = 100
    hidden_dim = 64
    batch_size = 64
    num_epochs = 15

    train_ds, test_ds, input_dim, class_names = load_imu_data(csv_file, sequence_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = BiLSTMModel(input_dim, hidden_size=128, num_layers=2, num_classes=3)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader, criterion, optimizer, num_epochs)
    class_names = ["Lay", "Stand", "Walk"]
    evaluate_model(model, test_loader, class_names)