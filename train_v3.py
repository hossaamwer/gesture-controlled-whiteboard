import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Configuration
DATA_FILE   = "gesture_data_v2.csv"
MODEL_FILE  = "gesture_v2.pth"
EPOCHS      = 30
BATCH_SIZE  = 8
LEARNING_RATE = 0.001

# AI Model Architecture
class HandDistConv(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [Batch, 1, 10 Features]
        # Layer 1: Conv -> BN -> ReLU -> Pool
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool  = nn.MaxPool1d(2) # Reduces dimension by half (10 -> 5)
        
        # Layer 2: Conv -> BN -> ReLU
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        
        # Fully Connected Layers
        # 64 channels * 5 input features = 320
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 3) 

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dim
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dataset Loader
class GestureDataset(Dataset):
    def __init__(self, csv_file):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Data file '{csv_file}' not found. Run record_v2.py first.")
        
        data = pd.read_csv(csv_file)
        self.labels = torch.tensor(data['label'].values, dtype=torch.long)
        self.features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{device}] Starting training...")

    try:
        dataset = GestureDataset(DATA_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = HandDistConv().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Loaded {len(dataset)} samples. Training for {EPOCHS} epochs.")
    
    # Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (torch.argmax(out, 1) == y).sum().item()
            
        # Progress Report
        if (epoch + 1) % 10 == 0:
            acc = 100 * correct / len(dataset)
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Accuracy: {acc:.1f}%")

    # Save Model
    torch.save(model.state_dict(), MODEL_FILE)
    print(f" Model saved to '{MODEL_FILE}'")

if __name__ == "__main__":
    train()