import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
file_path = "breast_cancer_data.csv"
df = pd.read_csv(file_path)

# Define features and target
X = df.drop(columns=['diagnosis'])  # Features
y = df['diagnosis']  # Target variable

# Split into train (80%) and test (20%) sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Standardize features using training set statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --- LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 50, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the output of the last time step
        x = self.dropout(lstm_out[:, -1, :])
        x = self.fc(x)
        return self.sigmoid(x)


# Initialize model, loss, and optimizer
input_size = X_train.shape[1]
device = torch.device("cpu")
lstm_model = LSTMModel(input_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.0001)

# Variables to store history for plotting
epochs = 100
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# Training loop with validation
for epoch in range(epochs):
    # --- Training Phase ---
    lstm_model.train()
    train_losses = []
    train_preds_list = []
    train_labels_list = []

    for batch_X, batch_y in train_loader:
        # Reshape for LSTM: [batch_size, seq_len=1, input_size]
        batch_X = batch_X.view(batch_X.shape[0], 1, batch_X.shape[1]).to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        preds = (outputs > 0.5).float()
        train_preds_list.extend(preds.cpu().numpy())
        train_labels_list.extend(batch_y.cpu().numpy())

    avg_train_loss = np.mean(train_losses)
    train_accuracy = accuracy_score(train_labels_list, train_preds_list)
    train_loss_history.append(avg_train_loss)
    train_acc_history.append(train_accuracy)

    # --- Validation Phase ---
    lstm_model.eval()
    val_losses = []
    val_preds_list = []
    val_labels_list = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.view(batch_X.shape[0], 1, batch_X.shape[1]).to(device)
            batch_y = batch_y.to(device)
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            val_losses.append(loss.item())

            preds = (outputs > 0.5).float()
            val_preds_list.extend(preds.cpu().numpy())
            val_labels_list.extend(batch_y.cpu().numpy())

    avg_val_loss = np.mean(val_losses)
    val_accuracy = accuracy_score(val_labels_list, val_preds_list)
    val_loss_history.append(avg_val_loss)
    val_acc_history.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_accuracy:.4f}")

# --- Plotting Training and Validation Loss & Accuracy ---
plt.figure(figsize=(12, 5))

# Plot Loss Curve
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_loss_history, label='Train Loss')
plt.plot(range(1, epochs + 1), val_loss_history, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss on LSTM Model on Airline ")
plt.legend()

# Plot Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_acc_history, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_acc_history, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy on LSTM Model on Airline")
plt.legend()

plt.tight_layout()
plt.show()

# --- Evaluate on Test Set ---
lstm_model.eval()
all_test_preds = []
all_test_labels = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.view(batch_X.shape[0], 1, batch_X.shape[1]).to(device)
        outputs = lstm_model(batch_X)
        preds = (outputs > 0.5).float()
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(batch_y.cpu().numpy())

test_accuracy = accuracy_score(all_test_labels, all_test_preds)
print("Test Accuracy:", test_accuracy)
print(classification_report(all_test_labels, all_test_preds))
