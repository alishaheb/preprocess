import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------

# Load the dataset
df = pd.read_csv('azazazaz_second_test_Hp_modified.csv')

# Drop identifier/high-cardinality columns to avoid memory issues
cols_to_drop = ["Patient_ID", "Unnamed: 18", "Unnamed: 19", "Unnamed: 20", "Unnamed: 21",
                "Treated_with_drugs"]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Define target
target = 'Survived_1_year'

# Separate features and target
X = df.drop(target, axis=1)
y = df[target]

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Build preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Fit and transform data
X_processed = preprocessor.fit_transform(X)
print("Processed data shape:", X_processed.shape)

# Split into training and testing sets
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_np.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# PyTorch LSTM Model
# -----------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, fc_dim=16):
        super(LSTMClassifier, self).__init__()
        # Since we use one time step, input shape will be (batch, 1, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def forward(self, x):
        # x shape: (batch, 1, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use the last hidden state
        x = torch.relu(self.fc1(hn[-1]))
        x = self.fc2(x)
        return x  # raw logits


# Prepare data for LSTM: add a time-step dimension (1)
X_train_lstm = X_train.unsqueeze(1)  # shape becomes (batch, 1, input_dim)
X_test_lstm = X_test.unsqueeze(1)

# Initialize LSTM model
input_dim = X_train.shape[1]
lstm_model = LSTMClassifier(input_dim=input_dim).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)

# Training loop for LSTM
epochs = 100
lstm_model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in DataLoader(TensorDataset(X_train_lstm, y_train), batch_size=batch_size, shuffle=True):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    print(f"LSTM Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataset):.4f}")

# Evaluation for LSTM
lstm_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in DataLoader(TensorDataset(X_test_lstm, y_test), batch_size=batch_size):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = lstm_model(batch_X)
        preds = torch.sigmoid(outputs) >= 0.5
        correct += (preds.float() == batch_y).sum().item()
        total += batch_y.size(0)
lstm_acc = correct / total
print("LSTM Test Accuracy:", lstm_acc)


# -----------------------------
# PyTorch Transformer Model
# -----------------------------
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Set the LSTM model to evaluation mode and prepare to gather predictions.
lstm_model.eval()
all_preds = []
all_labels = []
all_probs = []

# Use the existing DataLoader for the LSTM test data.
for batch_X, batch_y in DataLoader(TensorDataset(X_test_lstm, y_test), batch_size=batch_size):
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    with torch.no_grad():
        outputs = lstm_model(batch_X)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(batch_y.cpu().numpy())
    all_probs.extend(probs.cpu().numpy())

# Flatten the arrays for metric computation.
all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()
all_probs = np.array(all_probs).flatten()

# Compute additional metrics.
conf_mat = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)

print("Additional LSTM Evaluation Metrics:")
print("Confusion Matrix:\n", conf_mat)
print("Classification Report:\n", class_report)
print("ROC AUC Score:", roc_auc)

