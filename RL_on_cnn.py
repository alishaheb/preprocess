import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# Define the search space (number of layers and type of layers)
SEARCH_SPACE = {
    "num_layers": [1, 2, 3],  # Number of layers
    "layer_type": ["conv", "fc"]  # Convolutional or Fully Connected
}


# Define the RL Controller (LSTM-based policy network)
class NASController(nn.Module):
    def __init__(self, input_size=10, hidden_size=16, output_size=3):
        super(NASController, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get last timestep output
        return F.softmax(out, dim=-1)


# Function to sample an architecture
def sample_architecture(controller):
    input_tensor = torch.randn(1, 1, 10)  # Random input to LSTM
    probs = controller(input_tensor).detach().numpy().flatten()
    num_layers = random.choices(SEARCH_SPACE["num_layers"], weights=probs)[0]
    layer_type = random.choices(SEARCH_SPACE["layer_type"], weights=probs)[0]
    return num_layers, layer_type


# Define a simple evaluation function (random accuracy for demonstration)
def evaluate_architecture(num_layers, layer_type):
    return random.uniform(0.7, 0.95)  # Simulating accuracy score


# Train the RL agent to optimize architecture
controller = NASController()
optimizer = optim.Adam(controller.parameters(), lr=0.01)
num_epochs = 100

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Sample an architecture
    num_layers, layer_type = sample_architecture(controller)

    # Evaluate its performance (reward = accuracy)
    reward = evaluate_architecture(num_layers, layer_type)

    # Compute policy gradient loss
    loss = -torch.log(torch.tensor(reward))  # REINFORCE loss
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}: Layers = {num_layers}, Type = {layer_type}, Reward = {reward:.4f}")
