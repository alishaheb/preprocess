import torch
import torch.nn as nn

# Define CrossEntropyLoss function
loss_function = nn.CrossEntropyLoss()

# Example: Suppose we have 3 classes (A, B, C)
# True label (correct class is 0, meaning class A)
true_label = torch.tensor([0])  # Class index (not one-hot encoded)
a=1
# Predicted probabilities (before softmax, called logits)
logits = torch.tensor([[2.0, 1.0, 0.1]])  # Raw scores from the model
b=2
# Compute loss
loss = loss_function(logits, true_label)
c=3
print(f"Cross-Entropy Loss: {loss.item()}")
