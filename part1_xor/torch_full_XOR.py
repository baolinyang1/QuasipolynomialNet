import torch
import torch.nn as nn
import torch.optim as optim

# Define the XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the neural network
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden = nn.Linear(2, 2)  # One hidden layer with 2 neurons
        self.output = nn.Linear(2, 1)  # Output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

# Instantiate the model
model = XORModel()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Test the model
with torch.no_grad():
    test_outputs = model(X)
    predicted = (test_outputs > 0.5).float()
    print("Predicted outputs:", predicted)
