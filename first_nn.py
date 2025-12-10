import torch
import torch.nn as nn
import torch.optim as optim

# Training data: y = 2x + 3
X = torch.linspace(0, 10, 100).unsqueeze(1)
y = 2 * X + 3

# Simple linear model
model = nn.Linear(1, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 2000
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Test prediction
test_x = torch.tensor([[4.0]])
predicted_y = model(test_x).item()
print("Prediction for x=4:", predicted_y)
