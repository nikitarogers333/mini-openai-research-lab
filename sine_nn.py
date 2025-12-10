import torch
import torch.nn as nn
import torch.optim as optim
import math

# Make sure we're on CPU
device = torch.device("cpu")

# Training data: y = sin(x) on [0, 2π]
X = torch.linspace(0, 2 * math.pi, 200, device=device).unsqueeze(1)  # shape [200, 1]
y = torch.sin(X)

# 2-layer neural network: 1 -> 16 -> 1
class SineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

model = SineNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 3000
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Test a few points
test_points = torch.tensor([[0.0],
                            [math.pi / 2],
                            [math.pi],
                            [3 * math.pi / 2]], device=device)

with torch.no_grad():
    preds = model(test_points)

print("\nTest points and predictions:")
for x_val, y_hat in zip(test_points, preds):
    print(f"x={float(x_val.item()):.3f},  pred≈{float(y_hat.item()):.3f},  true≈{math.sin(float(x_val.item())):.3f}")

