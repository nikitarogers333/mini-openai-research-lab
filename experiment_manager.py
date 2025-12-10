import torch
import torch.nn as nn
import torch.optim as optim
import math

device = torch.device("cpu")

# ---- Create dataset: sin(x) ----
X = torch.linspace(0, 2 * math.pi, 200, device=device).unsqueeze(1)
y = torch.sin(X)

# ---- Define the model template ----
class SineNet(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_units)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ---- Experiment configurations ----
experiments = [
    {"hidden": 8, "lr": 0.01},
    {"hidden": 16, "lr": 0.005},
    {"hidden": 32, "lr": 0.001},
]

results = []


# ---- Training function ----
def run_experiment(hidden_units, lr, epochs=2000):
    model = SineNet(hidden_units).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        y_pred = model(X)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, loss.item()


# ---- Run all experiments ----
print("\nRUNNING EXPERIMENTS...\n")

for exp in experiments:
    print(f"Training model: hidden={exp['hidden']}, lr={exp['lr']}")
    model, final_loss = run_experiment(exp["hidden"], exp["lr"])
    exp["loss"] = final_loss
    exp["model"] = model
    print(f" → Final loss: {final_loss:.6f}\n")


# ---- Find best model ----
best = min(experiments, key=lambda d: d["loss"])

print("\n==============================")
print(" BEST EXPERIMENT RESULT")
print("==============================")
print(f"Hidden units: {best['hidden']}")
print(f"Learning rate: {best['lr']}")
print(f"Final loss: {best['loss']:.6f}")

print("\nTesting best model...\n")

# Test a few known points: 0, π/2, π, 3π/2
test_points = torch.tensor([[0.0],
                            [math.pi / 2],
                            [math.pi],
                            [3 * math.pi / 2]], device=device)

with torch.no_grad():
    preds = best["model"](test_points)

for x_val, y_hat in zip(test_points, preds):
    print(f"x={float(x_val.item()):.3f}, pred≈{float(y_hat.item()):.3f}, true≈{math.sin(float(x_val.item())):.3f}")
