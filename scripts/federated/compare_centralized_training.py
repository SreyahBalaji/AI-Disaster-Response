
import torch
from local_model import LocalReliefModel

torch.manual_seed(42)

# Recreate the same synthetic datasets used in FedAvg.
datasets = []

for index in range(3):
    x = torch.rand(40, 5)
    x[:, 0] = torch.clamp(x[:, 0] + index * 0.1, 0, 1)

    y = (
        0.5 * x[:, 0:1]
        + 0.3 * x[:, 1:2]
        + 0.2 * x[:, 2:3]
    )

    datasets.append((x, y))

x_train = torch.cat([x for x, y in datasets])
y_train = torch.cat([y for x, y in datasets])

model = LocalReliefModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = torch.nn.MSELoss()

for epoch in range(100):
    predictions = model(x_train)
    loss = loss_function(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Use the same test distribution as Step 6.7.
torch.manual_seed(123)
x_test = torch.rand(100, 5)

y_test = (
    0.5 * x_test[:, 0:1]
    + 0.3 * x_test[:, 1:2]
    + 0.2 * x_test[:, 2:3]
)

model.eval()

with torch.no_grad():
    centralized_mse = loss_function(
        model(x_test), y_test
    ).item()

print("CENTRALIZED TRAINING BASELINE")
print("--------------------------------")
print(f"Centralized test MSE: {centralized_mse:.6f}")
print("Federated test MSE (previous run): 0.000723")

print("\nSUCCESS! Centralized baseline evaluated.")
