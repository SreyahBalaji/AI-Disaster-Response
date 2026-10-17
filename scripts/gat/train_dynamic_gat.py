
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gat_model import AluvaGAT

BASE = Path("data/processed/aluva")

data = torch.load(
    BASE / "aluva_gat_prediction_data.pt",
    map_location="cpu",
    weights_only=True
)

inputs = data["inputs"]
targets = data["targets"]
edge_index = data["edge_index"]

train_indices = data["train_indices"]
test_indices = data["test_indices"]

torch.manual_seed(42)

model = AluvaGAT(
    input_features=13,
    hidden_features=16,
    output_features=5
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=0.0005
)

# Train only on earlier transitions.
for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    predictions = torch.stack([
        model(inputs[i], edge_index)
        for i in train_indices
    ])

    loss = F.mse_loss(
        predictions,
        targets[train_indices]
    )

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Training MSE = {loss.item():.6f}"
        )

model.eval()

with torch.no_grad():
    test_predictions = torch.stack([
        model(inputs[i], edge_index)
        for i in test_indices
    ])

    test_targets = targets[test_indices]

    gat_mse = F.mse_loss(
        test_predictions,
        test_targets
    ).item()

    # Persistence: next demand = current demand.
    baseline_predictions = inputs[
        test_indices, :, 8:
    ]

    baseline_mse = F.mse_loss(
        baseline_predictions,
        test_targets
    ).item()

assert torch.isfinite(test_predictions).all()

model_path = BASE / "aluva_dynamic_gat_model.pth"
torch.save(model.state_dict(), model_path)

print("\nDYNAMIC GAT EVALUATION")
print("-----------------------------")
print(f"GAT test MSE: {gat_mse:.6f}")
print(f"Persistence baseline MSE: {baseline_mse:.6f}")

if gat_mse < baseline_mse:
    print("Lower test error: GAT")
else:
    print("Lower test error: Persistence baseline")

print("\nSUCCESS! Dynamic GAT training completed.")
print("Saved:", model_path)
