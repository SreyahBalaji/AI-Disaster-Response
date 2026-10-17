
import json
from pathlib import Path

import torch
from local_model import LocalReliefModel

data_dir = Path("data/processed/federated")

with open(
    data_dir / "aluva_allocation_training_data.json",
    encoding="utf-8"
) as file:
    records = json.load(file)

model = LocalReliefModel()
model.load_state_dict(
    torch.load(
        data_dir / "aluva_allocation_federated_model.pth",
        map_location="cpu",
        weights_only=True
    )
)
model.eval()

features = torch.tensor(
    [record["features"] for record in records],
    dtype=torch.float32
)

targets = torch.tensor(
    [record["target"] for record in records],
    dtype=torch.float32
)

with torch.no_grad():
    predictions = model(features).squeeze(-1)

model_mse = torch.mean(
    (predictions - targets) ** 2
).item()

# Exact formula used to create the targets.
formula_predictions = torch.tensor([
    min(
        1.0,
        record["features"][1] / record["features"][0]
    )
    for record in records
], dtype=torch.float32)

formula_mse = torch.mean(
    (formula_predictions - targets) ** 2
).item()

print("ALLOCATION BASELINE COMPARISON")
print("--------------------------------")
print(f"Federated model MSE: {model_mse:.6f}")
print(f"Formula baseline MSE: {formula_mse:.6f}")

print("\nSUCCESS! Baseline comparison completed.")
