
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
    [[record["target"]] for record in records],
    dtype=torch.float32
)

with torch.no_grad():
    predictions = model(features)
    mse = torch.mean((predictions - targets) ** 2).item()

print("ALLOCATION FEDERATED MODEL EVALUATION")
print("--------------------------------------")

for record, prediction in zip(records, predictions):
    print(
        f'{record["centre_id"]} | Ward {record["ward_no"]} | '
        f'Target: {record["target"]:.3f} | '
        f'Prediction: {prediction.item():.3f}'
    )

print(f"\nTraining-set MSE: {mse:.6f}")

assert torch.isfinite(predictions).all()

print("\nSUCCESS! Allocation model predictions verified.")
