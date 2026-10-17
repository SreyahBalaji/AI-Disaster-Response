
from pathlib import Path
import torch

BASE = Path("data/processed/aluva")

data = torch.load(
    BASE / "aluva_dynamic_gat_normalized.pt",
    map_location="cpu",
    weights_only=True
)

x = data["x"]
minutes = data["simulation_minutes"]

assert x.shape == (10, 26, 13)
assert len(minutes) == 10

# Input: all 13 features at the current snapshot.
inputs = x[:-1].clone()

# Target: five demand features at the next snapshot.
targets = x[1:, :, 8:].clone()

assert inputs.shape == (9, 26, 13)
assert targets.shape == (9, 26, 5)

# First six transitions for training.
# Last three transitions for evaluation.
train_indices = list(range(6))
test_indices = list(range(6, 9))

assert max(train_indices) < min(test_indices)

output = BASE / "aluva_gat_prediction_data.pt"

torch.save({
    "inputs": inputs,
    "targets": targets,
    "edge_index": data["edge_index"],
    "ward_numbers": data["ward_numbers"],
    "input_minutes": minutes[:-1],
    "target_minutes": minutes[1:],
    "train_indices": train_indices,
    "test_indices": test_indices,
    "demand_mean": data["demand_mean"],
    "demand_std": data["demand_std"]
}, output)

print("GAT PREDICTION DATA")
print("-----------------------------")
print("Input shape:", tuple(inputs.shape))
print("Target shape:", tuple(targets.shape))
print("Training transitions:", len(train_indices))
print("Evaluation transitions:", len(test_indices))

print("\nTime transitions:")
for i, (start, end) in enumerate(
    zip(minutes[:-1], minutes[1:])
):
    group = "TRAIN" if i in train_indices else "TEST"
    print(f"{start:>3} -> {end:>3} minutes [{group}]")

print("\nSUCCESS! Prediction targets prepared.")
print("Saved:", output)
