
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = Path("data/processed/aluva")

data = torch.load(
    BASE / "aluva_feature_split.pt",
    map_location="cpu",
    weights_only=True
)

original = data["original_x"]
hidden_mask = data["hidden_mask"]
test_mask = data["test_mask"]

visible_mask = ~hidden_mask

predictions = torch.zeros_like(original)

for feature in range(original.shape[1]):
    visible_values = original[
        visible_mask[:, feature],
        feature
    ]

    assert len(visible_values) > 0

    mean_value = visible_values.mean()

    predictions[:, feature] = mean_value

test_mse = F.mse_loss(
    predictions[test_mask],
    original[test_mask]
).item()

print("MEAN BASELINE EVALUATION")
print("----------------------------")
print("Test entries:", test_mask.sum().item())
print(f"Mean baseline test MSE: {test_mse:.6f}")

print("\nPrevious model results:")
print("GAT test MSE: 2.925954")
print("MLP test MSE: 1.637226")

print("\nSUCCESS! Mean baseline evaluated.")
