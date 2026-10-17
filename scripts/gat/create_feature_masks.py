
from pathlib import Path

import torch

BASE = Path("data/processed/aluva")

graph = torch.load(
    BASE / "aluva_gat_data.pt",
    map_location="cpu",
    weights_only=False
)

torch.manual_seed(42)

num_nodes, num_features = graph.x.shape

# Randomly hide 20% of feature entries.
random_values = torch.rand(num_nodes, num_features)
hidden_mask = random_values < 0.20

# Ensure every feature has at least one hidden entry.
for feature in range(num_features):
    if not hidden_mask[:, feature].any():
        hidden_mask[feature, feature] = True

# Create model input with hidden values replaced by zero.
masked_x = graph.x.clone()
masked_x[hidden_mask] = 0.0

OUTPUT = BASE / "aluva_feature_masks.pt"

torch.save({
    "masked_x": masked_x,
    "hidden_mask": hidden_mask,
    "original_x": graph.x,
    "edge_index": graph.edge_index,
    "ward_numbers": graph.ward_numbers,
    "seed": 42
}, OUTPUT)

print("Total feature entries:", num_nodes * num_features)
print("Hidden entries:", hidden_mask.sum().item())
print("Visible entries:", (~hidden_mask).sum().item())
print("Masked tensor shape:", tuple(masked_x.shape))

print("\nSUCCESS! Feature masks created.")
print("Saved:", OUTPUT)
