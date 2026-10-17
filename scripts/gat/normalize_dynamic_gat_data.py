
from pathlib import Path
import torch

BASE = Path("data/processed/aluva")

INPUT = BASE / "aluva_dynamic_gat_data.pt"
OUTPUT = BASE / "aluva_dynamic_gat_normalized.pt"

data = torch.load(
    INPUT,
    map_location="cpu",
    weights_only=True
)

x = data["x"].clone()

assert x.shape == (10, 26, 13)
assert torch.isfinite(x).all()

# Static features are already standardized.
# The last five columns contain simulated demand.
demand = x[:, :, 8:]

# Fit scaling parameters on the first seven snapshots.
# Later snapshots are reserved for evaluation.
train_demand = demand[:7]

mean = train_demand.mean(dim=(0, 1))
std = train_demand.std(
    dim=(0, 1),
    unbiased=False
)

std = torch.where(
    std < 1e-8,
    torch.ones_like(std),
    std
)

x[:, :, 8:] = (demand - mean) / std

assert torch.isfinite(x).all()

torch.save({
    **data,
    "x": x,
    "demand_mean": mean,
    "demand_std": std,
    "train_snapshot_count": 7
}, OUTPUT)

print("NORMALIZED DYNAMIC GAT DATA")
print("--------------------------------")
print("Tensor shape:", tuple(x.shape))
print("Training snapshots:", 7)
print("Reserved snapshots:", 3)
print("Demand feature means:", mean.tolist())
print("Demand feature stds:", std.tolist())
print("Invalid values:", (~torch.isfinite(x)).sum().item())

print("\nSUCCESS! Dynamic features normalized.")
print("Saved:", OUTPUT)
