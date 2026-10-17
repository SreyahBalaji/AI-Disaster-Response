
from pathlib import Path
import torch

BASE = Path("data/processed/aluva")

INPUT = BASE / "aluva_feature_masks.pt"
OUTPUT = BASE / "aluva_feature_split.pt"

data = torch.load(
    INPUT,
    map_location="cpu",
    weights_only=True
)

hidden_mask = data["hidden_mask"]

# Positions of all hidden feature values.
hidden_positions = hidden_mask.nonzero(as_tuple=False)

generator = torch.Generator().manual_seed(42)

permutation = torch.randperm(
    len(hidden_positions),
    generator=generator
)

train_count = int(len(hidden_positions) * 0.75)

train_positions = hidden_positions[
    permutation[:train_count]
]

test_positions = hidden_positions[
    permutation[train_count:]
]

train_mask = torch.zeros_like(
    hidden_mask,
    dtype=torch.bool
)

test_mask = torch.zeros_like(
    hidden_mask,
    dtype=torch.bool
)

train_mask[
    train_positions[:, 0],
    train_positions[:, 1]
] = True

test_mask[
    test_positions[:, 0],
    test_positions[:, 1]
] = True

assert not (train_mask & test_mask).any()
assert torch.equal(
    train_mask | test_mask,
    hidden_mask
)

torch.save({
    **data,
    "train_mask": train_mask,
    "test_mask": test_mask
}, OUTPUT)

print("Total hidden values:", hidden_mask.sum().item())
print("Training targets:", train_mask.sum().item())
print("Testing targets:", test_mask.sum().item())
print("Overlap:", (train_mask & test_mask).sum().item())

print("\nSUCCESS! Train/test masks created.")
print("Saved:", OUTPUT)
