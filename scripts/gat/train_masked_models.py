
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gat_model import AluvaGAT

BASE = Path("data/processed/aluva")

data = torch.load(
    BASE / "aluva_feature_split.pt",
    map_location="cpu",
    weights_only=True
)

x = data["masked_x"]
target = data["original_x"]
edge_index = data["edge_index"]

train_mask = data["train_mask"]
test_mask = data["test_mask"]

assert train_mask.sum() == 33
assert test_mask.sum() == 11
assert not (train_mask & test_mask).any()

# Give both models a missing-value indicator.
# This distinguishes a masked zero from a real zero.
model_input = torch.cat(
    [x, data["hidden_mask"].float()],
    dim=1
)

torch.manual_seed(42)

gat = AluvaGAT(
    input_features=16,
    hidden_features=16,
    output_features=8
)

torch.manual_seed(42)

mlp = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 8)
)


def train_model(model, uses_graph):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=0.0005
    )

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()

        if uses_graph:
            output = model(model_input, edge_index)
        else:
            output = model(model_input)

        loss = nn.functional.mse_loss(
            output[train_mask],
            target[train_mask]
        )

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        if uses_graph:
            output = model(model_input, edge_index)
        else:
            output = model(model_input)

        train_mse = nn.functional.mse_loss(
            output[train_mask],
            target[train_mask]
        ).item()

        test_mse = nn.functional.mse_loss(
            output[test_mask],
            target[test_mask]
        ).item()

    return train_mse, test_mse


gat_train, gat_test = train_model(gat, True)
mlp_train, mlp_test = train_model(mlp, False)

print("MASKED FEATURE RECONSTRUCTION")
print("--------------------------------")
print(f"GAT | Train MSE: {gat_train:.6f}")
print(f"GAT | Test MSE:  {gat_test:.6f}")
print(f"MLP | Train MSE: {mlp_train:.6f}")
print(f"MLP | Test MSE:  {mlp_test:.6f}")

if gat_test < mlp_test:
    print("\nLower held-out error: GAT")
elif mlp_test < gat_test:
    print("\nLower held-out error: MLP")
else:
    print("\nHeld-out errors are equal")

print("\nSUCCESS! Masked-model comparison completed.")
