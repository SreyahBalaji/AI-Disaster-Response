
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gat_model import AluvaGAT

graph = torch.load(
    Path("data/processed/aluva/aluva_gat_data.pt"),
    map_location="cpu",
    weights_only=False
)

results = []

def train_and_evaluate(model, inputs, targets,
                       train_mask, test_mask,
                       edge_index=None):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=0.0005
    )

    for _ in range(200):
        model.train()
        optimizer.zero_grad()

        output = (
            model(inputs, edge_index)
            if edge_index is not None
            else model(inputs)
        )

        loss = F.mse_loss(
            output[train_mask],
            targets[train_mask]
        )

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        output = (
            model(inputs, edge_index)
            if edge_index is not None
            else model(inputs)
        )

        return F.mse_loss(
            output[test_mask],
            targets[test_mask]
        ).item()


for seed in [42, 43, 44, 45, 46]:
    generator = torch.Generator().manual_seed(seed)

    original = graph.x
    hidden = torch.rand(
        original.shape,
        generator=generator
    ) < 0.20

    positions = hidden.nonzero(as_tuple=False)
    permutation = torch.randperm(
        len(positions),
        generator=generator
    )

    train_count = int(len(positions) * 0.75)

    train_mask = torch.zeros_like(hidden)
    test_mask = torch.zeros_like(hidden)

    for index in permutation[:train_count]:
        row, col = positions[index].tolist()
        train_mask[row, col] = True

    for index in permutation[train_count:]:
        row, col = positions[index].tolist()
        test_mask[row, col] = True

    masked_x = original.clone()
    masked_x[hidden] = 0.0

    inputs = torch.cat(
        [masked_x, hidden.float()],
        dim=1
    )

    torch.manual_seed(seed)
    gat = AluvaGAT(16, 16, 8)

    torch.manual_seed(seed)
    mlp = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8)
    )

    gat_mse = train_and_evaluate(
        gat, inputs, original,
        train_mask, test_mask,
        graph.edge_index
    )

    mlp_mse = train_and_evaluate(
        mlp, inputs, original,
        train_mask, test_mask
    )

    baseline = torch.zeros_like(original)

    for feature in range(original.shape[1]):
        visible = original[
            ~hidden[:, feature],
            feature
        ]

        baseline[:, feature] = (
            visible.mean()
            if len(visible) > 0
            else 0.0
        )

    mean_mse = F.mse_loss(
        baseline[test_mask],
        original[test_mask]
    ).item()

    results.append([gat_mse, mlp_mse, mean_mse])

    print(
        f"Seed {seed}: "
        f"GAT={gat_mse:.4f}, "
        f"MLP={mlp_mse:.4f}, "
        f"Mean={mean_mse:.4f}"
    )

results = torch.tensor(results)

print("\nAVERAGE TEST MSE")
print("-------------------------")
print(f"GAT:  {results[:, 0].mean():.6f}")
print(f"MLP:  {results[:, 1].mean():.6f}")
print(f"Mean: {results[:, 2].mean():.6f}")

print("\nSUCCESS! Multi-split evaluation completed.")
