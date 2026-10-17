
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gat_model import AluvaGAT

BASE = Path("data/processed/aluva")

graph = torch.load(
    BASE / "aluva_gat_data.pt",
    map_location="cpu",
    weights_only=False
)

original = graph.x
num_nodes, num_features = original.shape

self_edges = torch.arange(num_nodes)
self_edge_index = torch.stack(
    [self_edges, self_edges],
    dim=0
)


def evaluate(seed, edge_index, hidden, train_mask, test_mask):
    masked_x = original.clone()
    masked_x[hidden] = 0.0

    inputs = torch.cat(
        [masked_x, hidden.float()],
        dim=1
    )

    torch.manual_seed(seed)

    model = AluvaGAT(
        input_features=16,
        hidden_features=16,
        output_features=8
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=0.0005
    )

    for _ in range(200):
        model.train()
        optimizer.zero_grad()

        output = model(inputs, edge_index)

        loss = F.mse_loss(
            output[train_mask],
            original[train_mask]
        )

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        output = model(inputs, edge_index)

        return F.mse_loss(
            output[test_mask],
            original[test_mask]
        ).item()


results = []

for seed in [42, 43, 44, 45, 46]:
    generator = torch.Generator().manual_seed(seed)

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

    assert train_mask.any()
    assert test_mask.any()
    assert not (train_mask & test_mask).any()

    real_mse = evaluate(
        seed, graph.edge_index,
        hidden, train_mask, test_mask
    )

    self_mse = evaluate(
        seed, self_edge_index,
        hidden, train_mask, test_mask
    )

    results.append([real_mse, self_mse])

    print(
        f"Seed {seed}: "
        f"Real graph={real_mse:.4f}, "
        f"Self-only={self_mse:.4f}"
    )

results = torch.tensor(results)

real_average = results[:, 0].mean().item()
self_average = results[:, 1].mean().item()

print("\nGRAPH ABLATION RESULTS")
print("----------------------------")
print(f"Real graph average MSE: {real_average:.6f}")
print(f"Self-only average MSE:  {self_average:.6f}")

if real_average < self_average:
    print("Lower error: Real ward graph")
else:
    print("Lower error: Self-loops only")

print("\nSUCCESS! Graph ablation completed.")
