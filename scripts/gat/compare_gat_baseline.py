
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_gat_model import AluvaGAT

BASE = Path("data/processed/aluva")

graph = torch.load(
    BASE / "aluva_gat_data.pt",
    map_location="cpu",
    weights_only=False
)

torch.manual_seed(42)

# Load our previously trained GAT.
gat = AluvaGAT(
    input_features=8,
    hidden_features=16,
    output_features=8
)

gat.load_state_dict(
    torch.load(
        BASE / "aluva_gat_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

gat.eval()

with torch.no_grad():
    gat_output = gat(graph.x, graph.edge_index)
    gat_loss = nn.functional.mse_loss(
        gat_output, graph.x
    ).item()

# Non-graph baseline: processes each ward independently.
mlp = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 8)
)

optimizer = torch.optim.Adam(
    mlp.parameters(),
    lr=0.01,
    weight_decay=0.0005
)

loss_fn = nn.MSELoss()

for epoch in range(200):
    mlp.train()
    optimizer.zero_grad()

    output = mlp(graph.x)
    loss = loss_fn(output, graph.x)

    loss.backward()
    optimizer.step()

mlp.eval()

with torch.no_grad():
    mlp_loss = loss_fn(
        mlp(graph.x),
        graph.x
    ).item()

print("MODEL COMPARISON")
print("----------------------------")
print(f"GAT reconstruction MSE: {gat_loss:.6f}")
print(f"MLP reconstruction MSE: {mlp_loss:.6f}")

if gat_loss < mlp_loss:
    print("Lower training error: GAT")
elif mlp_loss < gat_loss:
    print("Lower training error: MLP")
else:
    print("Training errors are equal")

print("\nNote: This is training-set reconstruction only.")
print("SUCCESS! Baseline comparison completed.")
