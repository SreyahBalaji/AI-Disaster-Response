
from pathlib import Path
import sys

import torch
from torch import nn

# Allow importing our existing GAT model.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_gat_model import AluvaGAT

DATA_PATH = Path(
    "data/processed/aluva/aluva_gat_data.pt"
)

MODEL_PATH = Path(
    "data/processed/aluva/aluva_gat_model.pth"
)

torch.manual_seed(42)

graph = torch.load(
    DATA_PATH,
    map_location="cpu",
    weights_only=False
)

model = AluvaGAT(
    input_features=graph.num_node_features,
    hidden_features=16,
    output_features=graph.num_node_features
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=0.0005
)

loss_function = nn.MSELoss()

EPOCHS = 200

print("Training GAT...")
print("Nodes:", graph.num_nodes)
print("Features:", graph.num_node_features)

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    reconstructed = model(
        graph.x,
        graph.edge_index
    )

    loss = loss_function(
        reconstructed,
        graph.x
    )

    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 20 == 0:
        print(
            f"Epoch {epoch:3d}/{EPOCHS} "
            f"| Reconstruction loss: {loss.item():.6f}"
        )

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH)

print("\nFinal loss:", round(loss.item(), 6))
print("Model saved:", MODEL_PATH)
print("SUCCESS! GAT training completed.")
