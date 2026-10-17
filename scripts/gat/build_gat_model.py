
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import GATConv

DATA_PATH = Path(
    "data/processed/aluva/aluva_gat_data.pt"
)


class AluvaGAT(nn.Module):
    def __init__(self, input_features, hidden_features=16,
                 output_features=8):
        super().__init__()

        self.gat1 = GATConv(
            in_channels=input_features,
            out_channels=hidden_features,
            heads=2,
            concat=True,
            dropout=0.0
        )

        self.gat2 = GATConv(
            in_channels=hidden_features * 2,
            out_channels=output_features,
            heads=1,
            concat=False,
            dropout=0.0
        )

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x


def main():
    # This file was generated locally in Step 4.3.
    # Load only trusted project files.
    graph = torch.load(
        DATA_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = AluvaGAT(
        input_features=graph.num_node_features
    )

    model.eval()

    with torch.no_grad():
        embeddings = model(
            graph.x,
            graph.edge_index
        )

    assert embeddings.shape == (26, 8)
    assert torch.isfinite(embeddings).all()

    print("Input features:", graph.num_node_features)
    print("GAT layers:", 2)
    print("Attention heads (first layer):", 2)
    print("Output embedding shape:", tuple(embeddings.shape))
    print("Model parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    print("\nSUCCESS! GAT forward pass completed.")


if __name__ == "__main__":
    main()
