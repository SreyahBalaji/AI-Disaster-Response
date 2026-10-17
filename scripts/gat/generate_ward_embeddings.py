
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_gat_model import AluvaGAT

BASE = Path("data/processed/aluva")

graph = torch.load(
    BASE / "aluva_gat_data.pt",
    map_location="cpu",
    weights_only=False
)

model = AluvaGAT(
    input_features=graph.num_node_features,
    hidden_features=16,
    output_features=8
)

model.load_state_dict(
    torch.load(
        BASE / "aluva_gat_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

with torch.no_grad():
    embeddings = model(
        graph.x,
        graph.edge_index
    )

assert embeddings.shape == (26, 8)
assert torch.isfinite(embeddings).all()

ward_numbers = graph.ward_numbers.tolist()

columns = [
    f"gat_feature_{i}"
    for i in range(1, 9)
]

df = pd.DataFrame(
    embeddings.numpy(),
    columns=columns
)

df.insert(0, "ward_no", ward_numbers)

OUTPUT = BASE / "aluva_gat_embeddings.csv"

df.to_csv(OUTPUT, index=False)

print("Total wards:", len(df))
print("Embedding dimensions:", len(columns))
print("Missing values:", df.isna().sum().sum())

print("\nFirst five ward embeddings:")
print(df.head().round(4).to_string(index=False))

print("\nSUCCESS! Saved:", OUTPUT)
