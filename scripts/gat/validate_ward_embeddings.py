
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

embeddings_df = pd.read_csv(
    BASE / "aluva_gat_embeddings.csv"
)

feature_columns = [
    f"gat_feature_{i}"
    for i in range(1, 9)
]

assert len(embeddings_df) == 26
assert embeddings_df["ward_no"].is_unique
assert set(embeddings_df["ward_no"]) == set(range(1, 27))
assert embeddings_df[feature_columns].shape == (26, 8)

assert not embeddings_df[feature_columns].isna().any().any()

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
    expected = model(
        graph.x,
        graph.edge_index
    ).numpy()

# Align CSV rows with the graph's ward ordering.
ordered = (
    embeddings_df
    .set_index("ward_no")
    .loc[graph.ward_numbers.tolist()]
)

actual = ordered[feature_columns].to_numpy()

assert np.isfinite(actual).all()

assert np.allclose(
    actual,
    expected,
    rtol=1e-5,
    atol=1e-6
), "Saved embeddings do not match model output"

print("Ward coverage: PASS")
print("Embedding dimensions: PASS")
print("Missing/invalid values: PASS")
print("Model-output consistency: PASS")

print("\nSUCCESS! All GAT embedding checks passed.")
