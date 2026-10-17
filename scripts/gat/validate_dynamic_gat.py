
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gat_model import AluvaGAT

BASE = Path("data/processed/aluva")

data = torch.load(
    BASE / "aluva_gat_prediction_data.pt",
    map_location="cpu",
    weights_only=True
)

model = AluvaGAT(
    input_features=13,
    hidden_features=16,
    output_features=5
)

model.load_state_dict(
    torch.load(
        BASE / "aluva_dynamic_gat_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

inputs = data["inputs"]
edge_index = data["edge_index"]

with torch.no_grad():
    predictions = torch.stack([
        model(inputs[i], edge_index)
        for i in range(len(inputs))
    ])

assert predictions.shape == (9, 26, 5)
assert torch.isfinite(predictions).all()
assert edge_index.shape == (2, 98)

print("DYNAMIC GAT VALIDATION")
print("-----------------------------")
print("Model loading: PASS")
print("Prediction shape: PASS")
print("Graph structure: PASS")
print("Finite predictions: PASS")

print("\nPrediction tensor:", tuple(predictions.shape))
print("Invalid values:", (~torch.isfinite(predictions)).sum().item())

print("\nSUCCESS! Dynamic GAT validation completed.")
