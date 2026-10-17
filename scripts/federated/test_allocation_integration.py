
import sys
from pathlib import Path

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "marl")
)

from resource_allocation_env import ResourceAllocationEnv
from local_model import LocalReliefModel

env = ResourceAllocationEnv()

model = LocalReliefModel()
model.load_state_dict(
    torch.load(
        "data/processed/federated/aluva_multiround_global_model.pth",
        map_location="cpu",
        weights_only=True
    )
)
model.eval()

centre = env.centres[0]
ward = 3

stock = centre["resources"]["water_litres"]
demand = env.demand[ward]["water_litres"]

# Demonstration input only; not yet validated model features.
features = torch.tensor([[
    min(demand / 1000, 1),
    min(stock / 1000, 1),
    0.5,
    0.5,
    0.5
]], dtype=torch.float32)

with torch.no_grad():
    prediction = model(features).item()

print("FEDERATED + ALLOCATION INTEGRATION TEST")
print("--------------------------------------")
print("Relief centre:", centre["centre_id"])
print("Ward:", ward)
print("Water demand:", demand)
print("Available water:", stock)
print("Model output:", round(prediction, 6))

assert torch.isfinite(torch.tensor(prediction))

print("\nSUCCESS! Federated model connected to allocation data.")
