
import json
from pathlib import Path

import torch

BASE = Path("data/processed/aluva")

graph = torch.load(
    BASE / "aluva_gat_data.pt",
    map_location="cpu",
    weights_only=False
)

demand_path = Path(
    "data/processed/crisis_reports/"
    "aluva_cumulative_demand.json"
)

with open(demand_path, "r", encoding="utf-8") as file:
    snapshots = json.load(file)

resource_names = [
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases"
]

ward_numbers = graph.ward_numbers.tolist()

assert len(snapshots) == 10
assert len(ward_numbers) == 26

dynamic_data = []
simulation_minutes = []

for snapshot in snapshots:
    ward_demand = snapshot["ward_demand"]

    assert set(ward_demand.keys()) == {
        str(ward) for ward in ward_numbers
    }

    demand_rows = []

    for ward in ward_numbers:
        resources = ward_demand[str(ward)]

        values = [
            float(resources[name])
            for name in resource_names
        ]

        assert all(value >= 0 for value in values)
        demand_rows.append(values)

    demand_tensor = torch.tensor(
        demand_rows,
        dtype=torch.float32
    )

    combined = torch.cat(
        [graph.x, demand_tensor],
        dim=1
    )

    dynamic_data.append(combined)

    simulation_minutes.append(
        snapshot["simulation_minute"]
    )

dynamic_tensor = torch.stack(dynamic_data)

assert dynamic_tensor.shape == (10, 26, 13)
assert torch.isfinite(dynamic_tensor).all()

output_path = BASE / "aluva_dynamic_gat_data.pt"

torch.save({
    "x": dynamic_tensor,
    "edge_index": graph.edge_index,
    "ward_numbers": graph.ward_numbers,
    "simulation_minutes": simulation_minutes,
    "feature_names": graph.feature_names + resource_names
}, output_path)

print("DYNAMIC GAT DATA")
print("-----------------------------")
print("Snapshots:", dynamic_tensor.shape[0])
print("Wards:", dynamic_tensor.shape[1])
print("Features per ward:", dynamic_tensor.shape[2])
print("Graph edges:", graph.edge_index.shape[1])
print("Simulation minutes:", simulation_minutes)

print("\nSUCCESS! Dynamic GAT data prepared.")
print("Saved:", output_path)
