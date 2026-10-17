
import json
from pathlib import Path

from test_allocation_integration import ResourceAllocationEnv

env = ResourceAllocationEnv()

records = []

for centre in env.centres:
    centre_id = centre["centre_id"]
    stock = centre["resources"]["water_litres"]

    for ward_no, demand in env.demand.items():
        water_demand = demand["water_litres"]

        if water_demand <= 0:
            continue

        # Normalized features based on current simulation.
        demand_feature = min(water_demand / 1000, 1)
        stock_feature = min(stock / 1000, 1)

        # Placeholder features until vulnerability and
        # accessibility are integrated.
        severity = 0.5
        vulnerability = 0.5
        accessibility = 0.5

        # Illustrative supervised target:
        # fraction of demand that current stock could fulfil.
        target = min(stock, water_demand) / water_demand

        records.append({
            "centre_id": centre_id,
            "ward_no": ward_no,
            "features": [
                demand_feature,
                stock_feature,
                severity,
                vulnerability,
                accessibility
            ],
            "target": round(target, 6)
        })

output_dir = Path("data/processed/federated")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "aluva_allocation_training_data.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(records, file, indent=4)

print("ALLOCATION TRAINING DATA")
print("----------------------------")
print("Training examples:", len(records))
print("Participating centres:", len(env.centres))
print("Saved:", output_path)

assert len(records) > 0

print("\nSUCCESS! Allocation training data prepared.")
