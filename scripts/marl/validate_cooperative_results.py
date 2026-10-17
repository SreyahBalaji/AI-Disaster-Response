
import json
from pathlib import Path

path = Path(
    "data/processed/marl/cooperative_experiment_results.json"
)

with open(path, "r", encoding="utf-8") as file:
    results = json.load(file)

scenario = results["scenario"]
models = results["models"]

total_demand = (
    scenario["initial_water_demand_litres"]
    + scenario["additional_ward9_demand_litres"]
)

stock = scenario["initial_water_stock_litres"]

print("COOPERATIVE MARL RESULTS VALIDATION")
print("--------------------------------")
print("Total demand:", total_demand)
print("Available stock:", stock)

for model_name, metrics in models.items():
    delivered = metrics.get(
        "average_delivered_litres",
        metrics.get("delivered_litres")
    )

    satisfaction = metrics["demand_satisfaction"]
    fairness_gap = metrics["fairness_gap"]

    calculated_satisfaction = delivered / total_demand

    assert 0 <= delivered <= stock
    assert 0 <= satisfaction <= 1
    assert 0 <= fairness_gap <= 1

    # Allow for rounded values in the saved report.
    assert abs(
        calculated_satisfaction - satisfaction
    ) < 0.001

    print(
        f"{model_name}: "
        f"delivered={delivered}, "
        f"satisfaction={satisfaction:.4f}, "
        f"fairness_gap={fairness_gap:.4f}"
    )

print("\nSUCCESS! All experimental results validated.")
