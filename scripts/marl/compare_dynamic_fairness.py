
import json
from pathlib import Path

from resource_allocation_env import ResourceAllocationEnv
from demand_responsive_allocation import demand_responsive_allocate
from fair_demand_allocation import fair_demand_allocate
from fairness_metrics import calculate_fairness

scenarios = [
    ("normal", None),
    ("ward_9_surge", 9),
    ("ward_5_surge", 5),
]

policies = {
    "demand_responsive": demand_responsive_allocate,
    "fairness_aware": fair_demand_allocate,
}

results = []

for scenario_name, surge_ward in scenarios:
    for policy_name, policy_function in policies.items():
        env = ResourceAllocationEnv()

        if surge_ward is not None:
            env.demand[surge_ward]["water_litres"] += 500

        allocated = policy_function(env, budget=300)

        fairness = calculate_fairness(
            env, "water_litres"
        )

        results.append({
            "scenario": scenario_name,
            "policy": policy_name,
            "allocated_litres": allocated,
            "fairness_gap": round(
                fairness["fairness_gap"], 4
            ),
            "ward_allocations": {
                str(ward): env.delivered[ward]["water_litres"]
                for ward in [3, 5, 9]
            }
        })

        assert allocated == 300

output_path = Path(
    "data/processed/marl/dynamic_fairness_comparison.json"
)

output_path.parent.mkdir(
    parents=True, exist_ok=True
)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("DYNAMIC FAIRNESS POLICY COMPARISON")
print("----------------------------------")

for result in results:
    print(
        f'{result["scenario"]} | '
        f'{result["policy"]} | '
        f'Fairness gap: {result["fairness_gap"]:.4f}'
    )

print("\nSaved:", output_path)
print("\nSUCCESS! Policy comparison saved.")
