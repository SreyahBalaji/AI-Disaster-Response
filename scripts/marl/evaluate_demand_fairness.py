
from resource_allocation_env import ResourceAllocationEnv
from demand_responsive_allocation import demand_responsive_allocate
from fairness_metrics import calculate_fairness

scenarios = [
    ("Normal", None),
    ("Ward 9 surge", 9),
    ("Ward 5 surge", 5),
]

print("DEMAND-RESPONSIVE FAIRNESS EVALUATION")
print("--------------------------------------")

for name, surge_ward in scenarios:
    env = ResourceAllocationEnv()

    if surge_ward is not None:
        env.demand[surge_ward]["water_litres"] += 500

    # Record original demand before allocating.
    original_demand = {
        ward: env.demand[ward]["water_litres"]
        for ward in [3, 5, 9]
    }

    allocated = demand_responsive_allocate(
        env, budget=300
    )

    fairness = calculate_fairness(
        env, "water_litres"
    )

    print(f"\nScenario: {name}")

    for ward in [3, 5, 9]:
        delivered = env.delivered[ward]["water_litres"]
        demand = original_demand[ward]

        fulfilment = (
            delivered / demand if demand > 0 else 0
        )

        print(
            f"Ward {ward}: "
            f"{delivered}/{demand} L "
            f"({fulfilment:.2%} fulfilled)"
        )

    print(
        f"Fairness gap: {fairness['fairness_gap']:.4f}"
    )

    assert allocated == 300

print("\nSUCCESS! Fairness evaluation completed.")
