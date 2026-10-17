
import pandas as pd

from resource_allocation_env import ResourceAllocationEnv
from fairness_metrics import calculate_fairness


def fair_demand_allocate(env, budget=300):
    resource = "water_litres"

    priorities = pd.read_csv(
        "data/processed/aluva/aluva_ward_priority.csv"
    ).set_index("ward_no")["flood_priority_score"]

    wards = [
        ward for ward in env.demand
        if env.demand[ward][resource] > 0
    ]

    original_demand = {
        ward: (
            env.demand[ward][resource]
            + env.delivered[ward][resource]
        )
        for ward in wards
    }

    allocated_total = 0

    while allocated_total < budget:
        eligible = [
            ward for ward in wards
            if env.demand[ward][resource] > 0
        ]

        if not eligible:
            break

        def score(ward):
            fulfilment = (
                env.delivered[ward][resource]
                / original_demand[ward]
            )

            return (
                0.9 * (1 - fulfilment)
                + 0.1 * priorities.loc[ward]
            )

        ward = max(eligible, key=score)

        allocated = 0

        for centre in env.centres:
            if centre["resources"][resource] <= 0:
                continue

            allocated = env.allocate(
                centre["centre_id"],
                ward,
                resource,
                1
            )

            if allocated > 0:
                break

        if allocated == 0:
            break

        allocated_total += allocated

    return allocated_total


if __name__ == "__main__":
    scenarios = [
        ("Normal", None),
        ("Ward 9 surge", 9),
        ("Ward 5 surge", 5),
    ]

    print("FAIR DEMAND ALLOCATION TEST")
    print("--------------------------------")

    for name, surge_ward in scenarios:
        env = ResourceAllocationEnv()

        if surge_ward is not None:
            env.demand[surge_ward]["water_litres"] += 500

        allocated = fair_demand_allocate(
            env, budget=300
        )

        fairness = calculate_fairness(
            env, "water_litres"
        )

        print(f"\nScenario: {name}")

        for ward in [3, 5, 9]:
            print(
                f"Ward {ward}: "
                f"{env.delivered[ward]['water_litres']} L"
            )

        print("Total allocated:", allocated)
        print(
            "Fairness gap:",
            round(fairness["fairness_gap"], 4)
        )

        assert allocated == 300

    print("\nSUCCESS! Fair demand allocation tested.")
