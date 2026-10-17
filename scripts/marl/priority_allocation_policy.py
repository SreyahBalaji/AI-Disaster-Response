
import pandas as pd

from resource_allocation_env import ResourceAllocationEnv
from fairness_metrics import calculate_fairness


def priority_allocate(env, resource, water_budget):
    priorities = pd.read_csv(
        "data/processed/aluva/aluva_ward_priority.csv"
    ).set_index("ward_no")["flood_priority_score"]

    remaining_budget = water_budget

    while remaining_budget > 0:
        eligible = [
            ward for ward in env.demand
            if env.demand[ward][resource] > 0
        ]

        if not eligible:
            break

        # Allocate to the highest-priority ward first.
        ward = max(eligible, key=lambda w: priorities.loc[w])

        centres = sorted(
            env.centres,
            key=lambda c: c["resources"][resource],
            reverse=True
        )

        centre = next(
            (
                c for c in centres
                if c["resources"][resource] > 0
            ),
            None
        )

        if centre is None:
            break

        quantity = min(10, remaining_budget)

        allocated = env.allocate(
            centre["centre_id"],
            ward,
            resource,
            quantity
        )

        if allocated == 0:
            break

        remaining_budget -= allocated

    return env


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    priority_allocate(
        env,
        resource="water_litres",
        water_budget=300
    )

    fairness = calculate_fairness(env, "water_litres")

    print("PRIORITY-AWARE ALLOCATION TEST")
    print("--------------------------------")

    for ward in [3, 5, 9]:
        print(
            f"Ward {ward}: "
            f"{env.delivered[ward]['water_litres']} L delivered"
        )

    print("Fairness gap:", fairness["fairness_gap"])

    total_delivered = sum(
        values["water_litres"]
        for values in env.delivered.values()
    )

    assert total_delivered == 300

    print("\nSUCCESS! Priority allocation completed.")
