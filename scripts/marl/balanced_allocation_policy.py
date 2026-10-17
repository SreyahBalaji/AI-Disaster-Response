
import pandas as pd

from resource_allocation_env import ResourceAllocationEnv
from fairness_metrics import calculate_fairness


def balanced_allocate(env, budget=300):
    resource = "water_litres"

    priorities = pd.read_csv(
        "data/processed/aluva/aluva_ward_priority.csv"
    ).set_index("ward_no")["flood_priority_score"]

    wards = [
        ward for ward in env.demand
        if env.demand[ward][resource] > 0
    ]

    def allocate_to_ward(ward, quantity):
        remaining = quantity
        delivered = 0

        for centre in env.centres:
            if remaining <= 0:
                break

            available = centre["resources"][resource]
            if available <= 0:
                continue

            amount = env.allocate(
                centre["centre_id"],
                ward,
                resource,
                min(remaining, available)
            )

            delivered += amount
            remaining -= amount

        return delivered

    # Stage 1: Minimum coverage for every ward.
    coverage_budget = budget // 2
    minimum_share = coverage_budget // len(wards)

    total_allocated = 0

    for ward in wards:
        total_allocated += allocate_to_ward(
            ward, minimum_share
        )

    # Stage 2: Prioritize higher flood exposure.
    remaining_budget = budget - total_allocated

    for ward in sorted(
        wards,
        key=lambda w: priorities.loc[w],
        reverse=True
    ):
        if remaining_budget <= 0:
            break

        allocated = allocate_to_ward(
            ward, remaining_budget
        )

        remaining_budget -= allocated
        total_allocated += allocated

    return total_allocated


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    allocated = balanced_allocate(env, budget=300)

    fairness = calculate_fairness(
        env, "water_litres"
    )

    print("BALANCED ALLOCATION TEST")
    print("--------------------------------")

    for ward in [3, 5, 9]:
        delivered = env.delivered[ward]["water_litres"]
        demand = 480 if ward == 3 else 300 if ward == 5 else 216

        print(
            f"Ward {ward}: {delivered} / {demand} L"
        )

    print("Total allocated:", allocated)
    print("Fairness gap:", fairness["fairness_gap"])

    assert allocated == 300

    print("\nSUCCESS! Balanced allocation completed.")
