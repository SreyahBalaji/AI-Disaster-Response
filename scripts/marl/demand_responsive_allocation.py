
import pandas as pd

from resource_allocation_env import ResourceAllocationEnv


def demand_responsive_allocate(env, budget=300):
    resource = "water_litres"

    priorities = pd.read_csv(
        "data/processed/aluva/aluva_ward_priority.csv"
    ).set_index("ward_no")["flood_priority_score"]

    wards = [
        ward for ward in env.demand
        if env.demand[ward][resource] > 0
    ]

    if not wards:
        return 0

    max_demand = max(
        env.demand[ward][resource] for ward in wards
    )

    weights = {}

    for ward in wards:
        normalized_demand = (
            env.demand[ward][resource] / max_demand
        )

        weights[ward] = (
            0.5 * priorities.loc[ward]
            + 0.5 * normalized_demand
        )

    total_weight = sum(weights.values())
    remaining_budget = budget
    allocated_total = 0

    # Allocate proportional shares.
    for ward in sorted(wards):
        share = int(
            budget * weights[ward] / total_weight
        )

        share = min(
            share,
            env.demand[ward][resource],
            remaining_budget
        )

        for centre in env.centres:
            if share <= 0:
                break

            available = centre["resources"][resource]
            quantity = min(share, available)

            if quantity <= 0:
                continue

            allocated = env.allocate(
                centre["centre_id"],
                ward,
                resource,
                quantity
            )

            share -= allocated
            remaining_budget -= allocated
            allocated_total += allocated

    while remaining_budget > 0:
        eligible_wards = [
            ward for ward in wards
            if env.demand[ward][resource] > 0
        ]

        if not eligible_wards:
            break

        # Give the next litre to the highest-weight
        # ward that still has unmet demand.
        ward = max(
            eligible_wards,
            key=lambda w: weights[w]
        )

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

        remaining_budget -= allocated
        allocated_total += allocated

    return allocated_total



if __name__ == "__main__":
    scenarios = [
        ("No new demand", None),
        ("Ward 9 demand surge", 9),
        ("Ward 5 demand surge", 5),
    ]

    print("DEMAND-RESPONSIVE ALLOCATION")
    print("--------------------------------")

    for name, surge_ward in scenarios:
        env = ResourceAllocationEnv()

        if surge_ward is not None:
            env.demand[surge_ward]["water_litres"] += 500

        allocated = demand_responsive_allocate(
            env, budget=300
        )

        print(f"\n{name}")

        for ward in [3, 5, 9]:
            print(
                f"Ward {ward}: "
                f"{env.delivered[ward]['water_litres']} L"
            )

        print("Total allocated:", allocated)

        assert allocated <= 300

    print("\nSUCCESS! Demand-responsive policy tested.")
