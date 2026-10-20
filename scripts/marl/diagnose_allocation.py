
from multi_resource_cooperative_env import MultiResourceCooperativeEnv


def diagnose():
    env = MultiResourceCooperativeEnv()

    print("=" * 55)
    print("RESOURCE AVAILABILITY AND FAIRNESS DIAGNOSIS")
    print("=" * 55)

    print("\nINITIAL RELIEF CENTRE STOCK")
    print("-" * 55)

    for centre in env.env.centres:
        if centre["centre_id"] not in env.centre_ids:
            continue

        print(f"\n{centre['centre_id']}")

        for resource in env.resources:
            print(
                f"  {resource}: "
                f"{centre['resources'][resource]}"
            )

    print("\nINITIAL WARD DEMAND")
    print("-" * 55)

    for ward in env.wards:
        print(f"\nWard {ward}")

        for resource in env.resources:
            demand = env.env.demand[ward][resource]

            if demand > 0:
                print(f"  {resource}: {demand}")

    print("\nRESOURCE FEASIBILITY")
    print("-" * 55)

    for resource in env.resources:
        total_stock = sum(
            centre["resources"][resource]
            for centre in env.env.centres
            if centre["centre_id"] in env.centre_ids
        )

        total_demand = sum(
            env.env.demand[ward][resource]
            for ward in env.wards
        )

        shortage = max(0, total_demand - total_stock)

        print(f"\n{resource}")
        print(f"  Stock: {total_stock}")
        print(f"  Demand: {total_demand}")
        print(f"  Supply shortage: {shortage}")

    print("\nWARD 3 WATER FEASIBILITY")
    print("-" * 55)

    ward = 3
    resource = "water_litres"

    demand = env.env.demand[ward][resource]

    eligible_centres = [
        centre["centre_id"]
        for centre in env.env.centres
        if (
            centre["centre_id"] in env.centre_ids
            and centre["resources"][resource] > 0
        )
    ]

    print("Ward 3 water demand:", demand)
    print("Centres with water stock:", eligible_centres)

    if demand > 0 and eligible_centres:
        print(
            "RESULT: Ward 3 can receive water under the "
            "current stock-only allocation rules."
        )
    else:
        print(
            "RESULT: Ward 3 has no demand or no centre "
            "currently has water stock."
        )

    print("\nNOTE:")
    print(
        "This checks aggregate stock feasibility, not "
        "travel time, road access, or simultaneous "
        "multi-ward allocation feasibility."
    )

    print("\nSUCCESS: Allocation diagnosis completed.")


if __name__ == "__main__":
    diagnose()
