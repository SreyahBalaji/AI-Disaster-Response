
from resource_allocation_env import ResourceAllocationEnv
from balanced_allocation_policy import balanced_allocate
from dynamic_demand_update import add_new_demand


def print_status(env, title):
    print(f"\n{title}")
    print("--------------------------------")

    for ward in [3, 5, 9]:
        delivered = env.delivered[ward]["water_litres"]
        remaining = env.demand[ward]["water_litres"]

        print(
            f"Ward {ward}: "
            f"Delivered = {delivered} L, "
            f"Unmet = {remaining} L"
        )


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    print("DYNAMIC RESOURCE REALLOCATION TEST")

    # First allocation cycle
    first_allocated = balanced_allocate(
        env,
        budget=300
    )

    print_status(env, "AFTER INITIAL ALLOCATION")

    # A new disaster report arrives.
    add_new_demand(
        env,
        ward_no=9,
        resource="water_litres",
        quantity=200
    )

    print_status(env, "AFTER NEW DISASTER REPORT")

    # Second allocation cycle
    second_allocated = balanced_allocate(
        env,
        budget=300
    )

    print_status(env, "AFTER REALLOCATION")

    total_delivered = sum(
        resources["water_litres"]
        for resources in env.delivered.values()
    )

    remaining_stock = sum(
        centre["resources"]["water_litres"]
        for centre in env.centres
    )

    print("\nRESOURCE SUMMARY")
    print("--------------------------------")
    print("Initial allocation:", first_allocated, "L")
    print("Additional allocation:", second_allocated, "L")
    print("Total delivered:", total_delivered, "L")
    print("Remaining stock:", remaining_stock, "L")

    assert first_allocated == 300
    assert second_allocated == 300
    assert total_delivered == 600
    assert remaining_stock == 700

    print(
        "\nSUCCESS! Dynamic resource reallocation completed."
    )
