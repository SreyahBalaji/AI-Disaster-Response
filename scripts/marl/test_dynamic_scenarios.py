
from resource_allocation_env import ResourceAllocationEnv
from balanced_allocation_policy import balanced_allocate
from dynamic_demand_update import add_new_demand


scenarios = [
    ("No new demand", None, 0),
    ("Ward 9 demand surge", 9, 500),
    ("Ward 5 demand surge", 5, 500),
]

print("DYNAMIC ALLOCATION SCENARIO TEST")
print("--------------------------------")

for name, ward_no, new_demand in scenarios:
    env = ResourceAllocationEnv()

    first = balanced_allocate(env, budget=300)

    if ward_no is not None:
        add_new_demand(
            env,
            ward_no=ward_no,
            resource="water_litres",
            quantity=new_demand
        )

    second = balanced_allocate(env, budget=300)

    print(f"\nScenario: {name}")

    for ward in [3, 5, 9]:
        delivered = env.delivered[ward]["water_litres"]
        unmet = env.demand[ward]["water_litres"]

        print(
            f"Ward {ward}: "
            f"Delivered={delivered} L, "
            f"Unmet={unmet} L"
        )

    total_delivered = sum(
        resources["water_litres"]
        for resources in env.delivered.values()
    )

    remaining_stock = sum(
        centre["resources"]["water_litres"]
        for centre in env.centres
    )

    assert first == 300
    assert second == 300
    assert total_delivered + remaining_stock == 1300
    assert all(
        resources["water_litres"] >= 0
        for resources in env.demand.values()
    )

    print("Resource conservation: PASS")

print("\nSUCCESS! All dynamic scenarios tested.")
