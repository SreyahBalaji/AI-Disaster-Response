
from resource_allocation_env import ResourceAllocationEnv

env = ResourceAllocationEnv()

print("ALL RESOURCE CONSERVATION TEST")
print("--------------------------------")

for resource in env.resource_types:

    # Reset inventory and demand for each test.
    env.reset()

    initial_stock = sum(
        centre["resources"][resource]
        for centre in env.centres
    )

    initial_demand = sum(
        ward[resource]
        for ward in env.demand.values()
    )

    # Attempt allocations across every ward.
    for ward_no in sorted(env.demand):
        for centre in env.centres:
            env.allocate(
                centre_id=centre["centre_id"],
                ward_no=ward_no,
                resource=resource,
                quantity=1000
            )

    remaining_stock = sum(
        centre["resources"][resource]
        for centre in env.centres
    )

    remaining_demand = sum(
        ward[resource]
        for ward in env.demand.values()
    )

    delivered = sum(
        ward[resource]
        for ward in env.delivered.values()
    )

    assert initial_stock == remaining_stock + delivered
    assert initial_demand == remaining_demand + delivered

    assert remaining_stock >= 0
    assert remaining_demand >= 0

    print(f"{resource}: PASS")
    print(
        f"  Delivered: {delivered} | "
        f"Remaining demand: {remaining_demand}"
    )

print("\nSUCCESS! All five resources validated.")
