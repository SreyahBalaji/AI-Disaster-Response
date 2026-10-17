
from resource_allocation_env import ResourceAllocationEnv

RESOURCE = "water_litres"
WARDS = [3, 5, 9]
CENTRES = ["RC01", "RC02", "RC03"]

env = ResourceAllocationEnv()

# Dynamic disaster report:
# Ward 9 needs an additional 500 litres.
env.demand[9][RESOURCE] += 500

initial_demand = {
    ward: env.demand[ward][RESOURCE]
    for ward in WARDS
}

initial_stock = sum(
    centre["resources"][RESOURCE]
    for centre in env.centres
    if centre["centre_id"] in CENTRES
)

# Allocate water in small increments to the ward
# with the lowest current demand-fulfilment ratio.
# This is a centralized, fairness-oriented baseline.
while True:
    remaining_stock = sum(
        centre["resources"][RESOURCE]
        for centre in env.centres
        if centre["centre_id"] in CENTRES
    )

    if remaining_stock <= 0:
        break

    eligible_wards = [
        ward for ward in WARDS
        if env.demand[ward][RESOURCE] > 0
    ]

    if not eligible_wards:
        break

    ward = min(
        eligible_wards,
        key=lambda w: (
            env.delivered[w][RESOURCE] / initial_demand[w],
            w
        )
    )

    allocated = 0

    for centre_id in CENTRES:
        allocated = env.allocate(
            centre_id,
            ward,
            RESOURCE,
            1
        )

        if allocated > 0:
            break

    if allocated <= 0:
        break


deliveries = {
    ward: env.delivered[ward][RESOURCE]
    for ward in WARDS
}

fulfilment = {
    ward: deliveries[ward] / initial_demand[ward]
    for ward in WARDS
}

total_delivered = sum(deliveries.values())
total_demand = sum(initial_demand.values())

fairness_gap = (
    max(fulfilment.values())
    - min(fulfilment.values())
)

print("DYNAMIC MULTI-CENTRE BASELINE")
print("--------------------------------")
print("Initial stock:", initial_stock)
print("Total demand after surge:", total_demand)

print("\nWard allocations:")
for ward in WARDS:
    print(
        f"Ward {ward}: "
        f"{deliveries[ward]} / {initial_demand[ward]} L "
        f"({fulfilment[ward] * 100:.2f}%)"
    )

print("\nBaseline results:")
print("Total delivered:", total_delivered)
print(
    "Demand satisfaction:",
    round(total_delivered / total_demand, 4)
)
print("Fairness gap:", round(fairness_gap, 4))

print("\nExisting Cooperative PPO results:")
print("Average delivered: 1001.50 L")
print("Average fairness gap: 0.6906")
print("Average Ward 9 delivery: 221.50 L")

print("\nSUCCESS! Dynamic baseline comparison completed.")
