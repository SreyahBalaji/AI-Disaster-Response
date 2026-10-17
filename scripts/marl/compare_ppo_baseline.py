
import numpy as np

from resource_allocation_env import ResourceAllocationEnv


env = ResourceAllocationEnv()

wards = [3, 5, 9]
resource = "water_litres"
centre_id = "RC01"

initial_demand = {
    ward: env.demand[ward][resource]
    for ward in wards
}

# Use the same 500 L budget as PPO.
budget = 500

# Allocate proportionally to original demand.
total_demand = sum(initial_demand.values())

targets = {
    ward: budget * initial_demand[ward] / total_demand
    for ward in wards
}

allocations = {
    ward: int(targets[ward])
    for ward in wards
}

# Distribute remaining litres by largest remainder.
remaining = budget - sum(allocations.values())

order = sorted(
    wards,
    key=lambda ward: targets[ward] - allocations[ward],
    reverse=True
)

for ward in order[:remaining]:
    allocations[ward] += 1

actual_deliveries = {}

for ward in wards:
    actual_deliveries[ward] = env.allocate(
        centre_id,
        ward,
        resource,
        allocations[ward]
    )

ratios = [
    actual_deliveries[ward] / initial_demand[ward]
    for ward in wards
]

fairness_gap = max(ratios) - min(ratios)

print("PROPORTIONAL ALLOCATION BASELINE")
print("--------------------------------")
print("Water budget:", budget)
print("Ward deliveries:", actual_deliveries)
print("Total delivered:", sum(actual_deliveries.values()))
print("Fulfilment ratios:", {
    ward: round(
        actual_deliveries[ward] / initial_demand[ward], 4
    )
    for ward in wards
})
print("Fairness gap:", round(fairness_gap, 4))

print("\nPPO 100-episode averages:")
print("Water delivered: 495.4 L")
print("Fairness gap: 0.7316")

print("\nSUCCESS! Baseline comparison completed.")
