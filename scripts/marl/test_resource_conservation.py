
from resource_allocation_env import ResourceAllocationEnv

env = ResourceAllocationEnv()

resource = "water_litres"

initial_stock = sum(
    c["resources"][resource]
    for c in env.centres
)

initial_demand = sum(
    ward[resource]
    for ward in env.demand.values()
)

# Make multiple allocation attempts.
env.allocate("RC01", 3, resource, 200)
env.allocate("RC02", 3, resource, 200)
env.allocate("RC03", 3, resource, 200)

remaining_stock = sum(
    c["resources"][resource]
    for c in env.centres
)

remaining_demand = sum(
    ward[resource]
    for ward in env.demand.values()
)

total_delivered = sum(
    ward[resource]
    for ward in env.delivered.values()
)

assert initial_stock == remaining_stock + total_delivered
assert initial_demand == remaining_demand + total_delivered

print("RESOURCE CONSERVATION TEST")
print("-----------------------------")
print("Initial water stock:", initial_stock)
print("Water delivered:", total_delivered)
print("Remaining stock:", remaining_stock)
print("Remaining demand:", remaining_demand)

print("\nInventory conservation: PASS")
print("Demand conservation: PASS")
print("\nSUCCESS! Resource conservation verified.")
