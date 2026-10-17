
from resource_allocation_env import ResourceAllocationEnv
from relief_centre_agent import ReliefCentreAgent

env = ResourceAllocationEnv()
env.reset()

print("MULTI-AGENT ALLOCATION TEST")
print("----------------------------")

for centre_id in ["RC01", "RC02", "RC03"]:
    agent = ReliefCentreAgent(centre_id)
    print(f"{centre_id} initialized")

print("\nAll three relief centres initialized successfully.")
print("\nSHARED WATER ALLOCATION TEST")
print("----------------------------")

for centre_id in ["RC01", "RC02", "RC03"]:
    allocated = env.allocate(centre_id, 3, "water_litres", 50)
    print(f"{centre_id} allocated: {allocated} litres")

state = env.get_state()

remaining = state["remaining_demand"][3]["water_litres"]
delivered = state["delivered"][3]["water_litres"]

print("\nWard 3 remaining water demand:", remaining)
print("Ward 3 total water delivered:", delivered)

assert delivered == 150

print("\nSUCCESS! Shared allocation verified.")

print("\nRESOURCE CONSERVATION TEST")
print("----------------------------")

initial_stock = sum(
    centre["resources"]["water_litres"]
    for centre in env.initial_centres
)

remaining_stock = sum(
    centre["resources"]["water_litres"]
    for centre in env.centres
)

total_delivered = sum(
    resources["water_litres"]
    for resources in env.delivered.values()
)

print("Initial water stock:", initial_stock)
print("Remaining water stock:", remaining_stock)
print("Total water delivered:", total_delivered)

assert initial_stock == remaining_stock + total_delivered

print("\nSUCCESS! Resource conservation verified.")
