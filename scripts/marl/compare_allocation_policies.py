
from resource_allocation_env import ResourceAllocationEnv
from priority_allocation_policy import priority_allocate
from balanced_allocation_policy import balanced_allocate
from fairness_metrics import calculate_fairness


def evaluate(env):
    resource = "water_litres"

    fairness = calculate_fairness(env, resource)

    delivered = {
        ward: env.delivered[ward][resource]
        for ward in [3, 5, 9]
    }

    return {
        "delivered": delivered,
        "total": sum(delivered.values()),
        "fairness_gap": fairness["fairness_gap"]
    }


priority_env = ResourceAllocationEnv()
priority_allocate(
    priority_env,
    resource="water_litres",
    water_budget=300
)

balanced_env = ResourceAllocationEnv()
balanced_allocate(balanced_env, budget=300)

priority = evaluate(priority_env)
balanced = evaluate(balanced_env)

reduction = (
    (priority["fairness_gap"] - balanced["fairness_gap"])
    / priority["fairness_gap"]
    * 100
)

print("ALLOCATION POLICY COMPARISON")
print("--------------------------------")

print("Priority-only allocation:")
print(priority)

print("\nBalanced allocation:")
print(balanced)

print(f"\nFairness gap reduction: {reduction:.2f}%")

assert priority["total"] == 300
assert balanced["total"] == 300
assert balanced["fairness_gap"] < priority["fairness_gap"]

print("\nSUCCESS! Policy comparison completed.")
