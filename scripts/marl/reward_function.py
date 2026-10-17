
from resource_allocation_env import ResourceAllocationEnv
from fairness_metrics import calculate_fairness


def calculate_reward(env, resource, allocated_quantity):
    if resource not in env.resource_types:
        raise ValueError("Invalid resource type")

    if allocated_quantity < 0:
        raise ValueError("Allocated quantity cannot be negative")

    fairness = calculate_fairness(env, resource)

    # Reward successful delivery, scaled by resource units.
    service_reward = min(allocated_quantity / 100, 1.0)

    # Penalize unequal demand fulfilment.
    fairness_penalty = fairness["fairness_gap"]

    # Discourage actions that deliver no resources.
    idle_penalty = 0.5 if allocated_quantity == 0 else 0.0

    reward = (
        service_reward
        - fairness_penalty
        - idle_penalty
    )

    return reward


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    allocated = env.allocate(
        centre_id="RC01",
        ward_no=3,
        resource="water_litres",
        quantity=100
    )

    reward = calculate_reward(
        env,
        resource="water_litres",
        allocated_quantity=allocated
    )

    print("RL REWARD FUNCTION TEST")
    print("-----------------------------")
    print("Allocated water:", allocated)
    print("Calculated reward:", round(reward, 4))

    assert -2 <= reward <= 1

    print("\nSUCCESS! Reward function works.")
