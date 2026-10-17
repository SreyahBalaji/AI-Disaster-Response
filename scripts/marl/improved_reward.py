
from resource_allocation_env import ResourceAllocationEnv
from fairness_metrics import calculate_fairness


def calculate_improved_reward(
    allocated_quantity,
    fairness_before,
    fairness_after
):
    if allocated_quantity < 0:
        raise ValueError("Allocation cannot be negative")

    # Positive reward for delivering water.
    delivery_reward = allocated_quantity / 10.0

    # Reward improvement in fairness.
    fairness_improvement = (
        fairness_before - fairness_after
    )

    fairness_reward = 5.0 * fairness_improvement

    # Penalize actions that deliver nothing.
    idle_penalty = 1.0 if allocated_quantity == 0 else 0.0

    return (
        delivery_reward
        + fairness_reward
        - idle_penalty
    )


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    before = calculate_fairness(
        env, "water_litres"
    )["fairness_gap"]

    allocated = env.allocate(
        "RC01", 3, "water_litres", 10
    )

    after = calculate_fairness(
        env, "water_litres"
    )["fairness_gap"]

    reward = calculate_improved_reward(
        allocated,
        before,
        after
    )

    print("IMPROVED REWARD TEST")
    print("-----------------------------")
    print("Allocated:", allocated)
    print("Fairness before:", round(before, 4))
    print("Fairness after:", round(after, 4))
    print("Reward:", round(reward, 4))

    assert allocated == 10
    assert -5 <= reward <= 6

    print("\nSUCCESS! Improved reward works.")
