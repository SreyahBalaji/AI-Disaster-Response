
from resource_allocation_env import ResourceAllocationEnv


def calculate_priority_reward(
    allocated_quantity,
    fulfilment_before
):
    if allocated_quantity < 0:
        raise ValueError("Allocation cannot be negative")

    if not 0 <= fulfilment_before <= 1:
        raise ValueError("Invalid fulfilment rate")

    if allocated_quantity == 0:
        return -1.0

    # Larger reward for serving underserved wards.
    priority_weight = 1.0 + 2.0 * (
        1.0 - fulfilment_before
    )

    delivery_reward = allocated_quantity / 10.0

    return delivery_reward * priority_weight


if __name__ == "__main__":
    print("PRIORITY REWARD TEST")
    print("-----------------------------")

    underserved = calculate_priority_reward(
        10, 0.0
    )

    partially_served = calculate_priority_reward(
        10, 0.5
    )

    well_served = calculate_priority_reward(
        10, 0.9
    )

    print("Unserved ward:", underserved)
    print("50% fulfilled ward:", partially_served)
    print("90% fulfilled ward:", well_served)

    assert underserved > partially_served > well_served

    print("\nSUCCESS! Priority reward works.")
