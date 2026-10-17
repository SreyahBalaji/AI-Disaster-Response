
import numpy as np

from resource_allocation_env import ResourceAllocationEnv


def get_agent_observation(env, centre_id):
    centre = next(
        (c for c in env.centres if c["centre_id"] == centre_id),
        None
    )

    if centre is None:
        raise ValueError("Invalid centre ID")

    resource = "water_litres"

    initial_centre = next(
        c for c in env.initial_centres
        if c["centre_id"] == centre_id
    )

    initial_stock = initial_centre["resources"][resource]
    current_stock = centre["resources"][resource]

    stock_ratio = (
        current_stock / initial_stock
        if initial_stock > 0 else 0.0
    )

    observation = [stock_ratio]

    for ward_no in sorted(env.demand):
        remaining = env.demand[ward_no][resource]
        delivered = env.delivered[ward_no][resource]

        requested = remaining + delivered

        if requested > 0:
            remaining_ratio = remaining / requested
            fulfilment_ratio = delivered / requested
        else:
            remaining_ratio = 0.0
            fulfilment_ratio = 1.0

        observation.extend([
            remaining_ratio,
            fulfilment_ratio
        ])

    return np.array(observation, dtype=np.float32)


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    observation = get_agent_observation(env, "RC01")

    print("AGENT OBSERVATION TEST")
    print("-----------------------------")
    print("Observation shape:", observation.shape)
    print("Observation length:", len(observation))
    print("Minimum value:", observation.min())
    print("Maximum value:", observation.max())

    assert observation.shape == (53,)
    assert np.isfinite(observation).all()
    assert np.all(observation >= 0)
    assert np.all(observation <= 1)

    print("\nSUCCESS! Agent observation created.")
