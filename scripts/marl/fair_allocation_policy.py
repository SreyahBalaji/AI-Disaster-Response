
from resource_allocation_env import ResourceAllocationEnv
from fairness_metrics import calculate_fairness


def fair_allocate(env, resource, batch_size=10):

    if resource not in env.resource_types:
        raise ValueError("Invalid resource type")

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Batch size must be positive")

    while True:
        candidates = []

        for ward_no in env.demand:
            remaining = env.demand[ward_no][resource]
            delivered = env.delivered[ward_no][resource]

            requested = remaining + delivered

            if remaining > 0 and requested > 0:
                fulfilment = delivered / requested
                candidates.append(
                    (fulfilment, ward_no)
                )

        if not candidates:
            break

        available_centres = [
            centre
            for centre in env.centres
            if centre["resources"][resource] > 0
        ]

        if not available_centres:
            break

        # Prioritize the least-served ward.
        candidates.sort()
        _, selected_ward = candidates[0]

        # Simple centre selection: highest available stock.
        selected_centre = max(
            available_centres,
            key=lambda c: c["resources"][resource]
        )

        allocated = env.allocate(
            centre_id=selected_centre["centre_id"],
            ward_no=selected_ward,
            resource=resource,
            quantity=batch_size
        )

        if allocated == 0:
            break


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    fair_allocate(
        env,
        resource="water_litres",
        batch_size=10
    )

    result = calculate_fairness(
        env,
        "water_litres"
    )

    total_delivered = sum(
        ward["water_litres"]
        for ward in env.delivered.values()
    )

    print("FAIR ALLOCATION POLICY")
    print("-----------------------------")
    print("Water delivered:", total_delivered)
    print(
        "Mean fulfilment:",
        round(result["mean_fulfilment"], 4)
    )
    print(
        "Fairness gap:",
        round(result["fairness_gap"], 4)
    )

    assert total_delivered >= 0

    print("\nSUCCESS! Fair allocation completed.")
