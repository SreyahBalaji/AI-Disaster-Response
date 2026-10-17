
from resource_allocation_env import ResourceAllocationEnv


def calculate_fairness(env, resource):
    if resource not in env.resource_types:
        raise ValueError("Invalid resource type")

    fulfilment_rates = {}

    for ward_no in env.demand:
        delivered = env.delivered[ward_no][resource]
        remaining = env.demand[ward_no][resource]

        requested = delivered + remaining

        # Wards with no requests are excluded.
        if requested > 0:
            fulfilment_rates[ward_no] = (
                delivered / requested
            )

    if not fulfilment_rates:
        return {
            "mean_fulfilment": 1.0,
            "fairness_gap": 0.0,
            "wards_with_demand": 0
        }

    rates = list(fulfilment_rates.values())

    return {
        "mean_fulfilment": sum(rates) / len(rates),
        "fairness_gap": max(rates) - min(rates),
        "wards_with_demand": len(rates)
    }


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    # Deliberately allocate water to only one ward.
    env.allocate("RC01", 3, "water_litres", 100)

    result = calculate_fairness(
        env,
        "water_litres"
    )

    print("FAIRNESS METRICS TEST")
    print("-----------------------------")
    print(
        "Wards with water demand:",
        result["wards_with_demand"]
    )
    print(
        "Mean fulfilment:",
        round(result["mean_fulfilment"], 4)
    )
    print(
        "Fairness gap:",
        round(result["fairness_gap"], 4)
    )

    assert 0 <= result["mean_fulfilment"] <= 1
    assert 0 <= result["fairness_gap"] <= 1

    print("\nSUCCESS! Fairness metrics calculated.")
