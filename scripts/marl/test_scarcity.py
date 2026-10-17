
from resource_allocation_env import ResourceAllocationEnv
from fair_allocation_policy import fair_allocate
from fairness_metrics import calculate_fairness


def create_scarcity_environment():
    env = ResourceAllocationEnv()

    # Hypothetical shortage: 300 litres total.
    for centre in env.centres:
        centre["resources"]["water_litres"] = 100

    return env


def sequential_allocate(env):
    for ward_no in sorted(env.demand):
        for centre in env.centres:
            remaining = env.demand[ward_no]["water_litres"]

            if remaining <= 0:
                break

            env.allocate(
                centre_id=centre["centre_id"],
                ward_no=ward_no,
                resource="water_litres",
                quantity=remaining
            )


def evaluate(name, env):
    result = calculate_fairness(env, "water_litres")

    delivered = sum(
        ward["water_litres"]
        for ward in env.delivered.values()
    )

    print(f"\n{name}")
    print("Water delivered:", delivered)
    print(
        "Mean fulfilment:",
        round(result["mean_fulfilment"], 4)
    )
    print(
        "Fairness gap:",
        round(result["fairness_gap"], 4)
    )

    return result, delivered


if __name__ == "__main__":
    sequential_env = create_scarcity_environment()
    fair_env = create_scarcity_environment()

    sequential_allocate(sequential_env)

    fair_allocate(
        fair_env,
        resource="water_litres",
        batch_size=10
    )

    print("RESOURCE SCARCITY EXPERIMENT")
    print("--------------------------------")
    print("Available water: 300 litres")
    print("Requested water: 996 litres")

    sequential_result, sequential_delivered = evaluate(
        "SEQUENTIAL ALLOCATION",
        sequential_env
    )

    fair_result, fair_delivered = evaluate(
        "FAIR ALLOCATION",
        fair_env
    )

    assert sequential_delivered == fair_delivered == 300

    print("\nCOMPARISON")
    print(
        "Fairness gap improvement:",
        round(
            sequential_result["fairness_gap"]
            - fair_result["fairness_gap"],
            4
        )
    )

    print("\nSUCCESS! Scarcity experiment completed.")
