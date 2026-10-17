
from resource_allocation_env import ResourceAllocationEnv
from balanced_allocation_policy import balanced_allocate


def add_new_demand(env, ward_no, resource, quantity):
    if ward_no not in env.demand:
        raise ValueError(f"Unknown ward: {ward_no}")

    if resource not in env.demand[ward_no]:
        raise ValueError(f"Unknown resource: {resource}")

    if quantity <= 0:
        raise ValueError("New demand must be positive")

    env.demand[ward_no][resource] += quantity

    print(
        f"New report: Ward {ward_no} requires "
        f"{quantity} additional units of {resource}"
    )


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    # Initial allocation
    balanced_allocate(env, budget=300)

    resource = "water_litres"

    print("DYNAMIC DEMAND UPDATE TEST")
    print("--------------------------------")

    print("BEFORE NEW REPORT")
    for ward in [3, 5, 9]:
        print(
            f"Ward {ward}: "
            f"{env.demand[ward][resource]} L unmet demand"
        )

    # Simulate a new report from Ward 9.
    add_new_demand(
        env,
        ward_no=9,
        resource=resource,
        quantity=200
    )

    print("\nAFTER NEW REPORT")
    for ward in [3, 5, 9]:
        print(
            f"Ward {ward}: "
            f"{env.demand[ward][resource]} L unmet demand"
        )

    assert env.demand[9][resource] == 366
    assert env.delivered[9][resource] == 50

    print("\nSUCCESS! Dynamic demand updated.")
