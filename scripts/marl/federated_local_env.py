
from cooperative_allocation_env import CooperativeAllocationEnv


class FederatedLocalEnv:
    """
    Simulated local training environment for one
    federated-learning participant.

    Each participant runs a separate cooperative
    allocation simulation with different demand.
    """

    SCENARIOS = {
        "RC01": {
            "surge_ward": None,
            "surge_quantity": 0
        },
        "RC02": {
            "surge_ward": 9,
            "surge_quantity": 500
        },
        "RC03": {
            "surge_ward": 5,
            "surge_quantity": 500
        }
    }

    def __init__(self, centre_id):
        if centre_id not in self.SCENARIOS:
            raise ValueError(
                f"Unknown relief centre: {centre_id}"
            )

        self.centre_id = centre_id
        self.scenario = self.SCENARIOS[centre_id]

        self.env = CooperativeAllocationEnv()
        self.env.dynamic_demand_enabled = False

        self.steps = 0
        self.surge_applied = False

    def reset(self):
        self.steps = 0
        self.surge_applied = False
        return self.env.reset()

    def step(self, actions):
        self.steps += 1

        observations, rewards, terminated, truncated, info = (
            self.env.step(actions)
        )

        # Apply the local disaster surge at step 10.
        if (
            self.steps == 10
            and self.scenario["surge_ward"] is not None
        ):
            ward = self.scenario["surge_ward"]
            quantity = self.scenario["surge_quantity"]

            self.env.env.demand[ward]["water_litres"] += quantity

            self.surge_applied = True

            # Refresh observations so the agents see
            # the new demand on the next decision.
            observations = self.env.get_observations()

        return (
            observations,
            rewards,
            terminated,
            truncated,
            info
        )


if __name__ == "__main__":
    print("FEDERATED LOCAL ENVIRONMENT TEST")
    print("--------------------------------")

    for centre_id in ["RC01", "RC02", "RC03"]:
        local_env = FederatedLocalEnv(centre_id)

        observations = local_env.reset()

        print(f"\n{centre_id}: initialized")
        print("Observation shape:", observations[centre_id].shape)

        for _ in range(10):
            actions = {
                agent_id: 3
                for agent_id in local_env.env.centre_ids
            }

            observations, rewards, done, truncated, info = (
                local_env.step(actions)
            )

            if done or truncated:
                break

        print("Surge applied:", local_env.surge_applied)

        if local_env.surge_applied:
            ward = local_env.scenario["surge_ward"]
            print(
                f"Ward {ward} remaining demand:",
                local_env.env.env.demand[ward]["water_litres"]
            )

    print(
        "\nSUCCESS! Federated local environments validated."
    )
