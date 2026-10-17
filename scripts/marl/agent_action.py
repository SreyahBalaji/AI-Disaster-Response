
from resource_allocation_env import ResourceAllocationEnv
from agent_observation import get_agent_observation
from priority_reward import calculate_priority_reward


class AgentActionSpace:
    def __init__(self, env):
        self.env = env
        self.ward_numbers = sorted(env.demand)
        self.action_count = len(self.ward_numbers) + 1

    def step(self, centre_id, action):
        if type(action) is not int:
            raise ValueError("Action must be an integer")

        if not 0 <= action < self.action_count:
            raise ValueError("Action outside valid range")

        resource = "water_litres"
        allocated = 0
        selected_ward = None
        fulfilment_before = 0.0

        if action < len(self.ward_numbers):
            selected_ward = self.ward_numbers[action]

            remaining = self.env.demand[selected_ward][resource]
            delivered = self.env.delivered[selected_ward][resource]

            requested = remaining + delivered

            if requested > 0:
                fulfilment_before = delivered / requested

            allocated = self.env.allocate(
                centre_id=centre_id,
                ward_no=selected_ward,
                resource=resource,
                quantity=10
            )

        reward = calculate_priority_reward(
            allocated_quantity=allocated,
            fulfilment_before=fulfilment_before
        )

        observation = get_agent_observation(
            self.env,
            centre_id
        )

        return observation, reward, {
            "ward_no": selected_ward,
            "allocated": allocated
        }


if __name__ == "__main__":
    env = ResourceAllocationEnv()
    actions = AgentActionSpace(env)

    observation, reward, info = actions.step(
        centre_id="RC01",
        action=2
    )

    print("PRIORITY REWARD INTEGRATION TEST")
    print("--------------------------------")
    print("Selected ward:", info["ward_no"])
    print("Allocated water:", info["allocated"])
    print("Reward:", round(reward, 4))
    print("Observation shape:", observation.shape)

    assert info["ward_no"] == 3
    assert info["allocated"] == 10
    assert reward == 3.0
    assert observation.shape == (53,)

    print("\nSUCCESS! Priority reward integrated.")
