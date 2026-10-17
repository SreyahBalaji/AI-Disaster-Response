
import gymnasium as gym
import numpy as np

from gymnasium import spaces
from resource_allocation_env import ResourceAllocationEnv


class PPOAllocationEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.wards = [3, 5, 9]
        self.centre_id = "RC01"
        self.resource = "water_litres"
        self.max_steps = 50

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        self.env = None
        self.steps = 0

    def _get_observation(self):
        demands = [
            min(
                self.env.demand[ward][self.resource] / 1000.0,
                1.0
            )
            for ward in self.wards
        ]

        fulfilment_ratios = []

        for ward in self.wards:
            delivered = self.env.delivered[ward][self.resource]
            remaining = self.env.demand[ward][self.resource]

            original = delivered + remaining

            ratio = (
                delivered / original
                if original > 0 else 1.0
            )

            fulfilment_ratios.append(ratio)

        centre = next(
            centre for centre in self.env.centres
            if centre["centre_id"] == self.centre_id
        )

        stock = min(
            centre["resources"][self.resource] / 1000.0,
            1.0
        )

        return np.array(
            demands + fulfilment_ratios + [stock],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.env = ResourceAllocationEnv()
        self.steps = 0

        return self._get_observation(), {}

    def step(self, action):
        action = int(action)
        self.steps += 1

        reward = 0.0

        if action == 3:
            reward = -0.1
        else:
            ward = self.wards[action]

            allocated = self.env.allocate(
                self.centre_id,
                ward,
                self.resource,
                10
            )


            reward = allocated / 10.0

            if allocated == 0:
                reward = -0.2
            else:
                # Fairness before the allocation.
                previous_ratios = []

                for w in self.wards:
                    delivered = self.env.delivered[w][self.resource]
                    remaining = self.env.demand[w][self.resource]
                    original = delivered + remaining

                    previous_delivered = delivered

                    if w == ward:
                        previous_delivered -= allocated

                    ratio = (
                        previous_delivered / original
                        if original > 0 else 1.0
                    )

                    previous_ratios.append(ratio)

                previous_gap = (
                    max(previous_ratios) - min(previous_ratios)
                )

                # Fairness after the allocation.
                current_ratios = []

                for w in self.wards:
                    delivered = self.env.delivered[w][self.resource]
                    remaining = self.env.demand[w][self.resource]
                    original = delivered + remaining

                    ratio = (
                        delivered / original
                        if original > 0 else 1.0
                    )

                    current_ratios.append(ratio)

                current_gap = (
                    max(current_ratios) - min(current_ratios)
                )

                # Reward reductions in inequality.
                fairness_improvement = previous_gap - current_gap

                reward += 10.0 * fairness_improvement

        centre = next(
            centre for centre in self.env.centres
            if centre["centre_id"] == self.centre_id
        )

        no_stock = (
            centre["resources"][self.resource] <= 0
        )

        no_demand = all(
            self.env.demand[ward][self.resource] <= 0
            for ward in self.wards
        )

        terminated = no_stock or no_demand
        truncated = self.steps >= self.max_steps

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            {}
        )


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = PPOAllocationEnv()

    check_env(env, skip_render_check=True)

    observation, info = env.reset(seed=42)

    print("FAIRNESS-AWARE PPO ENVIRONMENT TEST")
    print("-----------------------------------")
    print("Observation:", observation)
    print("Observation shape:", observation.shape)
    print("Action space:", env.action_space)

    observation, reward, terminated, truncated, info = (
        env.step(0)
    )

    print("Reward after action:", reward)
    print("New observation:", observation)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    env.close()

    print("\nSUCCESS! Fairness-aware PPO environment validated.")
