
import numpy as np

from resource_allocation_env import ResourceAllocationEnv


class CooperativeAllocationEnv:
    def __init__(self):
        self.wards = [3, 5, 9]
        self.centre_ids = ["RC01", "RC02", "RC03"]
        self.resource = "water_litres"

        self.max_steps = 50

        # Dynamic disaster demand settings
        self.dynamic_demand_enabled = False
        self.surge_step = 10
        self.surge_ward = 9
        self.surge_quantity = 500

        self.reset()

    def reset(self):
        self.env = ResourceAllocationEnv()
        self.steps = 0

        return self.get_observations()

    def get_observations(self):
        observations = {}

        for centre_id in self.centre_ids:
            centre = next(
                c for c in self.env.centres
                if c["centre_id"] == centre_id
            )

            # Current unmet demand
            demands = [
                self.env.demand[ward][self.resource] / 1000.0
                for ward in self.wards
            ]

            # Percentage of demand already fulfilled
            fulfilment = []

            for ward in self.wards:
                delivered = self.env.delivered[ward][self.resource]
                remaining = self.env.demand[ward][self.resource]
                original = delivered + remaining

                fulfilment.append(
                    delivered / original if original > 0 else 1.0
                )

            # Available stock at this relief centre
            stock = centre["resources"][self.resource] / 1000.0

            observations[centre_id] = np.array(
                demands + fulfilment + [stock],
                dtype=np.float32
            )

        return observations

    def step(self, actions):
        self.steps += 1

        # STEP 1: Introduce new disaster demand
        if (
            self.dynamic_demand_enabled
            and self.steps == self.surge_step
        ):
            self.env.demand[self.surge_ward][self.resource] += (
                self.surge_quantity
            )

            print(
                f"NEW DISASTER REPORT: Ward {self.surge_ward} "
                f"requires {self.surge_quantity} additional litres."
            )

        # STEP 2: Execute actions from all relief centres
        total_delivered_this_step = 0

        for centre_id in self.centre_ids:
            action = int(actions[centre_id])

            # Action 3 means wait
            if action == 3:
                continue

            ward = self.wards[action]

            delivered = self.env.allocate(
                centre_id,
                ward,
                self.resource,
                10
            )

            total_delivered_this_step += delivered

        # STEP 3: Calculate fairness
        ratios = []

        for ward in self.wards:
            delivered = self.env.delivered[ward][self.resource]
            remaining = self.env.demand[ward][self.resource]
            original = delivered + remaining

            ratios.append(
                delivered / original if original > 0 else 1.0
            )

        fairness_gap = max(ratios) - min(ratios)

        # STEP 4: Improved cooperative reward function

        # Reward successful deliveries
        delivery_reward = total_delivered_this_step / 10

        # Stronger penalty for unequal demand fulfilment
        fairness_penalty = 10 * fairness_gap

        # Calculate total unmet demand
        total_unmet_demand = sum(
            self.env.demand[ward][self.resource]
            for ward in self.wards
        )

        # Penalize inactivity when demand remains
        idle_penalty = (
            0.5
            if total_delivered_this_step == 0
            and total_unmet_demand > 0
            else 0.0
        )

        team_reward = (
            delivery_reward
            - fairness_penalty
            - idle_penalty
        )

        # STEP 5: Generate updated observations
        observations = self.get_observations()

        # STEP 6: Check termination conditions
        no_demand = all(
            self.env.demand[ward][self.resource] <= 0
            for ward in self.wards
        )

        no_stock = all(
            centre["resources"][self.resource] <= 0
            for centre in self.env.centres
            if centre["centre_id"] in self.centre_ids
        )

        terminated = no_demand or no_stock
        truncated = self.steps >= self.max_steps

        # All agents receive the same team reward
        rewards = {
            centre_id: team_reward
            for centre_id in self.centre_ids
        }

        info = {
            "fairness_gap": fairness_gap,
            "total_delivered_this_step": total_delivered_this_step
        }

        return (
            observations,
            rewards,
            terminated,
            truncated,
            info
        )


# --------------------------------------------------
# TEST COOPERATIVE MARL ENVIRONMENT
# --------------------------------------------------

if __name__ == "__main__":
    env = CooperativeAllocationEnv()
    observations = env.reset()

    print("COOPERATIVE MARL ENVIRONMENT TEST")
    print("--------------------------------")

    for centre_id, observation in observations.items():
        print(centre_id, observation)

    actions = {
        "RC01": 0,
        "RC02": 1,
        "RC03": 2
    }

    observations, rewards, terminated, truncated, info = (
        env.step(actions)
    )

    print("\nShared rewards:", rewards)
    print("Step information:", info)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    assert len(observations) == 3

    assert all(
        obs.shape == (7,)
        for obs in observations.values()
    )

    assert len(set(rewards.values())) == 1

    assert info["total_delivered_this_step"] == 30

    print("\nSUCCESS! Cooperative MARL environment validated.")

    # --------------------------------------------------
    # TEST DYNAMIC DISASTER DEMAND
    # --------------------------------------------------

    print("\nDYNAMIC DEMAND TEST")
    print("--------------------------------")

    dynamic_env = CooperativeAllocationEnv()
    dynamic_env.dynamic_demand_enabled = True

    for _ in range(10):
        dynamic_env.step({
            "RC01": 3,
            "RC02": 3,
            "RC03": 3
        })

    assert dynamic_env.env.demand[9]["water_litres"] == 716

    print(
        "Ward 9 updated demand:",
        dynamic_env.env.demand[9]["water_litres"]
    )

    print("SUCCESS! Dynamic demand update validated.")
