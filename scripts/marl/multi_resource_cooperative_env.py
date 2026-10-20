
import numpy as np

from resource_allocation_env import ResourceAllocationEnv


class MultiResourceCooperativeEnv:

    def __init__(self):

        # --------------------------------------------------
        # 1. ENVIRONMENT CONFIGURATION
        # --------------------------------------------------

        self.wards = [3, 5, 9]

        self.centre_ids = [
            "RC01",
            "RC02",
            "RC03"
        ]

        self.resources = [
            "water_litres",
            "food_meals",
            "rescue_cases",
            "shelter_households",
            "medical_cases"
        ]

        # Illustrative vulnerability weights.
        # Replace with validated ward-level data later.
        self.vulnerability = {
            3: 0.6,
            5: 0.7,
            9: 0.9
        }

        self.max_steps = 100

        # Dynamic disaster scenario.
        self.dynamic_demand_enabled = False
        self.surge_step = 10
        self.surge_ward = 9

        self.surge_resources = {
            "water_litres": 500,
            "food_meals": 100,
            "rescue_cases": 5,
            "shelter_households": 20,
            "medical_cases": 15
        }

        # --------------------------------------------------
        # 2. ACTION SPACE
        # --------------------------------------------------

        # 3 wards × 5 resources = 15 allocation actions.
        self.action_map = [
            (ward, resource)
            for ward in self.wards
            for resource in self.resources
        ]

        # Additional WAIT action.
        self.wait_action = len(self.action_map)

        self.action_size = self.wait_action + 1

        # 15 demands + 15 fulfilment ratios
        # + 5 stock values + 3 vulnerability scores.
        self.observation_size = 38

        self.batch_sizes = {
            "water_litres": 10,
            "food_meals": 5,
            "rescue_cases": 1,
            "shelter_households": 1,
            "medical_cases": 1
        }

        self.reset()

    # --------------------------------------------------
    # 3. RESET ENVIRONMENT
    # --------------------------------------------------

    def reset(self):

        self.env = ResourceAllocationEnv()

        self.steps = 0
        self.events = []

        return self.get_observations()

    # --------------------------------------------------
    # 4. GENERATE OBSERVATIONS
    # --------------------------------------------------

    def get_observations(self):

        observations = {}

        for centre_id in self.centre_ids:

            centre = next(
                c for c in self.env.centres
                if c["centre_id"] == centre_id
            )

            demands = []
            fulfilment = []

            for ward in self.wards:

                for resource in self.resources:

                    remaining = self.env.demand[ward][resource]

                    delivered = self.env.delivered[ward][resource]

                    total = remaining + delivered

                    demands.append(
                        np.log1p(max(remaining, 0)) / 10.0
                    )

                    fulfilment.append(
                        delivered / total
                        if total > 0 else 1.0
                    )

            stock = [
                np.log1p(
                    max(centre["resources"][resource], 0)
                ) / 10.0
                for resource in self.resources
            ]

            vulnerability = [
                self.vulnerability[ward]
                for ward in self.wards
            ]

            observations[centre_id] = np.array(
                demands
                + fulfilment
                + stock
                + vulnerability,
                dtype=np.float32
            )

        return observations

    # --------------------------------------------------
    # 5. VALID ACTION MASKING
    # --------------------------------------------------

    def get_action_masks(self):

        masks = {}

        for centre_id in self.centre_ids:

            centre = next(
                c for c in self.env.centres
                if c["centre_id"] == centre_id
            )

            valid_actions = []

            for ward, resource in self.action_map:

                has_demand = (
                    self.env.demand[ward][resource] > 0
                )

                has_stock = (
                    centre["resources"][resource] > 0
                )

                valid_actions.append(
                    has_demand and has_stock
                )

            # WAIT is always a valid action.
            valid_actions.append(True)

            masks[centre_id] = np.array(
                valid_actions,
                dtype=bool
            )

        return masks

    # --------------------------------------------------
    # 6. EXECUTE ALLOCATION ACTIONS
    # --------------------------------------------------

    def step(self, actions):

        self.steps += 1

        delivered_this_step = 0
        successful_actions = 0

        # ----------------------------------------------
        # 6.1 DYNAMIC DISASTER DEMAND
        # ----------------------------------------------

        if (
            self.dynamic_demand_enabled
            and self.steps == self.surge_step
        ):

            for resource, quantity in self.surge_resources.items():

                self.env.demand[
                    self.surge_ward
                ][resource] += quantity

            self.events.append({
                "step": self.steps,
                "type": "new_disaster_report",
                "ward_no": self.surge_ward,
                "additional_demand": self.surge_resources.copy()
            })

        # ----------------------------------------------
        # 6.2 RESOURCE ALLOCATION
        # ----------------------------------------------

        for centre_id in self.centre_ids:

            action = int(actions[centre_id])

            if not 0 <= action < self.action_size:
                raise ValueError(
                    f"Invalid action: {action}"
                )

            if action == self.wait_action:
                continue

            ward, resource = self.action_map[action]

            quantity = self.batch_sizes[resource]

            delivered = self.env.allocate(
                centre_id,
                ward,
                resource,
                quantity
            )

            if delivered > 0:

                successful_actions += 1
                delivered_this_step += delivered

                self.events.append({
                    "step": self.steps,
                    "type": "allocation",
                    "centre_id": centre_id,
                    "ward_no": ward,
                    "resource": resource,
                    "quantity": delivered
                })

        # ----------------------------------------------
        # 6.3 CALCULATE FAIRNESS
        # ----------------------------------------------

        satisfaction = []
        weighted_satisfaction = []

        unmet_items = 0

        for ward in self.wards:

            for resource in self.resources:

                delivered = self.env.delivered[ward][resource]

                remaining = self.env.demand[ward][resource]

                total = delivered + remaining

                if total > 0:

                    ratio = delivered / total

                    satisfaction.append(ratio)

                    weighted_satisfaction.append(
                        (
                            self.vulnerability[ward],
                            ratio
                        )
                    )

                unmet_items += remaining

        fairness_gap = (
            max(satisfaction) - min(satisfaction)
            if satisfaction else 0.0
        )

        vulnerability_gap = (
            sum(
                weight * (1.0 - ratio)
                for weight, ratio in weighted_satisfaction
            )
            / max(
                sum(
                    weight
                    for weight, _ in weighted_satisfaction
                ),
                1e-8
            )
            if weighted_satisfaction else 0.0
        )

        # ----------------------------------------------
        # 6.4 RESOURCE PROGRESS
        # ----------------------------------------------

        resource_progress = 0.0

        for resource in self.resources:

            total_delivered = sum(
                self.env.delivered[ward][resource]
                for ward in self.wards
            )

            total_demand = sum(
                self.env.delivered[ward][resource]
                + self.env.demand[ward][resource]
                for ward in self.wards
            )

            if total_demand > 0:

                resource_progress += (
                    total_delivered / total_demand
                )

        # ----------------------------------------------
        # 6.5 COOPERATIVE TEAM REWARD
        # ----------------------------------------------

        reward = (
            2.0 * successful_actions
            + 5.0 * resource_progress
            - 4.0 * fairness_gap
            - 4.0 * vulnerability_gap
            - (
                1.0
                if successful_actions == 0
                and unmet_items > 0
                else 0.0
            )
        )

        # ----------------------------------------------
        # 6.6 TERMINATION CONDITIONS
        # ----------------------------------------------

        no_demand = all(
            self.env.demand[ward][resource] <= 0
            for ward in self.wards
            for resource in self.resources
        )

        no_stock = all(
            centre["resources"][resource] <= 0
            for centre in self.env.centres
            if centre["centre_id"] in self.centre_ids
            for resource in self.resources
        )

        terminated = no_demand or no_stock

        truncated = self.steps >= self.max_steps

        # ----------------------------------------------
        # 6.7 SHARED TEAM REWARDS
        # ----------------------------------------------

        rewards = {
            centre_id: reward
            for centre_id in self.centre_ids
        }

        # ----------------------------------------------
        # 6.8 STEP INFORMATION
        # ----------------------------------------------

        info = {
            "fairness_gap": fairness_gap,
            "vulnerability_weighted_unmet_ratio": (
                vulnerability_gap
            ),
            "resource_progress": resource_progress,
            "successful_actions": successful_actions,
            "delivered_units_this_step": (
                delivered_this_step
            ),
            "events": self.events.copy()
        }

        return (
            self.get_observations(),
            rewards,
            terminated,
            truncated,
            info
        )


# --------------------------------------------------
# 7. TEST ENVIRONMENT
# --------------------------------------------------

if __name__ == "__main__":

    env = MultiResourceCooperativeEnv()

    observations = env.reset()

    assert env.action_size == 16

    assert env.observation_size == 38

    assert all(
        observation.shape == (38,)
        for observation in observations.values()
    )

    masks = env.get_action_masks()

    assert all(
        len(mask) == 16
        for mask in masks.values()
    )

    print("MULTI-RESOURCE ENVIRONMENT TEST")
    print("--------------------------------")

    print("Relief centres:", env.centre_ids)
    print("Resources:", env.resources)
    print("Actions:", env.action_size)
    print("Observation features:", env.observation_size)

    print(
        "Valid actions:",
        {
            centre_id: int(mask.sum())
            for centre_id, mask in masks.items()
        }
    )

    actions = {
        "RC01": 0,
        "RC02": 6,
        "RC03": 14
    }

    observations, rewards, terminated, truncated, info = (
        env.step(actions)
    )

    print("Rewards:", rewards)

    print(
        "Successful allocations:",
        info["successful_actions"]
    )

    print(
        "Fairness gap:",
        round(info["fairness_gap"], 4)
    )

    print(
        "Resource progress:",
        round(info["resource_progress"], 4)
    )

    print(
        "Vulnerability-weighted unmet ratio:",
        round(
            info["vulnerability_weighted_unmet_ratio"],
            4
        )
    )

    print(
        "\nSUCCESS: Multi-resource environment validated."
    )
