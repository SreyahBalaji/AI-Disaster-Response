
from pathlib import Path

import numpy as np
import torch

from multi_resource_cooperative_env import MultiResourceCooperativeEnv
from multi_resource_ppo_network import MultiResourceCooperativePPO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "data/processed/marl/aluva_multi_resource_masked_ppo.pth"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )


# ============================================================
# INITIALIZE MODEL
# ============================================================

env = MultiResourceCooperativeEnv()

model = MultiResourceCooperativePPO(
    observation_size=env.observation_size,
    action_size=env.action_size,
    num_agents=len(env.centre_ids),
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()


# ============================================================
# RUN EVALUATION
# ============================================================

def run_scenario(dynamic=False):

    env.dynamic_demand_enabled = dynamic
    observations = env.reset()

    total_successful_actions = 0

    while True:

        # Convert observations to a model input tensor.
        state = torch.tensor(
            np.stack([
                observations[centre_id]
                for centre_id in env.centre_ids
            ]),
            dtype=torch.float32
        ).unsqueeze(0)

        # Retrieve valid actions for each relief centre.
        masks = env.get_action_masks()

        action_masks = torch.tensor(
            np.stack([
                masks[centre_id]
                for centre_id in env.centre_ids
            ]),
            dtype=torch.bool
        ).unsqueeze(0)

        # Predict actions using the trained PPO actor.
        with torch.no_grad():

            logits = model.actor(state)

            # Prevent selection of invalid actions.
            logits = logits.masked_fill(
                ~action_masks,
                torch.finfo(logits.dtype).min
            )

            # Deterministic policy for reproducible evaluation.
            actions = torch.argmax(
                logits,
                dim=-1
            )

        action_dict = {
            centre_id: int(actions[0, index].item())
            for index, centre_id in enumerate(env.centre_ids)
        }

        observations, rewards, terminated, truncated, info = (
            env.step(action_dict)
        )

        total_successful_actions += info[
            "successful_actions"
        ]

        if terminated or truncated:
            break

    # ========================================================
    # PRINT SCENARIO RESULTS
    # ========================================================

    scenario_name = (
        "DYNAMIC DISASTER SCENARIO"
        if dynamic
        else "STATIC DISASTER SCENARIO"
    )

    print("\n" + "=" * 55)
    print(scenario_name)
    print("=" * 55)

    print("Simulation steps:", env.steps)

    print(
        "Fairness gap:",
        round(info["fairness_gap"], 4)
    )

    print(
        "Vulnerability-weighted unmet ratio:",
        round(
            info["vulnerability_weighted_unmet_ratio"],
            4
        )
    )

    print(
        "Successful allocation actions:",
        total_successful_actions
    )

    print("\nRESOURCE ALLOCATION RESULTS")
    print("-" * 55)

    for resource in env.resources:

        total_demand = sum(
            env.env.demand[ward][resource]
            + env.env.delivered[ward][resource]
            for ward in env.wards
        )

        total_delivered = sum(
            env.env.delivered[ward][resource]
            for ward in env.wards
        )

        total_unmet = sum(
            env.env.demand[ward][resource]
            for ward in env.wards
        )

        satisfaction = (
            100.0 * total_delivered / total_demand
            if total_demand > 0
            else None
        )

        print(f"\nResource: {resource}")
        print(f"  Total demand: {total_demand}")
        print(f"  Delivered: {total_delivered}")
        print(f"  Unmet demand: {total_unmet}")

        if satisfaction is None:
            print("  Satisfaction: N/A (no demand)")
        else:
            print(f"  Satisfaction: {satisfaction:.2f}%")

    # ========================================================
    # WARD-LEVEL RESULTS
    # ========================================================

    print("\nWARD-LEVEL SATISFACTION")
    print("-" * 55)

    for ward in env.wards:

        print(f"\nWard {ward}")

        for resource in env.resources:

            delivered = env.env.delivered[ward][resource]
            remaining = env.env.demand[ward][resource]
            demand = delivered + remaining

            if demand > 0:
                satisfaction = 100 * delivered / demand

                print(
                    f"  {resource}: {satisfaction:.2f}%"
                )

    # ========================================================
    # EVENT SUMMARY
    # ========================================================

    allocation_events = [
        event
        for event in env.events
        if event["type"] == "allocation"
    ]

    disaster_events = [
        event
        for event in env.events
        if event["type"] == "new_disaster_report"
    ]

    print("\nEVENT SUMMARY")
    print("-" * 55)

    print("Allocation events:", len(allocation_events))
    print("New disaster reports:", len(disaster_events))

    for event in disaster_events:
        print(
            f"  Step {event['step']}: "
            f"New demand reported in Ward {event['ward_no']}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("MULTI-RESOURCE MASKED PPO EVALUATION")
    print("Model:", MODEL_PATH)

    run_scenario(dynamic=False)
    run_scenario(dynamic=True)

    print(
        "\nSUCCESS: Masked multi-resource PPO evaluation completed."
    )
