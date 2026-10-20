
from pathlib import Path

import torch

from benchmark_gat_models import (
    generate_scenarios,
    GATInformedMultiResourceEnv,
)

from train_gat_multi_resource_ppo import (
    create_model,
    tensors_from_env,
)


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    ROOT
    / "data"
    / "processed"
    / "marl"
    / "federated_multi_resource"
    / "fedavg_gat_multi_resource_ppo.pth"
)

TARGET_WARD = 3
TARGET_RESOURCE = "water_litres"


def diagnose(scenario, model):
    # Use the same environment as the benchmark.
    env = GATInformedMultiResourceEnv()

    env.dynamic_demand_enabled = scenario["dynamic"]
    env.surge_ward = scenario["surge_ward"]
    env.surge_step = scenario["surge_step"]
    env.surge_resources = scenario["surge_resources"].copy()

    env.reset()

    # Load the exact scenario demand.
    for ward in env.wards:
        for resource in env.resources:
            env.env.demand[ward][resource] = (
                scenario["initial"][ward][resource]
            )

            env.env.delivered[ward][resource] = 0

    observations = env.get_observations()

    action_index = env.action_map.index(
        (TARGET_WARD, TARGET_RESOURCE)
    )

    eligible_steps = 0
    selected_actions = 0

    initial_demand = env.env.demand[
        TARGET_WARD
    ][TARGET_RESOURCE]

    while True:
        state, masks = tensors_from_env(
            env,
            observations,
        )

        # Check whether Ward 3 water can be allocated.
        if any(
            env.get_action_masks()[centre][action_index]
            for centre in env.centre_ids
        ):
            eligible_steps += 1

        # GREEDY ACTION SELECTION:
        # Identical to the benchmark's policy selection.
        with torch.no_grad():
            logits = model.actor(state)

            logits = logits.masked_fill(
                ~masks,
                torch.finfo(logits.dtype).min,
            )

            actions = logits.argmax(dim=-1)

        action_dict = {
            centre: int(actions[0, i].item())
            for i, centre in enumerate(env.centre_ids)
        }

        selected_actions += sum(
            action == action_index
            for action in action_dict.values()
        )

        (
            observations,
            _,
            terminated,
            truncated,
            _,
        ) = env.step(action_dict)

        if terminated or truncated:
            break

    delivered = env.env.delivered[
        TARGET_WARD
    ][TARGET_RESOURCE]

    unmet = env.env.demand[
        TARGET_WARD
    ][TARGET_RESOURCE]

    remaining_stock = sum(
        centre["resources"][TARGET_RESOURCE]
        for centre in env.env.centres
        if centre["centre_id"] in env.centre_ids
    )

    total_demand = delivered + unmet

    coverage = (
        100.0 * delivered / total_demand
        if total_demand > 0
        else 100.0
    )

    print(f"\nScenario: {scenario['id']}")
    print(f"Initial Ward 3 water demand: {initial_demand}")
    print(f"Eligible steps: {eligible_steps}")
    print(f"Ward 3 water actions selected: {selected_actions}")
    print(f"Water delivered to Ward 3: {delivered}")
    print(f"Unmet Ward 3 water demand: {unmet}")
    print(f"Remaining water stock: {remaining_stock}")
    print(f"Ward 3 water coverage: {coverage:.2f}%")
    print(f"Simulation steps: {env.steps}")


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    model = create_model()

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    print("=" * 60)
    print("WARD 3 WATER ALLOCATION DIAGNOSTIC")
    print("ACTION SELECTION: GREEDY")
    print("=" * 60)

    scenarios = [
        scenario
        for scenario in generate_scenarios()
        if scenario["dynamic"]
    ]

    for scenario in scenarios:
        diagnose(scenario, model)

    print("\nDiagnostic completed.")
    print("No checkpoints or saved inventory modified.")


if __name__ == "__main__":
    main()
