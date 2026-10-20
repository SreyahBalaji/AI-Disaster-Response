
from pathlib import Path

import torch

from gat_informed_multi_resource_env import (
    GATInformedMultiResourceEnv,
)

from train_gat_multi_resource_ppo import (
    create_model,
    tensors_from_env,
    select_baseline_actions,
    CHECKPOINT,
)


def evaluate_method(method, model):
    env = GATInformedMultiResourceEnv()

    env.dynamic_demand_enabled = True
    observations = env.reset()

    successful_actions = 0
    wait_counts = {
        centre: 0
        for centre in env.centre_ids
    }

    while True:
        if method == "baseline":
            actions = select_baseline_actions(env)

        else:
            state, masks = tensors_from_env(
                env, observations
            )

            with torch.no_grad():
                logits = model.actor(state)

                logits = logits.masked_fill(
                    ~masks,
                    torch.finfo(logits.dtype).min,
                )

                selected = logits.argmax(dim=-1)

            actions = {
                centre: int(selected[0, i])
                for i, centre in enumerate(
                    env.centre_ids
                )
            }

        for centre, action in actions.items():
            if action == env.wait_action:
                wait_counts[centre] += 1

        (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = env.step(actions)

        successful_actions += info[
            "successful_actions"
        ]

        if terminated or truncated:
            break

    print()
    print("=" * 72)
    print(f"METHOD: {method.upper()}")
    print("=" * 72)

    print("Simulation steps:", env.steps)
    print("Successful actions:", successful_actions)
    print("Wait actions:", wait_counts)
    print("Fairness gap:", round(
        info["fairness_gap"], 4
    ))

    print()
    print(
        f"{'Ward':<8}"
        f"{'Resource':<23}"
        f"{'Demand':>10}"
        f"{'Delivered':>12}"
        f"{'Unmet':>10}"
        f"{'Coverage':>11}"
    )

    print("-" * 74)

    for ward in env.wards:
        for resource in env.resources:
            delivered = env.env.delivered[
                ward
            ][resource]

            unmet = env.env.demand[
                ward
            ][resource]

            total = delivered + unmet

            if total == 0:
                continue

            coverage = 100 * delivered / total

            print(
                f"{ward:<8}"
                f"{resource:<23}"
                f"{total:>10}"
                f"{delivered:>12}"
                f"{unmet:>10}"
                f"{coverage:>10.1f}%"
            )

    print()
    print("Remaining centre inventory:")

    for centre in env.env.centres:
        print(
            centre["centre_id"],
            centre["resources"],
        )

    print()
    print("Disaster events:")

    for event in env.events:
        if event["type"] == "new_disaster_report":
            print(event)


def main():
    if not Path(CHECKPOINT).exists():
        raise FileNotFoundError(
            f"Trained model missing: {CHECKPOINT}"
        )

    model = create_model()

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    print("PHASE 23 — DYNAMIC ALLOCATION DIAGNOSTICS")

    evaluate_method("baseline", model)
    evaluate_method("gat_ppo", model)

    print()
    print("Diagnostics completed.")


if __name__ == "__main__":
    main()
