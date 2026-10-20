
from pathlib import Path
from statistics import mean

import torch

from benchmark_gat_models import (
    GATInformedMultiResourceEnv,
    generate_scenarios,
    simulate,
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

COVERAGE_FLOOR = 0.10


def choose_guarded_actions(env, model, observations):
    state, masks = tensors_from_env(
        env,
        observations,
    )

    with torch.no_grad():
        logits = model.actor(state)
        masked_logits = logits.masked_fill(
            ~masks,
            torch.finfo(logits.dtype).min,
        )
        greedy_actions = masked_logits.argmax(dim=-1)

    selected = {}

    for agent_index, centre_id in enumerate(
        env.centre_ids
    ):
        candidates = []

        for action_index, (ward, resource) in enumerate(
            env.action_map
        ):
            if not bool(
                masks[0, agent_index, action_index].item()
            ):
                continue

            delivered = env.env.delivered[
                ward
            ][resource]

            remaining = env.env.demand[
                ward
            ][resource]

            total = delivered + remaining

            if total <= 0:
                continue

            coverage = delivered / total

            if coverage < COVERAGE_FLOOR:
                candidates.append((
                    coverage,
                    -float(env.vulnerability[ward]),
                    -float(
                        logits[
                            0,
                            agent_index,
                            action_index,
                        ].item()
                    ),
                    action_index,
                ))

        if candidates:
            # Prioritize lowest coverage, then
            # vulnerability, then PPO preference.
            selected[centre_id] = min(
                candidates
            )[-1]
        else:
            selected[centre_id] = int(
                greedy_actions[
                    0,
                    agent_index,
                ].item()
            )

    return selected


def run_guarded_scenario(scenario, model):
    env = GATInformedMultiResourceEnv()

    env.dynamic_demand_enabled = scenario["dynamic"]
    env.surge_ward = scenario["surge_ward"]
    env.surge_step = scenario["surge_step"]
    env.surge_resources = scenario[
        "surge_resources"
    ].copy()

    env.reset()

    for ward in env.wards:
        for resource in env.resources:
            env.env.demand[ward][resource] = (
                scenario["initial"][ward][resource]
            )
            env.env.delivered[ward][resource] = 0

    observations = env.get_observations()

    while True:
        actions = choose_guarded_actions(
            env,
            model,
            observations,
        )

        (
            observations,
            _,
            terminated,
            truncated,
            _,
        ) = env.step(actions)

        if terminated or truncated:
            break

    coverage_values = []

    for ward in env.wards:
        for resource in env.resources:
            delivered = env.env.delivered[
                ward
            ][resource]

            remaining = env.env.demand[
                ward
            ][resource]

            total = delivered + remaining

            if total > 0:
                coverage_values.append(
                    100.0 * delivered / total
                )

    delivered_water = env.env.delivered[
        3
    ]["water_litres"]

    unmet_water = env.env.demand[
        3
    ]["water_litres"]

    return {
        "minimum_coverage": min(coverage_values),
        "average_coverage": mean(coverage_values),
        "zero_coverage_items": sum(
            value <= 1e-6
            for value in coverage_values
        ),
        "ward3_water_delivered": delivered_water,
        "ward3_water_unmet": unmet_water,
    }


def main():
    model = create_model()

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    scenarios = [
        scenario
        for scenario in generate_scenarios()
        if scenario["dynamic"]
    ]

    summary = {
        "baseline": [],
        "federated_greedy": [],
        "federated_guarded": [],
    }

    print("=" * 72)
    print("PHASE 27 — FAIRNESS GUARDRAIL EVALUATION")
    print("=" * 72)

    for scenario in scenarios:
        baseline = simulate(
            scenario,
            "baseline",
        )

        greedy = simulate(
            scenario,
            "fedavg_gat_ppo",
            model,
        )

        guarded = run_guarded_scenario(
            scenario,
            model,
        )

        summary["baseline"].append(baseline)
        summary["federated_greedy"].append(greedy)
        summary["federated_guarded"].append(guarded)

        print(
            f"{scenario['id']} | "
            f"Greedy min={greedy['minimum_coverage']:.2f}% | "
            f"Guarded min={guarded['minimum_coverage']:.2f}% | "
            f"Ward 3 water delivered="
            f"{guarded['ward3_water_delivered']}"
        )

    print("\nFINAL COMPARISON")
    print("-" * 72)

    for name, results in summary.items():
        minimum = mean(
            row["minimum_coverage"]
            for row in results
        )

        average = mean(
            row["average_coverage"]
            for row in results
        )

        zero = mean(
            row["zero_coverage_items"]
            for row in results
        )

        print(
            f"{name:19} "
            f"Min={minimum:6.2f}% "
            f"Avg={average:6.2f}% "
            f"Zero={zero:.2f}"
        )

    print("\nNo model checkpoints were modified.")
    print("No rolling inventory was modified.")


if __name__ == "__main__":
    main()
