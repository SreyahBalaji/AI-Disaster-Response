
"""
RESPONDA — GAT-PPO Research Benchmark

Compare:
1. Fairness heuristic baseline
2. Original GAT-PPO v1
3. Fairness-trained GAT-PPO v2

Uses 12 matched scenarios generated from
3 scenario seeds, with static and dynamic demand.

This is a multi-scenario evaluation, NOT a
multiple-training-seed experiment.
"""

import json
import random
from pathlib import Path
from statistics import mean

import torch

from gat_informed_multi_resource_env import (
    GATInformedMultiResourceEnv,
)

from train_gat_multi_resource_ppo import (
    create_model,
    tensors_from_env,
    select_baseline_actions,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "processed" / "marl"

MODELS = {
    "gat_ppo_v1": (
        MODEL_DIR / "aluva_gat_multi_resource_ppo.pth"
    ),
    "gat_ppo_v2": (
        MODEL_DIR / "aluva_gat_fairness_ppo_v2.pth"
    ),
}

OUTPUT = MODEL_DIR / "gat_ppo_research_benchmark.json"


def load_models():
    loaded = {}

    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}"
            )

        model = create_model()

        model.load_state_dict(
            torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        )

        model.eval()
        loaded[name] = model

        print(f"Loaded: {name}")

    return loaded


def generate_scenarios():
    template = GATInformedMultiResourceEnv()
    template.reset()

    original = {
        ward: {
            resource: int(
                template.env.demand[ward][resource]
            )
            for resource in template.resources
        }
        for ward in template.wards
    }

    scenarios = []

    for seed in [7, 19, 41]:
        rng = random.Random(seed)

        for case_number in range(4):
            # Cases 1 and 3 are static.
            # Cases 2 and 4 introduce new demand.
            dynamic = case_number % 2 == 1

            initial = {}

            for ward in template.wards:
                initial[ward] = {}

                for resource in template.resources:
                    base = original[ward][resource]

                    initial[ward][resource] = (
                        max(
                            1,
                            round(
                                base * rng.uniform(
                                    0.75, 1.35
                                )
                            ),
                        )
                        if base > 0
                        else 0
                    )

            surge = {
                "water_litres": round(
                    500 * rng.uniform(0.6, 1.4)
                ),
                "food_meals": round(
                    100 * rng.uniform(0.6, 1.4)
                ),
                "rescue_cases": max(
                    1,
                    round(
                        5 * rng.uniform(0.6, 1.4)
                    ),
                ),
                "shelter_households": max(
                    1,
                    round(
                        20 * rng.uniform(0.6, 1.4)
                    ),
                ),
                "medical_cases": max(
                    1,
                    round(
                        15 * rng.uniform(0.6, 1.4)
                    ),
                ),
            }

            scenarios.append({
                "id": f"S{seed}-{case_number + 1}",
                "seed": seed,
                "dynamic": dynamic,
                "initial": initial,
                "surge_ward": rng.choice(
                    template.wards
                ),
                "surge_step": 10,
                "surge_resources": surge,
            })

    return scenarios


def simulate(scenario, method, model=None):
    env = GATInformedMultiResourceEnv()

    env.dynamic_demand_enabled = scenario["dynamic"]
    env.surge_ward = scenario["surge_ward"]
    env.surge_step = scenario["surge_step"]
    env.surge_resources = scenario["surge_resources"].copy()

    env.reset()

    for ward in env.wards:
        for resource in env.resources:
            env.env.demand[ward][resource] = (
                scenario["initial"][ward][resource]
            )

            env.env.delivered[ward][resource] = 0

    observations = env.get_observations()

    while True:
        if method == "baseline":
            actions = select_baseline_actions(env)

        else:
            state, masks = tensors_from_env(
                env,
                observations,
            )

            with torch.no_grad():
                logits = model.actor(state)

                logits = logits.masked_fill(
                    ~masks,
                    torch.finfo(logits.dtype).min,
                )

                chosen = logits.argmax(dim=-1)

            actions = {
                centre: int(
                    chosen[0, index].item()
                )
                for index, centre in enumerate(
                    env.centre_ids
                )
            }

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
    shortages = []

    for ward in env.wards:
        for resource in env.resources:
            delivered = int(
                env.env.delivered[ward][resource]
            )

            remaining = int(
                env.env.demand[ward][resource]
            )

            total = delivered + remaining

            if total <= 0:
                continue

            coverage = (
                100.0 * delivered / total
            )

            coverage_values.append(coverage)

            if remaining > 0:
                shortages.append({
                    "ward": ward,
                    "resource": resource,
                    "unmet": remaining,
                    "coverage_percent": round(
                        coverage, 2
                    ),
                })

    return {
        "scenario": scenario["id"],
        "dynamic": scenario["dynamic"],
        "method": method,
        "minimum_coverage": round(
            min(coverage_values), 2
        ),
        "average_coverage": round(
            mean(coverage_values), 2
        ),
        "fully_satisfied_items": sum(
            value >= 100 - 1e-6
            for value in coverage_values
        ),
        "zero_coverage_items": sum(
            value <= 1e-6
            for value in coverage_values
        ),
        "total_demand_items": len(
            coverage_values
        ),
        "unmet_water_litres": sum(
            env.env.demand[ward]["water_litres"]
            for ward in env.wards
        ),
        "unmet_rescue_cases": sum(
            env.env.demand[ward]["rescue_cases"]
            for ward in env.wards
        ),
        "steps": env.steps,
        "shortages": shortages,
    }


def summarize(results):
    summary = {}

    for dynamic in [False, True]:
        scenario_name = (
            "DYNAMIC"
            if dynamic
            else "STATIC"
        )

        summary[scenario_name] = {}

        for method in [
            "baseline",
            "gat_ppo_v1",
            "gat_ppo_v2",
        ]:
            rows = [
                row
                for row in results
                if (
                    row["dynamic"] == dynamic
                    and row["method"] == method
                )
            ]

            summary[scenario_name][method] = {
                "scenarios": len(rows),
                "mean_minimum_coverage": round(
                    mean(
                        row["minimum_coverage"]
                        for row in rows
                    ),
                    2,
                ),
                "mean_average_coverage": round(
                    mean(
                        row["average_coverage"]
                        for row in rows
                    ),
                    2,
                ),
                "mean_zero_coverage_items": round(
                    mean(
                        row["zero_coverage_items"]
                        for row in rows
                    ),
                    2,
                ),
                "mean_unmet_water_litres": round(
                    mean(
                        row["unmet_water_litres"]
                        for row in rows
                    ),
                    2,
                ),
            }

    return summary


def main():
    torch.set_num_threads(2)

    models = load_models()
    scenarios = generate_scenarios()

    results = []

    print()
    print("=" * 72)
    print("PHASE 23 — MULTI-SCENARIO RESEARCH EVALUATION")
    print("=" * 72)

    for scenario in scenarios:
        for method in [
            "baseline",
            "gat_ppo_v1",
            "gat_ppo_v2",
        ]:
            result = simulate(
                scenario,
                method,
                models.get(method),
            )

            results.append(result)

            print(
                f"{result['scenario']:8} "
                f"{method:11} "
                f"Min={result['minimum_coverage']:6.2f}% "
                f"Avg={result['average_coverage']:6.2f}%"
            )

    summary = summarize(results)

    print()
    print("=" * 72)
    print("SUMMARY — MATCHED DISASTER SCENARIOS")
    print("=" * 72)

    for scenario_name, methods in summary.items():
        print()
        print(scenario_name)

        for method, values in methods.items():
            print(
                f"{method:11} "
                f"Min={values['mean_minimum_coverage']:6.2f}% "
                f"Avg={values['mean_average_coverage']:6.2f}% "
                f"ZeroCoverage="
                f"{values['mean_zero_coverage_items']:.2f}"
            )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            {
                "description": (
                    "Exploratory 12-scenario comparison "
                    "with three scenario-generation seeds. "
                    "Model training seed is unchanged."
                ),
                "scenario_count": len(scenarios),
                "summary": summary,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved results to:", OUTPUT)
    print(
        "Research note: Multiple training seeds "
        "and independent held-out datasets "
        "are still required."
    )


if __name__ == "__main__":
    main()
