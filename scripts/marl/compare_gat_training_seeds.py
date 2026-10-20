
"""
RESPONDA — Multi-seed GAT ablation.

Trains:
  1. Fairness-aware PPO with fixed GAT features
  2. Identical PPO without GAT features

Uses matched training seeds and the same
12 simulated evaluation scenarios.
"""

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from train_gat_multi_resource_ppo import (
    create_model,
    update_model,
)

from train_gat_ppo_v2 import (
    collect_episode as collect_gat,
)

from ablation_no_gat import (
    collect_episode as collect_no_gat,
    evaluate_scenario,
)

from benchmark_gat_models import generate_scenarios


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    ROOT / "data" / "processed" / "marl"
    / "multiseed"
)


def train_model(seed, episodes, use_gat):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = create_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    rng = random.Random(seed)
    trajectories = []

    collector = (
        collect_gat if use_gat
        else collect_no_gat
    )

    for episode in range(1, episodes + 1):
        trajectory, _ = collector(
            model,
            rng,
            dynamic=(episode % 2 == 0),
        )

        trajectories.append(trajectory)

        if episode % 8 == 0 or episode == episodes:
            update_model(
                model,
                optimizer,
                trajectories,
            )
            trajectories.clear()

    model.eval()
    return model


def summarize(rows):
    if not rows:
        return {}

    return {
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
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 19, 42],
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("Episodes must be positive")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Training seeds must be unique")

    torch.set_num_threads(2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = generate_scenarios()

    results = []
    seed_summaries = []

    print("=" * 64)
    print("PHASE 23 — MULTI-SEED GAT ABLATION")
    print("=" * 64)
    print("Training seeds:", args.seeds)
    print("Episodes per model:", args.episodes)
    print("Evaluation scenarios:", len(scenarios))
    print()

    for seed in args.seeds:
        print(f"Training seed {seed}...", flush=True)

        for label, use_gat, method in [
            ("gat", True, "gat_v2"),
            ("no_gat", False, "no_gat"),
        ]:
            model = train_model(
                seed,
                args.episodes,
                use_gat,
            )

            checkpoint = (
                OUTPUT_DIR
                / f"{label}_seed{seed}.pth"
            )

            torch.save(
                model.state_dict(),
                checkpoint,
            )

            current_rows = []

            for scenario in scenarios:
                result = evaluate_scenario(
                    scenario,
                    method,
                    model,
                )

                result["training_seed"] = seed
                result["model"] = label

                current_rows.append(result)
                results.append(result)

            dynamic_rows = [
                row for row in current_rows
                if row["dynamic"]
            ]

            metrics = summarize(dynamic_rows)

            seed_summaries.append({
                "training_seed": seed,
                "model": label,
                **metrics,
            })

            print(
                f"  {label:7} "
                f"Min={metrics['mean_minimum_coverage']:6.2f}% "
                f"Avg={metrics['mean_average_coverage']:6.2f}% "
                f"Zero={metrics['mean_zero_coverage_items']:.2f}",
                flush=True,
            )

    print()
    print("=" * 64)
    print("MULTI-SEED SUMMARY — DYNAMIC SCENARIOS")
    print("=" * 64)

    overall = {}

    for label in ("gat", "no_gat"):
        rows = [
            row for row in results
            if row["model"] == label
            and row["dynamic"]
        ]

        overall[label] = summarize(rows)

        print(
            f"{label:7} "
            f"Min={overall[label]['mean_minimum_coverage']:6.2f}% "
            f"Avg={overall[label]['mean_average_coverage']:6.2f}% "
            f"Zero={overall[label]['mean_zero_coverage_items']:.2f}"
        )

    output = OUTPUT_DIR / "multiseed_comparison.json"

    output.write_text(
        json.dumps(
            {
                "training_seeds": args.seeds,
                "episodes_per_model": args.episodes,
                "scenarios": len(scenarios),
                "summary": overall,
                "per_seed": seed_summaries,
                "results": results,
                "limitations": [
                    "GAT embeddings are fixed.",
                    "All scenarios are simulated.",
                    "These are preliminary experiments.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Results saved:", output)
    print("Existing checkpoints remain unchanged.")


if __name__ == "__main__":
    main()
