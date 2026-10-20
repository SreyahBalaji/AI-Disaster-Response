
"""
RESPONDA — GAT Feature Impact Test

Use the SAME trained GAT-PPO checkpoint on
matched disaster scenarios.

Test A: Real saved GAT features.
Test B: GAT features replaced with zeros.

This measures model sensitivity to graph inputs.
It does not prove that GAT improves performance.
"""

from pathlib import Path
from statistics import mean

import torch

from ablation_no_gat import evaluate_scenario
from benchmark_gat_models import generate_scenarios
from train_gat_multi_resource_ppo import create_model


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "marl"
    / "aluva_gat_fairness_ppo_v2.pth"
)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    torch.set_num_threads(2)

    model = create_model()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
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

    with_gat = []
    without_gat = []
    changed = 0

    print("=" * 64)
    print("PHASE 23 — GAT INPUT SENSITIVITY")
    print("=" * 64)

    for scenario in scenarios:
        real = evaluate_scenario(
            scenario,
            "gat_v2",
            model,
        )

        zeroed = evaluate_scenario(
            scenario,
            "no_gat",
            model,
        )

        with_gat.append(real)
        without_gat.append(zeroed)

        different = (
            abs(
                real["average_coverage"]
                - zeroed["average_coverage"]
            ) > 0.01
            or abs(
                real["minimum_coverage"]
                - zeroed["minimum_coverage"]
            ) > 0.01
        )

        changed += int(different)

        print(
            f"{scenario['id']:7} "
            f"GAT Min={real['minimum_coverage']:6.2f}% "
            f"Zeroed Min={zeroed['minimum_coverage']:6.2f}% "
            f"Changed={'YES' if different else 'NO'}"
        )

    def averages(rows):
        return {
            "min": mean(
                item["minimum_coverage"]
                for item in rows
            ),
            "avg": mean(
                item["average_coverage"]
                for item in rows
            ),
        }

    real_result = averages(with_gat)
    zeroed_result = averages(without_gat)

    print()
    print("SUMMARY")
    print("-" * 64)

    print(
        "With GAT:    "
        f"Min={real_result['min']:.2f}% "
        f"Avg={real_result['avg']:.2f}%"
    )

    print(
        "GAT zeroed:  "
        f"Min={zeroed_result['min']:.2f}% "
        f"Avg={zeroed_result['avg']:.2f}%"
    )

    print(
        f"Scenarios with different coverage: "
        f"{changed}/{len(scenarios)}"
    )

    print()
    print(
        "Note: Changed results indicate sensitivity "
        "to GAT inputs, not necessarily better decisions."
    )


if __name__ == "__main__":
    main()
