
"""
RESPONDA — Phase 27
Federated GAT-PPO Diagnostic Comparison

Compare trained federated PPO with:
- Fairness baseline
- Untrained PPO
- Centralized GAT-PPO v2

Evaluation only. No retraining or file changes.
"""

from pathlib import Path
from statistics import mean

import torch

from benchmark_gat_models import (
    generate_scenarios,
    simulate,
)
from train_gat_multi_resource_ppo import create_model


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "processed" / "marl"

FEDERATED_PATH = (
    MODEL_DIR
    / "federated_multi_resource"
    / "fedavg_gat_multi_resource_ppo.pth"
)

CENTRALIZED_PATH = (
    MODEL_DIR
    / "aluva_gat_fairness_ppo_v2.pth"
)


def load_model(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    model = create_model()

    model.load_state_dict(
        torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()
    return model


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)

    # Same initialization seed used at the start
    # of the federated training experiment.
    untrained = create_model()
    untrained.eval()

    models = {
        "baseline": None,
        "untrained": untrained,
        "federated": load_model(FEDERATED_PATH),
        "centralized_v2": load_model(CENTRALIZED_PATH),
    }

    scenarios = [
        scenario
        for scenario in generate_scenarios()
        if scenario["dynamic"]
    ]

    print("=" * 68)
    print("PHASE 27 — FEDERATED MODEL DIAGNOSTICS")
    print("=" * 68)

    for name, model in models.items():
        results = [
            simulate(
                scenario,
                name,
                model,
            )
            for scenario in scenarios
        ]

        avg_minimum = mean(
            result["minimum_coverage"]
            for result in results
        )

        avg_coverage = mean(
            result["average_coverage"]
            for result in results
        )

        avg_zero = mean(
            result["zero_coverage_items"]
            for result in results
        )

        print()
        print(name.upper())
        print("-" * 68)

        print(
            f"Average minimum coverage: "
            f"{avg_minimum:.2f}%"
        )

        print(
            f"Average overall coverage: "
            f"{avg_coverage:.2f}%"
        )

        print(
            f"Average unserved categories: "
            f"{avg_zero:.2f}"
        )

        if name == "federated":
            print()
            print("Completely unserved demands:")

            for result in results:
                unserved = [
                    item
                    for item in result["shortages"]
                    if item["coverage_percent"] <= 0.01
                ]

                if not unserved:
                    print(
                        result["scenario"],
                        "None",
                    )
                    continue

                for item in unserved:
                    print(
                        f"{result['scenario']} | "
                        f"Ward {item['ward']} | "
                        f"{item['resource']} | "
                        f"Unmet={item['unmet']}"
                    )

    print()
    print("=" * 68)
    print("Diagnostics completed.")
    print("No models or inventory were modified.")


if __name__ == "__main__":
    main()
