
"""
RESPONDA — Phase 23
Sparse Dynamic GAT forecasting experiment.

Compare:
1. Persistence
2. Original Dynamic GAT v3
3. Thresholded Dynamic GAT v3

Thresholds are selected ONLY on validation data.
Test data is used once for final evaluation.

Existing checkpoints are never modified.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from build_gat_model import AluvaGAT
from train_dynamic_gat_v3 import forecast


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "processed" / "aluva"

DATA_PATH = BASE / "aluva_forecast_scenarios_v2.pt"
MODEL_PATH = BASE / "aluva_dynamic_gat_v3.pth"

RESULT_PATH = (
    BASE / "aluva_sparse_gat_experiment.json"
)

RESOURCE_NAMES = [
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases",
]

THRESHOLDS = [
    0.0,
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    1.0,
]


def flatten(data, scenario_ids):
    x = data["inputs"][scenario_ids]
    y = data["targets"][scenario_ids]

    return (
        x.reshape(-1, 26, 13).float(),
        y.reshape(-1, 26, 5).float(),
    )


@torch.no_grad()
def predict_all(model, x, edges):
    outputs = []

    for start in range(0, len(x), 64):
        batch = x[start:start + 64]

        outputs.append(
            forecast(model, batch, edges)
        )

    return torch.cat(outputs, dim=0)


def apply_thresholds(predictions, current, thresholds):
    changes = (
        predictions - current
    ).clamp(min=0)

    threshold_tensor = torch.tensor(
        thresholds,
        dtype=changes.dtype,
    ).view(1, 1, 5)

    accepted_changes = torch.where(
        changes >= threshold_tensor,
        changes,
        torch.zeros_like(changes),
    )

    return current + accepted_changes


def select_thresholds(
    predicted,
    current,
    actual,
):
    selected = []

    for resource in range(5):
        best_threshold = 0.0
        best_mse = float("inf")

        for threshold in THRESHOLDS:
            change = (
                predicted[..., resource]
                - current[..., resource]
            ).clamp(min=0)

            accepted = torch.where(
                change >= threshold,
                change,
                torch.zeros_like(change),
            )

            estimate = (
                current[..., resource]
                + accepted
            )

            mse = F.mse_loss(
                estimate,
                actual[..., resource],
            ).item()

            # Prefer a larger threshold on ties:
            # fewer unsupported demand increases.
            if mse <= best_mse:
                best_mse = mse
                best_threshold = threshold

        selected.append(best_threshold)

    return selected


def metrics(
    name,
    prediction,
    current,
    target,
    std,
):
    changed = (
        target - current
    ).abs() > 1e-6

    unchanged = ~changed

    absolute_error = (
        prediction - target
    ).abs() * std.view(1, 1, 5)

    predicted_change = (
        prediction - current
    ).abs() > 1e-6

    result = {
        "mse": F.mse_loss(
            prediction,
            target,
        ).item(),

        "mae_original_units": (
            absolute_error.mean().item()
        ),

        "changed_cell_mae": (
            absolute_error[changed]
            .mean()
            .item()
        ),

        "unchanged_cell_mae": (
            absolute_error[unchanged]
            .mean()
            .item()
        ),

        "false_change_rate_percent": (
            100.0
            * predicted_change[unchanged]
            .float()
            .mean()
            .item()
        ),
    }

    print()
    print(name)
    print("-" * 55)

    for key, value in result.items():
        print(f"{key}: {value:.6f}")

    return result


def main():
    torch.set_num_threads(2)

    data = torch.load(
        DATA_PATH,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = AluvaGAT(
        input_features=checkpoint["input_features"],
        hidden_features=checkpoint["hidden_features"],
        output_features=checkpoint["output_features"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    edges = data["edge_index"].long()
    std = data["demand_std"].float()

    val_x, val_y = flatten(
        data,
        data["validation_scenarios"],
    )

    test_x, test_y = flatten(
        data,
        data["test_scenarios"],
    )

    val_current = val_x[:, :, 8:13]
    test_current = test_x[:, :, 8:13]

    val_pred = predict_all(
        model,
        val_x,
        edges,
    )

    test_pred = predict_all(
        model,
        test_x,
        edges,
    )

    thresholds = select_thresholds(
        val_pred,
        val_current,
        val_y,
    )

    sparse_pred = apply_thresholds(
        test_pred,
        test_current,
        thresholds,
    )

    print("=" * 60)
    print("PHASE 23 — SPARSE GAT FORECAST TEST")
    print("=" * 60)

    print("Validation-selected thresholds:")

    for resource, threshold in zip(
        RESOURCE_NAMES,
        thresholds,
    ):
        print(f"  {resource}: {threshold}")

    changed_count = (
        (test_y - test_current).abs() > 1e-6
    ).sum().item()

    total_cells = test_y.numel()

    print()
    print("Test cells:", total_cells)
    print("Changed cells:", changed_count)
    print(
        "Changed percentage:",
        round(
            100 * changed_count / total_cells,
            2,
        ),
    )

    results = {}

    results["persistence"] = metrics(
        "PERSISTENCE",
        test_current,
        test_current,
        test_y,
        std,
    )

    results["gat_v3"] = metrics(
        "ORIGINAL GAT V3",
        test_pred,
        test_current,
        test_y,
        std,
    )

    results["sparse_gat"] = metrics(
        "SPARSE GAT V3",
        sparse_pred,
        test_current,
        test_y,
        std,
    )

    RESULT_PATH.write_text(
        json.dumps(
            {
                "thresholds": dict(
                    zip(
                        RESOURCE_NAMES,
                        thresholds,
                    )
                ),
                "test_cells": total_cells,
                "changed_cells": changed_count,
                "results": results,
                "synthetic_data": True,
                "threshold_selection": (
                    "Validation scenarios only"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved:", RESULT_PATH)
    print()
    print(
        "No trained checkpoints were changed."
    )


if __name__ == "__main__":
    main()
