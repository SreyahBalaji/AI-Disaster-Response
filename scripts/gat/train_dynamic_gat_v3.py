
"""
RESPONDA — Phase 23: Dynamic GAT v3

Train on 90 synthetic disaster scenarios using
separate train, validation, and test scenarios.

GAT predicts non-negative changes in demand.
Compare against persistence on held-out scenarios.

Existing models are not overwritten.
"""

import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from build_gat_model import AluvaGAT


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "processed" / "aluva"

DATA_PATH = BASE / "aluva_forecast_scenarios_v2.pt"
MODEL_PATH = BASE / "aluva_dynamic_gat_v3.pth"
RESULTS_PATH = BASE / "aluva_dynamic_gat_v3_results.json"

SEED = 42
EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.003


def load_dataset():
    data = torch.load(
        DATA_PATH,
        map_location="cpu",
        weights_only=True,
    )

    x = data["inputs"].float()
    y = data["targets"].float()
    edges = data["edge_index"].long()

    assert x.shape == (90, 16, 26, 13)
    assert y.shape == (90, 16, 26, 5)
    assert edges.shape[0] == 2

    splits = [
        data["train_scenarios"],
        data["validation_scenarios"],
        data["test_scenarios"],
    ]

    sets = [set(ids) for ids in splits]

    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])
    assert len(set.union(*sets)) == 90

    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()

    return data, x, y, edges


def flatten_scenarios(x, y, scenario_ids):
    selected_x = x[scenario_ids]
    selected_y = y[scenario_ids]

    return (
        selected_x.reshape(-1, 26, 13),
        selected_y.reshape(-1, 26, 5),
    )


def batched_edges(edge_index, batch_size):
    """
    Build disconnected copies of the same
    ward graph for efficient batch training.
    """
    edge_count = edge_index.shape[1]

    offsets = (
        torch.arange(batch_size)
        .repeat_interleave(edge_count)
        * 26
    )

    return (
        edge_index.repeat(1, batch_size)
        + offsets.unsqueeze(0)
    )


def forecast(model, features, edge_index):
    batch_size = features.shape[0]

    edges = batched_edges(
        edge_index,
        batch_size,
    )

    graph_input = features.reshape(
        batch_size * 26,
        13,
    )

    raw_change = model(
        graph_input,
        edges,
    ).reshape(batch_size, 26, 5)

    # Demand is cumulative in this simulation.
    # Predicted change must not be negative.
    positive_change = (
        0.1 * F.softplus(raw_change)
    )

    current_demand = features[:, :, 8:13]

    return current_demand + positive_change


@torch.no_grad()
def get_predictions(
    model,
    x,
    y,
    edges,
):
    model.eval()

    predictions = []

    for start in range(
        0,
        len(x),
        BATCH_SIZE,
    ):
        batch = x[start:start + BATCH_SIZE]

        predictions.append(
            forecast(model, batch, edges)
        )

    return torch.cat(predictions), y


def normalized_mse(
    model,
    x,
    y,
    edges,
):
    predictions, actual = get_predictions(
        model, x, y, edges
    )

    return F.mse_loss(
        predictions, actual
    ).item()


def calculate_test_metrics(
    model,
    x,
    y,
    edges,
    mean,
    std,
):
    predictions, targets = get_predictions(
        model, x, y, edges
    )

    current = x[:, :, 8:13]

    gat_mse = F.mse_loss(
        predictions,
        targets,
    ).item()

    baseline_mse = F.mse_loss(
        current,
        targets,
    ).item()

    gat_original = (
        predictions * std + mean
    )

    target_original = (
        targets * std + mean
    )

    baseline_original = (
        current * std + mean
    )

    gat_mae = (
        gat_original - target_original
    ).abs().mean().item()

    baseline_mae = (
        baseline_original - target_original
    ).abs().mean().item()

    # Evaluate changed-demand cells separately.
    # Most cells may have unchanged demand,
    # making overall errors look artificially low.
    changed_mask = (
        targets - current
    ).abs() > 1e-6

    changed_count = int(
        changed_mask.sum().item()
    )

    if changed_count:
        gat_changed_mae = (
            gat_original - target_original
        ).abs()[changed_mask].mean().item()

        baseline_changed_mae = (
            baseline_original - target_original
        ).abs()[changed_mask].mean().item()
    else:
        gat_changed_mae = None
        baseline_changed_mae = None

    resource_names = [
        "Water",
        "Food",
        "Rescue",
        "Shelter",
        "Medical",
    ]

    resource_results = {}

    for i, resource in enumerate(
        resource_names
    ):
        resource_results[resource] = {
            "gat_mae": round(
                (
                    gat_original[..., i]
                    - target_original[..., i]
                ).abs().mean().item(),
                4,
            ),
            "persistence_mae": round(
                (
                    baseline_original[..., i]
                    - target_original[..., i]
                ).abs().mean().item(),
                4,
            ),
        }

    return {
        "gat_normalized_mse": gat_mse,
        "persistence_normalized_mse": baseline_mse,
        "gat_original_units_mae": gat_mae,
        "persistence_original_units_mae": baseline_mae,
        "changed_demand_cells": changed_count,
        "gat_changed_cells_mae": gat_changed_mae,
        "persistence_changed_cells_mae": (
            baseline_changed_mae
        ),
        "resources": resource_results,
    }


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    data, inputs, targets, edges = load_dataset()

    train_x, train_y = flatten_scenarios(
        inputs,
        targets,
        data["train_scenarios"],
    )

    val_x, val_y = flatten_scenarios(
        inputs,
        targets,
        data["validation_scenarios"],
    )

    test_x, test_y = flatten_scenarios(
        inputs,
        targets,
        data["test_scenarios"],
    )

    model = AluvaGAT(
        input_features=13,
        hidden_features=16,
        output_features=5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0005,
    )

    print("=" * 64)
    print("PHASE 23 — DYNAMIC GAT V3 TRAINING")
    print("=" * 64)

    print("Training scenarios: 62")
    print("Validation scenarios: 13")
    print("Testing scenarios: 15")

    print("Training transitions:", len(train_x))
    print("Validation transitions:", len(val_x))
    print("Test transitions:", len(test_x))

    print("Input features: 13")
    print("Predicted demand resources: 5")
    print()

    best_validation = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        order = torch.randperm(
            len(train_x)
        )

        epoch_loss = 0.0
        batch_count = 0

        for start in range(
            0,
            len(train_x),
            BATCH_SIZE,
        ):
            indices = order[
                start:start + BATCH_SIZE
            ]

            batch_x = train_x[indices]
            batch_y = train_y[indices]

            prediction = forecast(
                model,
                batch_x,
                edges,
            )

            # Emphasize newly emerging demand,
            # while retaining overall MSE.
            changed = (
                batch_y - batch_x[:, :, 8:13]
            ).abs() > 1e-6

            weights = (
                1.0 + 2.0 * changed.float()
            )

            error = (
                prediction - batch_y
            ).square()

            weighted_loss = (
                error * weights
            ).mean()

            correction = (
                prediction
                - batch_x[:, :, 8:13]
            )

            loss = (
                weighted_loss
                + 0.01 * correction.square().mean()
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        val_mse = normalized_mse(
            model,
            val_x,
            val_y,
            edges,
        )

        if val_mse < best_validation - 1e-7:
            best_validation = val_mse
            best_epoch = epoch
            best_state = deepcopy(
                model.state_dict()
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3} "
                f"| Train loss="
                f"{epoch_loss / batch_count:.6f} "
                f"| Validation MSE="
                f"{val_mse:.6f}",
                flush=True,
            )

        if stale_epochs >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    model.load_state_dict(best_state)
    model.eval()

    print()
    print("=" * 64)
    print("HELD-OUT TEST RESULTS")
    print("=" * 64)

    results = calculate_test_metrics(
        model,
        test_x,
        test_y,
        edges,
        data["demand_mean"].float(),
        data["demand_std"].float(),
    )

    print("Best epoch:", best_epoch)

    print(
        "Dynamic GAT v3 MSE:",
        f"{results['gat_normalized_mse']:.6f}",
    )

    print(
        "Persistence MSE:   ",
        f"{results['persistence_normalized_mse']:.6f}",
    )

    print()
    print("OVERALL MAE IN ORIGINAL RESOURCE UNITS")

    print(
        "Dynamic GAT v3:",
        f"{results['gat_original_units_mae']:.4f}",
    )

    print(
        "Persistence:   ",
        f"{results['persistence_original_units_mae']:.4f}",
    )

    print()
    print("NEW/CHANGED DEMAND CELLS")

    print(
        "Changed cells:",
        results["changed_demand_cells"],
    )

    print(
        "GAT changed-cell MAE:",
        results["gat_changed_cells_mae"],
    )

    print(
        "Persistence changed-cell MAE:",
        results["persistence_changed_cells_mae"],
    )

    print()
    print("PER-RESOURCE TEST MAE")
    print("-" * 64)

    for resource, metrics in results[
        "resources"
    ].items():
        print(
            f"{resource:9} "
            f"GAT={metrics['gat_mae']:10.4f} "
            f"Persistence="
            f"{metrics['persistence_mae']:10.4f}"
        )

    print()
    print("=" * 64)
    print("FINAL STATUS")
    print("=" * 64)

    if (
        results["gat_normalized_mse"]
        < results["persistence_normalized_mse"]
    ):
        print(
            "PASS: GAT v3 beats persistence "
            "on held-out synthetic scenarios."
        )
    else:
        print(
            "Persistence remains better "
            "on held-out synthetic scenarios."
        )

    # Save a separate model with all parameters
    # needed for future consistent inference.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": 13,
            "hidden_features": 16,
            "output_features": 5,
            "correction_scale": 0.1,
            "demand_mean": data["demand_mean"],
            "demand_std": data["demand_std"],
            "best_epoch": best_epoch,
            "seed": SEED,
            "synthetic": True,
        },
        MODEL_PATH,
    )

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "training_scenarios": 62,
                "validation_scenarios": 13,
                "test_scenarios": 15,
                "best_epoch": best_epoch,
                "synthetic_data": True,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved model:", MODEL_PATH)
    print("Saved results:", RESULTS_PATH)
    print()
    print(
        "Experimental forecasts only; "
        "not approved for real deployments."
    )


if __name__ == "__main__":
    main()
