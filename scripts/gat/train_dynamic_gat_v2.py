
"""
RESPONDA — Dynamic GAT v2

Goal:
Predict how resource demand changes between
consecutive disaster snapshots.

Training:
- 5 chronological transitions
- 1 validation transition
- 3 held-out test transitions

The model predicts a correction to current demand.
This provides a stronger persistence-based starting point.

Existing models and website remain unchanged.
"""

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from build_gat_model import AluvaGAT


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "aluva"

DATA_PATH = (
    DATA_DIR / "aluva_dynamic_gat_normalized.pt"
)

MODEL_PATH = (
    DATA_DIR / "aluva_dynamic_gat_v2.pth"
)

RESOURCES = [
    "Water",
    "Food",
    "Rescue",
    "Shelter",
    "Medical",
]

SEED = 42
MAX_EPOCHS = 160
PATIENCE = 25


def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Normalized dataset missing: {DATA_PATH}"
        )

    data = torch.load(
        DATA_PATH,
        map_location="cpu",
        weights_only=True,
    )

    x = data["x"].float()
    edges = data["edge_index"].long()

    mean = data["demand_mean"].float()
    std = data["demand_std"].float()

    if x.shape != (10, 26, 13):
        raise ValueError(
            f"Expected (10, 26, 13), got {tuple(x.shape)}"
        )

    if mean.shape != (5,) or std.shape != (5,):
        raise ValueError(
            "Invalid demand normalization statistics"
        )

    if not torch.isfinite(x).all():
        raise ValueError(
            "Dataset contains non-finite values"
        )

    if (std <= 0).any():
        raise ValueError(
            "Normalization std must be positive"
        )

    # Snapshot t predicts snapshot t+1.
    inputs = x[:-1]
    targets = x[1:, :, 8:13]

    # Chronological split: no random shuffling.
    train_indices = list(range(5))
    validation_indices = [5]
    test_indices = [6, 7, 8]

    return (
        inputs,
        targets,
        edges,
        mean,
        std,
        train_indices,
        validation_indices,
        test_indices,
    )


def create_model():
    return AluvaGAT(
        input_features=13,
        hidden_features=8,
        output_features=5,
    )


def predict(model, current, edges):
    """
    Current demand is already normalized.

    Model outputs the adjustment to apply to
    the current demand, not absolute demand.
    """
    current_demand = current[:, 8:13]

    correction = model(current, edges)

    return current_demand + 0.15 * correction


def evaluate(model, inputs, targets, edges, indices):
    model.eval()

    with torch.no_grad():
        predictions = torch.stack([
            predict(model, inputs[i], edges)
            for i in indices
        ])

        actual = targets[indices]

        persistence = inputs[
            indices, :, 8:13
        ]

        model_mse = F.mse_loss(
            predictions,
            actual,
        ).item()

        persistence_mse = F.mse_loss(
            persistence,
            actual,
        ).item()

    return model_mse, persistence_mse


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    (
        inputs,
        targets,
        edges,
        mean,
        std,
        train_indices,
        validation_indices,
        test_indices,
    ) = load_data()

    model = create_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.003,
        weight_decay=0.001,
    )

    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    no_improvement = 0

    print("=" * 60)
    print("PHASE 23 — DYNAMIC GAT V2 TRAINING")
    print("=" * 60)

    print("Snapshots: 10")
    print("Total transitions: 9")
    print("Training transitions: 5")
    print("Validation transitions: 1")
    print("Test transitions: 3")
    print("Input features: 13")
    print("Predicted resources: 5")
    print()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        predictions = torch.stack([
            predict(
                model,
                inputs[i],
                edges,
            )
            for i in train_indices
        ])

        actual = targets[train_indices]

        prediction_loss = F.mse_loss(
            predictions,
            actual,
        )

        # Encourage small adjustments when the
        # evidence for a large change is weak.
        current_demand = inputs[
            train_indices, :, 8:13
        ]

        correction_penalty = (
            predictions - current_demand
        ).square().mean()

        loss = (
            prediction_loss
            + 0.01 * correction_penalty
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        validation_mse, _ = evaluate(
            model,
            inputs,
            targets,
            edges,
            validation_indices,
        )

        if validation_mse < best_validation - 1e-7:
            best_validation = validation_mse
            best_state = deepcopy(
                model.state_dict()
            )
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:3} "
                f"| Train loss={loss.item():.6f} "
                f"| Validation MSE={validation_mse:.6f}",
                flush=True,
            )

        if no_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    model.load_state_dict(best_state)
    model.eval()

    validation_gat, validation_persistence = evaluate(
        model,
        inputs,
        targets,
        edges,
        validation_indices,
    )

    test_gat, test_persistence = evaluate(
        model,
        inputs,
        targets,
        edges,
        test_indices,
    )

    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print("Best epoch:", best_epoch)
    print(f"Dynamic GAT v2 MSE: {validation_gat:.6f}")
    print(
        f"Persistence MSE:    "
        f"{validation_persistence:.6f}"
    )

    print()
    print("=" * 60)
    print("HELD-OUT TEST RESULTS")
    print("=" * 60)

    print(f"Dynamic GAT v2 MSE: {test_gat:.6f}")
    print(f"Persistence MSE:    {test_persistence:.6f}")

    # Report per-resource errors in original
    # resource units, not just normalized units.
    with torch.no_grad():
        predicted = torch.stack([
            predict(model, inputs[i], edges)
            for i in test_indices
        ])

        actual = targets[test_indices]

        predicted_original = (
            predicted * std + mean
        ).clamp(min=0)

        actual_original = (
            actual * std + mean
        ).clamp(min=0)

        persistence_original = (
            inputs[
                test_indices, :, 8:13
            ] * std + mean
        ).clamp(min=0)

        gat_mae = (
            predicted_original - actual_original
        ).abs().mean(dim=(0, 1))

        baseline_mae = (
            persistence_original - actual_original
        ).abs().mean(dim=(0, 1))

    print()
    print("PER-RESOURCE TEST MAE")
    print("-" * 60)

    for index, resource in enumerate(RESOURCES):
        print(
            f"{resource:9} "
            f"GAT={gat_mae[index].item():9.3f} "
            f"Persistence={baseline_mae[index].item():9.3f}"
        )

    print()
    print("=" * 60)
    print("FINAL EXPERIMENT STATUS")
    print("=" * 60)

    if test_gat < test_persistence:
        print(
            "GAT v2 has lower test MSE than "
            "persistence on this split."
        )
    else:
        print(
            "Persistence still has lower test MSE."
        )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": 13,
            "hidden_features": 8,
            "output_features": 5,
            "demand_mean": mean,
            "demand_std": std,
            "best_epoch": best_epoch,
            "seed": SEED,
            "test_gat_mse": test_gat,
            "test_persistence_mse": test_persistence,
        },
        MODEL_PATH,
    )

    print()
    print("Saved experimental model:", MODEL_PATH)

    print(
        "Model is not approved for live allocations."
    )

    print(
        "This test uses only three held-out "
        "transitions and requires broader validation."
    )


if __name__ == "__main__":
    main()
