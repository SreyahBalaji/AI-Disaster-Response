
"""
RESPONDA — Phase 23
Validate the existing trained Dynamic GAT.

No retraining.
No changes to existing checkpoints.
No modifications to the working website.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from build_gat_model import AluvaGAT


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "aluva"

DATA_PATH = DATA_DIR / "aluva_gat_prediction_data.pt"
MODEL_PATH = DATA_DIR / "aluva_dynamic_gat_model.pth"


def main():
    print("=" * 58)
    print("PHASE 23 — DYNAMIC GAT INFERENCE TEST")
    print("=" * 58)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    data = torch.load(
        DATA_PATH,
        map_location="cpu",
        weights_only=True,
    )

    inputs = data["inputs"].float()
    targets = data["targets"].float()
    edges = data["edge_index"].long()

    if inputs.ndim != 3 or inputs.shape[1:] != (26, 13):
        raise ValueError(
            f"Expected [snapshots, 26, 13], got {inputs.shape}"
        )

    if targets.shape != (
        inputs.shape[0], 26, 5
    ):
        raise ValueError(
            f"Unexpected target shape: {targets.shape}"
        )

    model = AluvaGAT(
        input_features=13,
        hidden_features=16,
        output_features=5,
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    with torch.no_grad():
        predictions = torch.stack([
            model(snapshot, edges)
            for snapshot in inputs
        ])

    assert predictions.shape == targets.shape
    assert torch.isfinite(predictions).all()

    print("Saved model: LOADED")
    print("Input shape:", tuple(inputs.shape))
    print("Output shape:", tuple(predictions.shape))
    print("Graph edges:", edges.shape[1])

    first_input = inputs[0]
    last_input = inputs[-1]

    first_prediction = predictions[0]
    last_prediction = predictions[-1]

    input_change = (
        last_input - first_input
    ).abs().mean().item()

    prediction_change = (
        last_prediction - first_prediction
    ).abs().mean().item()

    print()
    print("SNAPSHOT CHANGE CHECK")
    print("-" * 58)
    print(f"Mean input change:      {input_change:.6f}")
    print(f"Mean prediction change: {prediction_change:.6f}")

    if prediction_change > 1e-8:
        print("PASS: Dynamic GAT output changes with input.")
    else:
        print(
            "WARNING: Outputs did not change between "
            "the selected snapshots."
        )

    test_indices = data["test_indices"]

    with torch.no_grad():
        test_predictions = predictions[test_indices]
        test_targets = targets[test_indices]

        persistence_predictions = inputs[
            test_indices, :, 8:13
        ]

        gat_mse = F.mse_loss(
            test_predictions,
            test_targets,
        ).item()

        persistence_mse = F.mse_loss(
            persistence_predictions,
            test_targets,
        ).item()

    print()
    print("EXISTING TEST-SPLIT EVALUATION")
    print("-" * 58)
    print(f"Dynamic GAT test MSE: {gat_mse:.6f}")
    print(f"Persistence test MSE: {persistence_mse:.6f}")

    print()
    print("RESULT")
    print("-" * 58)
    print("Model loaded and inference completed.")

    print(
        "IMPORTANT: Outputs use the model's training "
        "data scale; they are not automatically litres, "
        "meals or rescue counts."
    )

    print(
        "Next: Reuse the original input normalization "
        "when processing newly uploaded reports."
    )


if __name__ == "__main__":
    main()
