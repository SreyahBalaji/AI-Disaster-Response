
from pathlib import Path
import torch

from local_model import LocalReliefModel

torch.manual_seed(123)

model_path = Path(
    "data/processed/federated/aluva_multiround_global_model.pth"
)

trained_model = LocalReliefModel()
trained_model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )
)
trained_model.eval()

untrained_model = LocalReliefModel()
untrained_model.eval()

# Unseen synthetic test samples.
x_test = torch.rand(100, 5)

y_test = (
    0.5 * x_test[:, 0:1]
    + 0.3 * x_test[:, 1:2]
    + 0.2 * x_test[:, 2:3]
)

loss_function = torch.nn.MSELoss()

with torch.no_grad():
    trained_mse = loss_function(
        trained_model(x_test), y_test
    ).item()

    untrained_mse = loss_function(
        untrained_model(x_test), y_test
    ).item()

print("FEDERATED MODEL EVALUATION")
print("----------------------------")
print(f"Untrained model MSE: {untrained_mse:.6f}")
print(f"Trained model MSE:   {trained_mse:.6f}")

assert torch.isfinite(torch.tensor(trained_mse))

if trained_mse < untrained_mse:
    print("\nTrained model outperformed the untrained baseline.")
else:
    print("\nTrained model did not outperform the baseline.")

print("\nSUCCESS! Federated model evaluation completed.")
