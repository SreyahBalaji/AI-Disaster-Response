
import torch
from pathlib import Path

from local_model import LocalReliefModel
from federated_server import federated_average

model_dir = Path("data/processed/federated")

centres = ["RC01", "RC02", "RC03"]
local_weights = []

print("FEDERATED MODEL AGGREGATION")
print("----------------------------")

for centre_id in centres:
    model_path = model_dir / f"{centre_id}_local_model.pth"

    weights = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    local_weights.append(weights)
    print(f"Loaded model from {centre_id}")

# Average the three sets of model parameters.
global_weights = federated_average(local_weights)

global_model = LocalReliefModel()
global_model.load_state_dict(global_weights)

output_path = model_dir / "aluva_global_federated_model.pth"

torch.save(
    global_model.state_dict(),
    output_path
)

print("\nParticipating relief centres:", len(centres))
print("Global model saved:", output_path)

# Verify the global model produces a valid prediction.
sample_input = torch.tensor(
    [[0.8, 0.6, 0.4, 0.7, 0.5]],
    dtype=torch.float32
)

global_model.eval()

with torch.no_grad():
    prediction = global_model(sample_input)

assert prediction.shape == (1, 1)
assert torch.isfinite(prediction).all()

print("Sample prediction:", prediction.item())
print("\nSUCCESS! Global federated model created.")
