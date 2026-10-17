
import torch
from pathlib import Path

from local_model import LocalReliefModel

model_dir = Path("data/processed/federated")

global_path = model_dir / "aluva_global_federated_model.pth"

global_weights = torch.load(
    global_path,
    map_location="cpu",
    weights_only=True
)

centres = ["RC01", "RC02", "RC03"]

print("FEDERATED MODEL SYNCHRONIZATION")
print("--------------------------------")

for centre_id in centres:
    model = LocalReliefModel()
    model.load_state_dict(global_weights)

    output_path = model_dir / f"{centre_id}_synchronized_model.pth"
    torch.save(model.state_dict(), output_path)

    synchronized_weights = torch.load(
        output_path,
        map_location="cpu",
        weights_only=True
    )

    assert all(
        torch.equal(global_weights[key], synchronized_weights[key])
        for key in global_weights
    )

    print(f"{centre_id}: Global model synchronized")

print("\nSUCCESS! All three relief centres synchronized.")
