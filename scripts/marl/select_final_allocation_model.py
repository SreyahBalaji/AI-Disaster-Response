
from pathlib import Path
import shutil

import torch

from cooperative_ppo_network import CooperativePPO


SOURCE = Path(
    "data/processed/marl/federated/"
    "multiround_federated_ppo.pth"
)

DESTINATION = Path(
    "data/processed/marl/"
    "selected_cooperative_ppo.pth"
)

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

DESTINATION.parent.mkdir(
    parents=True,
    exist_ok=True
)

shutil.copy2(SOURCE, DESTINATION)

model = CooperativePPO()

model.load_state_dict(
    torch.load(
        DESTINATION,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

print("FINAL PPO MODEL SELECTION")
print("-------------------------")
print("Selected:", SOURCE)
print("Saved as:", DESTINATION)
print("Model loaded successfully.")

print(
    "\nSUCCESS! Selected PPO model ready "
    "for backend integration."
)
