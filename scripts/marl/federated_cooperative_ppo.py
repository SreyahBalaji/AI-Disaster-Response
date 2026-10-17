
from pathlib import Path
import copy

import torch

from cooperative_ppo_network import CooperativePPO


torch.manual_seed(42)

CENTRE_IDS = ["RC01", "RC02", "RC03"]

SOURCE_MODEL = Path(
    "data/processed/marl/aluva_dynamic_cooperative_ppo.pth"
)

OUTPUT_DIR = Path("data/processed/marl/federated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    model = CooperativePPO()

    model.load_state_dict(
        torch.load(
            SOURCE_MODEL,
            map_location="cpu",
            weights_only=True
        )
    )

    return model


def federated_average(local_models):
    global_state = {}

    state_dicts = [
        model.state_dict()
        for model in local_models
    ]

    for key in state_dicts[0]:
        global_state[key] = torch.stack(
            [
                state[key].float()
                for state in state_dicts
            ]
        ).mean(dim=0)

    return global_state


print("FEDERATED COOPERATIVE PPO INITIALIZATION")
print("--------------------------------")

local_models = []

for centre_id in CENTRE_IDS:
    model = load_model()
    local_models.append(model)

    local_path = OUTPUT_DIR / f"{centre_id}_local_ppo.pth"

    torch.save(
        model.state_dict(),
        local_path
    )

    print(f"{centre_id}: local model initialized")


global_state = federated_average(local_models)

global_model = CooperativePPO()
global_model.load_state_dict(global_state)

global_path = OUTPUT_DIR / "global_cooperative_ppo.pth"

torch.save(
    global_model.state_dict(),
    global_path
)

print("\nGlobal model saved:", global_path)

# Confirm that initialization preserves the source weights.
source_state = local_models[0].state_dict()

for key in source_state:
    assert torch.allclose(
        source_state[key],
        global_model.state_dict()[key]
    )

print("SUCCESS! Federated PPO initialization validated.")
