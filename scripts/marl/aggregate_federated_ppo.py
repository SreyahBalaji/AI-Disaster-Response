
from pathlib import Path

import torch

from cooperative_ppo_network import CooperativePPO


MODEL_DIR = Path("data/processed/marl/federated")

CENTRES = ["RC01", "RC02", "RC03"]

OUTPUT_PATH = (
    MODEL_DIR / "fedavg_cooperative_ppo.pth"
)


def load_local_model(centre_id):
    path = MODEL_DIR / f"{centre_id}_ppo_trained.pth"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing local model: {path}"
        )

    model = CooperativePPO()

    model.load_state_dict(
        torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    )

    return model


print("FEDERATED PPO AGGREGATION")
print("-------------------------")

local_models = []

for centre_id in CENTRES:
    model = load_local_model(centre_id)
    local_models.append(model)

    print(f"{centre_id}: trained model loaded")


local_states = [
    model.state_dict()
    for model in local_models
]

averaged_state = {}

for key in local_states[0]:
    tensors = [
        state[key]
        for state in local_states
    ]

    if torch.is_floating_point(tensors[0]):
        averaged_state[key] = torch.stack(
            [tensor.float() for tensor in tensors]
        ).mean(dim=0).to(tensors[0].dtype)
    else:
        averaged_state[key] = tensors[0].clone()


global_model = CooperativePPO()
global_model.load_state_dict(averaged_state)

torch.save(
    global_model.state_dict(),
    OUTPUT_PATH
)

print("\nGlobal model saved:", OUTPUT_PATH)


# Verify that the saved model can be loaded.
verified_model = CooperativePPO()

verified_model.load_state_dict(
    torch.load(
        OUTPUT_PATH,
        map_location="cpu",
        weights_only=True
    )
)

# Verify the aggregation mathematically.
for key in averaged_state:
    assert torch.allclose(
        verified_model.state_dict()[key],
        averaged_state[key]
    ), f"Aggregation mismatch: {key}"


print("SUCCESS! Federated PPO aggregation validated.")
