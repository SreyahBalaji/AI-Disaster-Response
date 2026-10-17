
from pathlib import Path
import torch

from cooperative_ppo_network import CooperativePPO


MODEL_DIR = Path("data/processed/marl/federated")
CENTRES = ["RC01", "RC02", "RC03"]


def load_state(path):
    if not path.exists():
        raise FileNotFoundError(path)

    model = CooperativePPO()

    model.load_state_dict(
        torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    )

    return model.state_dict()


local_states = [
    load_state(
        MODEL_DIR / f"{centre}_ppo_trained.pth"
    )
    for centre in CENTRES
]

global_state = load_state(
    MODEL_DIR / "fedavg_cooperative_ppo.pth"
)

print("FEDERATED PPO VALIDATION")
print("------------------------")

max_difference = 0.0

for key in global_state:
    local_tensors = [
        state[key]
        for state in local_states
    ]

    expected = torch.stack(
        [tensor.float() for tensor in local_tensors]
    ).mean(dim=0)

    actual = global_state[key].float()

    difference = torch.max(
        torch.abs(expected - actual)
    ).item()

    max_difference = max(
        max_difference,
        difference
    )

    assert torch.allclose(
        expected,
        actual,
        atol=1e-6,
        rtol=1e-5
    ), f"FedAvg mismatch: {key}"

print("Local models verified:", len(local_states))
print("Maximum parameter difference:", max_difference)

print(
    "\nSUCCESS! Federated PPO aggregation "
    "verified mathematically."
)
