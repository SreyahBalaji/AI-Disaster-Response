
from pathlib import Path
import torch

from cooperative_ppo_network import CooperativePPO


original_path = Path(
    "data/processed/marl/aluva_dynamic_cooperative_ppo.pth"
)

federated_path = Path(
    "data/processed/marl/federated/"
    "federated_trained_cooperative_ppo.pth"
)


def load_model(path):
    model = CooperativePPO()
    model.load_state_dict(
        torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    )
    return model


original = load_model(original_path)
federated = load_model(federated_path)

actor_changes = []
critic_changes = []
other_parameters = []

for name, original_param in original.named_parameters():
    federated_param = dict(
        federated.named_parameters()
    )[name]

    difference = torch.norm(
        original_param.detach()
        - federated_param.detach()
    ).item()

    if name.startswith("actor."):
        actor_changes.append(difference)

    elif name.startswith("critic."):
        critic_changes.append(difference)

    else:
        other_parameters.append((name, difference))


print("FEDERATED MODEL WEIGHT VERIFICATION")
print("-----------------------------------")

print("Actor parameters checked:", len(actor_changes))
print("Critic parameters checked:", len(critic_changes))
print("Other parameters:", other_parameters)

print(
    "Actor total difference:",
    sum(actor_changes)
)

print(
    "Critic total difference:",
    sum(critic_changes)
)

assert len(actor_changes) > 0, (
    "No actor parameters found."
)

assert len(critic_changes) > 0, (
    "No critic parameters found."
)

assert sum(actor_changes) > 0, (
    "Actor weights did not change."
)

assert all(
    change < 1e-5
    for change in critic_changes
), "Critic weights changed beyond numerical tolerance."

assert not other_parameters, (
    "Unexpected model parameters found."
)

print("\nSUCCESS! Federated weight verification passed.")
