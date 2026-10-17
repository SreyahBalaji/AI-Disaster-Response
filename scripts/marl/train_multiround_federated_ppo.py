
from pathlib import Path
import json
import random

import numpy as np
import torch

from cooperative_ppo_network import CooperativePPO
from federated_ppo_trainer import train_local_ppo


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CENTRES = ["RC01", "RC02", "RC03"]

ROUNDS = 5
LOCAL_EPISODES = 10

MODEL_DIR = Path("data/processed/marl/federated")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_MODEL = MODEL_DIR / "fedavg_cooperative_ppo.pth"

OUTPUT_MODEL = (
    MODEL_DIR / "multiround_federated_ppo.pth"
)

OUTPUT_HISTORY = (
    MODEL_DIR / "multiround_training_history.json"
)


def federated_average(local_states):
    averaged_state = {}

    for key in local_states[0]:
        tensors = [
            state[key]
            for state in local_states
        ]

        if torch.is_floating_point(tensors[0]):
            averaged_state[key] = torch.stack(
                [
                    tensor.float()
                    for tensor in tensors
                ]
            ).mean(dim=0).to(tensors[0].dtype)
        else:
            averaged_state[key] = tensors[0].clone()

    return averaged_state


if not INITIAL_MODEL.exists():
    raise FileNotFoundError(
        f"Initial model not found: {INITIAL_MODEL}"
    )

global_model = CooperativePPO()

global_model.load_state_dict(
    torch.load(
        INITIAL_MODEL,
        map_location="cpu",
        weights_only=True
    )
)

history = []

print("MULTI-ROUND FEDERATED PPO TRAINING")
print("---------------------------------")
print("Federated rounds:", ROUNDS)
print("Relief centres:", len(CENTRES))
print("Local episodes per centre:", LOCAL_EPISODES)

for round_number in range(1, ROUNDS + 1):
    print(f"\nROUND {round_number}/{ROUNDS}")

    local_states = []
    centre_rewards = {}

    for centre_id in CENTRES:
        # Every centre starts this round with
        # the same current global model.
        local_state, mean_reward = train_local_ppo(
            global_state=global_model.state_dict(),
            centre_id=centre_id,
            episodes=LOCAL_EPISODES
        )

        local_states.append(local_state)
        centre_rewards[centre_id] = mean_reward

        print(
            f"{centre_id} | "
            f"Mean team reward: {mean_reward:.2f}"
        )

    # Aggregate locally trained parameters.
    averaged_state = federated_average(local_states)

    # Distribute the new global weights for
    # the next federated round.
    global_model.load_state_dict(averaged_state)

    round_mean = float(
        np.mean(list(centre_rewards.values()))
    )

    history.append({
        "round": round_number,
        "centre_rewards": centre_rewards,
        "mean_team_reward": round_mean
    })

    print(
        f"FedAvg completed | "
        f"Round mean reward: {round_mean:.2f}"
    )

torch.save(
    global_model.state_dict(),
    OUTPUT_MODEL
)

with open(
    OUTPUT_HISTORY,
    "w",
    encoding="utf-8"
) as file:
    json.dump(history, file, indent=4)

# Verify the saved checkpoint can be loaded.
verification_model = CooperativePPO()

verification_model.load_state_dict(
    torch.load(
        OUTPUT_MODEL,
        map_location="cpu",
        weights_only=True
    )
)

print("\nFINAL RESULTS")
print("-------------")
print("Total federated rounds:", ROUNDS)
print(
    "Total local training episodes:",
    ROUNDS * len(CENTRES) * LOCAL_EPISODES
)
print("Global model saved:", OUTPUT_MODEL)
print("Training history saved:", OUTPUT_HISTORY)

print(
    "\nSUCCESS! Multi-round federated PPO "
    "training completed."
)
