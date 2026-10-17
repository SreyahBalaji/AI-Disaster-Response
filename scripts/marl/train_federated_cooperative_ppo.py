
from pathlib import Path
import copy

import torch
import torch.nn.functional as F

from cooperative_ppo_network import CooperativePPO


torch.manual_seed(42)

CENTRES = ["RC01", "RC02", "RC03"]

BASE_PATH = Path(
    "data/processed/marl/federated/global_cooperative_ppo.pth"
)

OUTPUT_DIR = Path("data/processed/marl/federated")

ROUNDS = 5
LOCAL_EPOCHS = 10
LEARNING_RATE = 0.0001


def load_model():
    model = CooperativePPO()

    model.load_state_dict(
        torch.load(
            BASE_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    return model


def make_local_data(centre_index):
    # Observation structure:
    # unmet demand for 3 wards,
    # fulfilment ratios for 3 wards,
    # local centre stock.

    observations = []
    targets = []

    for ward_index in range(3):
        for level in [0.2, 0.4, 0.6, 0.8]:
            observation = torch.tensor(
                [
                    0.48 * level,
                    0.30 * level,
                    0.216 * level,
                    1.0 - level,
                    1.0 - level,
                    1.0 - level,
                    [0.5, 0.4, 0.4][centre_index]
                ],
                dtype=torch.float32
            )

            # Centre-specific synthetic preference.
            # This is supervised imitation, not PPO.
            observations.append(observation)
            targets.append(ward_index)

    return (
        torch.stack(observations),
        torch.tensor(targets, dtype=torch.long)
    )


def train_local(model, centre_index):
    model.train()

    observations, targets = make_local_data(
        centre_index
    )

    # Train actor parameters only.
    optimizer = torch.optim.Adam(
        model.actor.parameters(),
        lr=LEARNING_RATE
    )

    for _ in range(LOCAL_EPOCHS):
        optimizer.zero_grad()

        logits = model.actor(observations)

        loss = F.cross_entropy(
            logits,
            targets
        )

        loss.backward()
        optimizer.step()

    return model, loss.item()


def federated_average(models):
    states = [
        model.state_dict()
        for model in models
    ]

    averaged = {}

    for key in states[0]:
        averaged[key] = torch.stack(
            [
                state[key].float()
                for state in states
            ]
        ).mean(dim=0)

    return averaged


global_model = load_model()

print("FEDERATED COOPERATIVE PPO TRAINING")
print("--------------------------------")

for round_number in range(1, ROUNDS + 1):
    local_models = []
    local_losses = []

    for centre_index, centre_id in enumerate(CENTRES):
        local_model = copy.deepcopy(global_model)

        local_model, loss = train_local(
            local_model,
            centre_index
        )

        local_models.append(local_model)
        local_losses.append(loss)

    global_state = federated_average(
        local_models
    )

    global_model.load_state_dict(
        global_state
    )

    print(
        f"Round {round_number}/{ROUNDS} | "
        f"Mean local loss: "
        f"{sum(local_losses) / len(local_losses):.4f}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = (
    OUTPUT_DIR / "federated_trained_cooperative_ppo.pth"
)

torch.save(
    global_model.state_dict(),
    output_path
)

print("\nModel saved:", output_path)
print("SUCCESS! Federated actor training completed.")

