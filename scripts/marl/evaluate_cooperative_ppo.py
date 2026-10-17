
import numpy as np
import torch

from cooperative_allocation_env import CooperativeAllocationEnv
from cooperative_ppo_network import CooperativePPO


model = CooperativePPO()

model.load_state_dict(
    torch.load(
        "data/processed/marl/aluva_cooperative_ppo.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

results = []


def to_tensor(observations, centre_ids):
    return torch.tensor(
        np.stack([
            observations[centre_id]
            for centre_id in centre_ids
        ]),
        dtype=torch.float32
    ).unsqueeze(0)


for episode in range(100):
    torch.manual_seed(1000 + episode)

    env = CooperativeAllocationEnv()
    observations = env.reset()

    total_reward = 0.0

    while True:
        state = to_tensor(observations, env.centre_ids)

        with torch.no_grad():
            actions, _, _, _ = (
                model.get_actions_and_value(state)
            )

        action_dict = {
            centre_id: int(actions[0, i].item())
            for i, centre_id in enumerate(env.centre_ids)
        }

        observations, rewards, terminated, truncated, _ = (
            env.step(action_dict)
        )

        total_reward += rewards[env.centre_ids[0]]

        if terminated or truncated:
            break

    deliveries = {
        ward: env.env.delivered[ward]["water_litres"]
        for ward in env.wards
    }

    fulfilment = []

    for ward in env.wards:
        delivered = deliveries[ward]
        remaining = env.env.demand[ward]["water_litres"]

        original = delivered + remaining

        fulfilment.append(
            delivered / original if original > 0 else 1.0
        )

    initial_stock = 1300
    total_delivered = sum(deliveries.values())

    results.append({
        "reward": total_reward,
        "delivered": total_delivered,
        "utilization": total_delivered / initial_stock,
        "fairness_gap": max(fulfilment) - min(fulfilment),
        "ward3": deliveries[3],
        "ward5": deliveries[5],
        "ward9": deliveries[9]
    })


print("COOPERATIVE PPO 100-EPISODE EVALUATION")
print("--------------------------------")

for metric in [
    "reward",
    "delivered",
    "utilization",
    "fairness_gap",
    "ward3",
    "ward5",
    "ward9"
]:
    values = [result[metric] for result in results]

    print(
        f"{metric}: "
        f"mean={np.mean(values):.4f}, "
        f"std={np.std(values):.4f}"
    )

print(
    "Episodes serving all 3 wards:",
    sum(
        all(r[f"ward{ward}"] > 0 for ward in [3, 5, 9])
        for r in results
    )
)

print("\nSUCCESS! Cooperative PPO evaluation completed.")
