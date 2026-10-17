
import contextlib
import io

import numpy as np
import torch

from cooperative_allocation_env import CooperativeAllocationEnv
from cooperative_ppo_network import CooperativePPO


MODEL_PATH = (
    "data/processed/marl/"
    "aluva_dynamic_cooperative_ppo.pth"
)

model = CooperativePPO()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()


def evaluate(dynamic=False, episodes=100):
    results = []

    for episode in range(episodes):
        torch.manual_seed(1000 + episode)

        env = CooperativeAllocationEnv()
        env.dynamic_demand_enabled = dynamic

        observations = env.reset()
        total_reward = 0.0

        while True:
            state = torch.tensor(
                np.stack([
                    observations[centre_id]
                    for centre_id in env.centre_ids
                ]),
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                actions, _, _, _ = (
                    model.get_actions_and_value(state)
                )

            action_dict = {
                centre_id: int(actions[0, i].item())
                for i, centre_id in enumerate(env.centre_ids)
            }

            # Suppress repeated disaster-report messages.
            with contextlib.redirect_stdout(io.StringIO()):
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

        ratios = []

        for ward in env.wards:
            delivered = deliveries[ward]
            remaining = env.env.demand[ward]["water_litres"]
            original = delivered + remaining

            ratios.append(
                delivered / original if original > 0 else 1.0
            )

        total_delivered = sum(deliveries.values())
        total_demand = 1496 if dynamic else 996

        results.append({
            "delivered": total_delivered,
            "satisfaction": total_delivered / total_demand,
            "fairness_gap": max(ratios) - min(ratios),
            "ward3": deliveries[3],
            "ward5": deliveries[5],
            "ward9": deliveries[9],
            "reward": total_reward,
            "steps": env.steps
        })

    return {
        key: float(np.mean([r[key] for r in results]))
        for key in results[0]
    }


print("RETRAINED COOPERATIVE PPO EVALUATION")
print("--------------------------------")

static = evaluate(dynamic=False)
dynamic = evaluate(dynamic=True)

print("\nStatic demand:")
for key, value in static.items():
    print(f"{key}: {value:.4f}")

print("\nDynamic demand (Ward 9 +500 L):")
for key, value in dynamic.items():
    print(f"{key}: {value:.4f}")

print("\nPrevious PPO dynamic results:")
print("Delivered: 1001.5000")
print("Fairness gap: 0.6906")
print("Ward 9: 221.5000")

print("\nTime-constrained baseline:")
print("Delivered: 1300")
print("Fairness gap: 0.0091")
print("Ward 9: 620")

print("\nSUCCESS! Retrained PPO evaluation completed.")
