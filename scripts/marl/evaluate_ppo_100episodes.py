
import numpy as np
import torch

from ppo_allocation_env import PPOAllocationEnv
from ppo_network import PPOActorCritic


model = PPOActorCritic(observation_dim=7)

model.load_state_dict(
    torch.load(
        "data/processed/marl/aluva_ppo_fairness_improvement.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

results = []

for episode in range(100):
    torch.manual_seed(1000 + episode)

    env = PPOAllocationEnv()
    observation, _ = env.reset(seed=1000 + episode)

    total_reward = 0.0

    while True:
        state = torch.tensor(
            observation,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(state)

        observation, reward, terminated, truncated, _ = (
            env.step(action.item())
        )

        total_reward += reward

        if terminated or truncated:
            break

    deliveries = {
        ward: env.env.delivered[ward]["water_litres"]
        for ward in [3, 5, 9]
    }

    ratios = []

    for ward in [3, 5, 9]:
        delivered = deliveries[ward]
        remaining = env.env.demand[ward]["water_litres"]

        original = delivered + remaining

        ratios.append(
            delivered / original if original > 0 else 1.0
        )

    results.append({
        "reward": total_reward,
        "delivered": sum(deliveries.values()),
        "fairness_gap": max(ratios) - min(ratios),
        "ward3": deliveries[3],
        "ward5": deliveries[5],
        "ward9": deliveries[9]
    })

    env.close()


print("PPO 100-EPISODE EVALUATION")
print("--------------------------------")

for metric in [
    "reward",
    "delivered",
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
    "Episodes delivering 500 L:",
    sum(r["delivered"] == 500 for r in results)
)

print(
    "Episodes serving all 3 wards:",
    sum(
        all(r[f"ward{ward}"] > 0 for ward in [3, 5, 9])
        for r in results
    )
)

print("\nSUCCESS! 100-episode evaluation completed.")
