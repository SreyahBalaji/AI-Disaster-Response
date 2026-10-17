
import numpy as np
import torch

from ppo_allocation_env import PPOAllocationEnv
from ppo_network import PPOActorCritic

torch.manual_seed(42)

trained_model = PPOActorCritic()

trained_model.load_state_dict(
    torch.load(
        "data/processed/marl/aluva_ppo_policy.pth",
        map_location="cpu",
        weights_only=True
    )
)

trained_model.eval()

untrained_model = PPOActorCritic()
untrained_model.eval()


def evaluate(model, episodes=20):
    rewards = []
    deliveries = []
    ward_deliveries = []

    for episode in range(episodes):
        env = PPOAllocationEnv()
        observation, _ = env.reset(seed=episode + 1000)

        total_reward = 0

        while True:
            state = torch.tensor(
                observation,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(state)
                action = torch.argmax(
                    logits, dim=-1
                ).item()

            observation, reward, terminated, truncated, _ = (
                env.step(action)
            )

            total_reward += reward

            if terminated or truncated:
                break

        delivered = {
            ward: env.env.delivered[ward]["water_litres"]
            for ward in [3, 5, 9]
        }

        rewards.append(total_reward)
        deliveries.append(sum(delivered.values()))
        ward_deliveries.append(delivered)

        env.close()

    return {
        "average_reward": float(np.mean(rewards)),
        "average_delivery": float(np.mean(deliveries)),
        "ward_delivery": ward_deliveries[0]
    }


print("PPO EVALUATION")
print("--------------------------------")

untrained = evaluate(untrained_model)
trained = evaluate(trained_model)

print("Untrained agent:")
print(untrained)

print("\nTrained PPO agent:")
print(trained)

print("\nSUCCESS! PPO evaluation completed.")
