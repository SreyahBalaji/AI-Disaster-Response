
import torch

from ppo_allocation_env import PPOAllocationEnv
from ppo_network import PPOActorCritic


def load_model(path):
    model = PPOActorCritic()

    model.load_state_dict(
        torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()
    return model


def evaluate(model):
    env = PPOAllocationEnv()
    observation, _ = env.reset(seed=42)

    total_reward = 0.0

    while True:
        state = torch.tensor(
            observation,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(state)
            action = torch.argmax(logits, dim=-1).item()

        observation, reward, terminated, truncated, _ = (
            env.step(action)
        )

        total_reward += reward

        if terminated or truncated:
            break

    deliveries = {
        ward: env.env.delivered[ward]["water_litres"]
        for ward in [3, 5, 9]
    }

    fulfilment = []

    for ward in [3, 5, 9]:
        delivered = deliveries[ward]
        remaining = env.env.demand[ward]["water_litres"]

        original = delivered + remaining
        fulfilment.append(delivered / original if original else 1.0)

    result = {
        "total_reward": round(total_reward, 4),
        "total_delivered": sum(deliveries.values()),
        "ward_deliveries": deliveries,
        "fairness_gap": round(
            max(fulfilment) - min(fulfilment), 4
        )
    }

    env.close()
    return result


original = load_model(
    "data/processed/marl/aluva_ppo_policy.pth"
)

fair = load_model(
    "data/processed/marl/aluva_fair_ppo_policy.pth"
)

print("PPO FAIRNESS COMPARISON")
print("--------------------------------")

print("\nOriginal PPO:")
print(evaluate(original))

print("\nFairness-aware PPO:")
print(evaluate(fair))

print("\nSUCCESS! PPO fairness comparison completed.")
