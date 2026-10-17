
import torch

from ppo_allocation_env import PPOAllocationEnv
from ppo_network import PPOActorCritic


torch.manual_seed(42)

model = PPOActorCritic(observation_dim=7)

model.load_state_dict(
    torch.load(
        "data/processed/marl/aluva_ppo_fairness_improvement.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

env = PPOAllocationEnv()
observation, _ = env.reset(seed=42)

total_reward = 0.0
actions = []

while True:
    state = torch.tensor(
        observation,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        action_tensor, _, _, _ = model.get_action_and_value(state)
        action = action_tensor.item()

    actions.append(action)

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

fulfilment = {}

for ward in [3, 5, 9]:
    delivered = deliveries[ward]
    remaining = env.env.demand[ward]["water_litres"]

    original = delivered + remaining

    fulfilment[ward] = (
        delivered / original if original > 0 else 1.0
    )

fairness_gap = max(fulfilment.values()) - min(
    fulfilment.values()
)

print("7-FEATURE PPO EVALUATION")
print("--------------------------------")
print("Total reward:", round(total_reward, 4))
print("Total water delivered:", sum(deliveries.values()))
print("Ward deliveries:", deliveries)
print("Fulfilment ratios:", {
    ward: round(value, 4)
    for ward, value in fulfilment.items()
})
print("Fairness gap:", round(fairness_gap, 4))
print("Actions taken:", len(actions))
print("Wait actions:", actions.count(3))

env.close()

print("\nSUCCESS! 7-feature PPO evaluation completed.")
