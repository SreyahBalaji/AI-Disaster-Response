
from pathlib import Path

import torch
from torch.distributions import Categorical

from resource_allocation_env import ResourceAllocationEnv
from agent_observation import get_agent_observation
from agent_action import AgentActionSpace
from action_masking import (
    get_valid_action_mask,
    mask_action_logits
)
from policy_network import PolicyNetwork


torch.manual_seed(42)

model = PolicyNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

episodes = 100
max_steps = 100
gamma = 0.99

for episode in range(episodes):
    env = ResourceAllocationEnv()
    actions = AgentActionSpace(env)

    log_probs = []
    rewards = []

    for _ in range(max_steps):
        observation = get_agent_observation(env, "RC01")
        state = torch.tensor(observation, dtype=torch.float32)

        logits = model(state)

        mask = get_valid_action_mask(env, "RC01")
        masked_logits = mask_action_logits(logits, mask)

        distribution = Categorical(logits=masked_logits)
        action = distribution.sample()

        _, reward, info = actions.step(
            centre_id="RC01",
            action=int(action.item())
        )

        log_probs.append(distribution.log_prob(action))
        rewards.append(reward)

        # Stop when no useful allocation is possible.
        if not any(
            env.demand[ward]["water_litres"] > 0
            for ward in env.demand
        ):
            break

        if env.centres[0]["resources"]["water_litres"] == 0:
            break

    returns = []
    discounted_return = 0.0

    for reward in reversed(rewards):
        discounted_return = reward + gamma * discounted_return
        returns.insert(0, discounted_return)

    returns = torch.tensor(returns, dtype=torch.float32)

    if len(returns) > 1:
        returns = (
            (returns - returns.mean())
            / (returns.std(unbiased=False) + 1e-8)
        )

    loss = -torch.stack([
        log_prob * value
        for log_prob, value in zip(log_probs, returns)
    ]).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 20 == 0:
        total_delivered = sum(
            ward["water_litres"]
            for ward in env.delivered.values()
        )

        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Reward: {sum(rewards):.4f} | "
            f"Delivered: {total_delivered} L"
        )

output = Path("data/processed/marl")
output.mkdir(parents=True, exist_ok=True)

torch.save(
    model.state_dict(),
    output / "aluva_improved_rl_policy.pth"
)

print("\nSUCCESS! Improved RL training completed.")
print("Model saved: data/processed/marl/aluva_improved_rl_policy.pth")