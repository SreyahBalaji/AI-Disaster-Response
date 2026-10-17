
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ppo_allocation_env import PPOAllocationEnv
from ppo_network import PPOActorCritic


torch.manual_seed(42)
np.random.seed(42)

env = PPOAllocationEnv()
model = PPOActorCritic()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003
)

gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2

episodes = 200
update_epochs = 4

episode_rewards = []

print("PPO TRAINING")
print("--------------------------------")

for episode in range(1, episodes + 1):
    observation, _ = env.reset(seed=episode)

    observations = []
    actions = []
    old_log_probs = []
    rewards = []
    values = []
    dones = []

    total_reward = 0.0

    for step in range(env.max_steps):
        state = torch.tensor(
            observation,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = (
                model.get_action_and_value(state)
            )

        next_observation, reward, terminated, truncated, _ = (
            env.step(action.item())
        )

        observations.append(state.squeeze(0))
        actions.append(action.squeeze(0))
        old_log_probs.append(log_prob.squeeze(0))
        rewards.append(reward)
        values.append(value.item())
        dones.append(terminated or truncated)

        total_reward += reward
        observation = next_observation

        if terminated or truncated:
            break

    # Generalized Advantage Estimation (GAE).
    with torch.no_grad():
        final_state = torch.tensor(
            observation,
            dtype=torch.float32
        ).unsqueeze(0)

        _, final_value = model(final_state)

        next_value = (
            0.0 if dones[-1] else final_value.item()
        )

    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - float(dones[t])

        delta = (
            rewards[t]
            + gamma * next_value * nonterminal
            - values[t]
        )

        gae = (
            delta
            + gamma * gae_lambda * nonterminal * gae
        )

        advantages.insert(0, gae)
        next_value = values[t]

    observations = torch.stack(observations)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)

    advantages = torch.tensor(
        advantages,
        dtype=torch.float32
    )

    returns = advantages + torch.tensor(
        values,
        dtype=torch.float32
    )

    # Normalize advantages.
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)

    # PPO optimization.
    for _ in range(update_epochs):
        _, new_log_probs, entropy, new_values = (
            model.get_action_and_value(
                observations,
                actions
            )
        )

        ratios = torch.exp(
            new_log_probs - old_log_probs
        )

        surrogate1 = ratios * advantages

        surrogate2 = torch.clamp(
            ratios,
            1 - clip_epsilon,
            1 + clip_epsilon
        ) * advantages

        actor_loss = -torch.min(
            surrogate1, surrogate2
        ).mean()

        critic_loss = F.mse_loss(
            new_values, returns
        )

        entropy_bonus = entropy.mean()

        loss = (
            actor_loss
            + 0.5 * critic_loss
            - 0.01 * entropy_bonus
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.5
        )

        optimizer.step()

    episode_rewards.append(total_reward)

    if episode % 20 == 0:
        average_reward = np.mean(
            episode_rewards[-20:]
        )

        print(
            f"Episode {episode}/{episodes} | "
            f"Average reward: {average_reward:.2f}"
        )

output_path = Path(
    "data/processed/marl/aluva_ppo_fairness_improvement.pth"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(model.state_dict(), output_path)

env.close()

print("\nModel saved:", output_path)
print("SUCCESS! PPO training completed.")
