
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cooperative_allocation_env import CooperativeAllocationEnv
from cooperative_ppo_network import CooperativePPO


torch.manual_seed(42)
np.random.seed(42)

env = CooperativeAllocationEnv()
model = CooperativePPO()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003
)

episodes = 400
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
ppo_epochs = 4

rewards_history = []
static_rewards = []
dynamic_rewards = []


def to_tensor(observations):
    return torch.tensor(
        np.stack([
            observations[centre_id]
            for centre_id in env.centre_ids
        ]),
        dtype=torch.float32
    ).unsqueeze(0)


print("DYNAMIC COOPERATIVE PPO TRAINING")
print("--------------------------------")

for episode in range(episodes):

    # Alternate between static and dynamic scenarios.
    is_dynamic = episode % 2 == 1

    env.dynamic_demand_enabled = is_dynamic

    observations = env.reset()

    states = []
    actions_history = []
    log_probs_history = []
    rewards = []
    values = []
    dones = []

    episode_reward = 0.0

    while True:
        state = to_tensor(observations)

        with torch.no_grad():
            actions, log_probs, _, value = (
                model.get_actions_and_value(state)
            )

        action_dict = {
            centre_id: int(actions[0, i].item())
            for i, centre_id in enumerate(env.centre_ids)
        }

        next_observations, reward_dict, terminated, truncated, _ = (
            env.step(action_dict)
        )

        team_reward = reward_dict[env.centre_ids[0]]

        states.append(state.squeeze(0))
        actions_history.append(actions.squeeze(0))
        log_probs_history.append(log_probs.squeeze(0))
        rewards.append(team_reward)
        values.append(value.item())
        dones.append(terminated or truncated)

        episode_reward += team_reward
        observations = next_observations

        if terminated or truncated:
            break

    # Generalized Advantage Estimation
    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        next_value = (
            0.0 if t == len(rewards) - 1
            else values[t + 1]
        )

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

    states_tensor = torch.stack(states)
    actions_tensor = torch.stack(actions_history)
    old_log_probs = torch.stack(log_probs_history)

    advantages_tensor = torch.tensor(
        advantages,
        dtype=torch.float32
    )

    returns_tensor = (
        advantages_tensor
        + torch.tensor(values, dtype=torch.float32)
    )

    if len(advantages_tensor) > 1:
        advantages_tensor = (
            advantages_tensor - advantages_tensor.mean()
        ) / (
            advantages_tensor.std(unbiased=False) + 1e-8
        )

    # PPO updates
    for _ in range(ppo_epochs):
        _, new_log_probs, entropy, new_values = (
            model.get_actions_and_value(
                states_tensor,
                actions_tensor
            )
        )

        old_joint_log_prob = old_log_probs.sum(dim=1)
        new_joint_log_prob = new_log_probs.sum(dim=1)

        ratio = torch.exp(
            new_joint_log_prob - old_joint_log_prob
        )

        surrogate1 = ratio * advantages_tensor

        surrogate2 = torch.clamp(
            ratio,
            1.0 - clip_epsilon,
            1.0 + clip_epsilon
        ) * advantages_tensor

        actor_loss = -torch.min(
            surrogate1,
            surrogate2
        ).mean()

        critic_loss = F.mse_loss(
            new_values,
            returns_tensor
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

    rewards_history.append(episode_reward)

    if is_dynamic:
        dynamic_rewards.append(episode_reward)
    else:
        static_rewards.append(episode_reward)

    if (episode + 1) % 40 == 0:
        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Average team reward: "
            f"{np.mean(rewards_history[-40:]):.2f}"
        )


output_path = Path(
    "data/processed/marl/aluva_dynamic_cooperative_ppo.pth"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    output_path
)

print("\nAverage static reward:", round(np.mean(static_rewards), 2))
print("Average dynamic reward:", round(np.mean(dynamic_rewards), 2))

print("\nModel saved:", output_path)
print("SUCCESS! Dynamic cooperative PPO training completed.")
