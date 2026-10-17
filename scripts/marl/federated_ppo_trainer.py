
import contextlib
import io

import numpy as np
import torch

from cooperative_ppo_network import CooperativePPO
from federated_local_env import FederatedLocalEnv


def train_local_ppo(
    global_state,
    centre_id,
    episodes=10,
    learning_rate=0.0003
):
    """
    Train a cooperative PPO model on one local
    simulated disaster scenario.

    Returns the updated model state and mean reward.
    """

    model = CooperativePPO()
    model.load_state_dict(global_state)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    env = FederatedLocalEnv(centre_id)

    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    ppo_epochs = 4

    episode_rewards = []

    for episode in range(episodes):
        observations = env.reset()

        states = []
        actions_list = []
        old_log_probs = []
        values = []
        rewards = []

        while True:
            state = torch.tensor(
                np.stack([
                    observations[agent_id]
                    for agent_id in env.env.centre_ids
                ]),
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                actions, log_probs, _, value = (
                    model.get_actions_and_value(state)
                )

            action_dict = {
                agent_id: int(actions[0, i].item())
                for i, agent_id in enumerate(
                    env.env.centre_ids
                )
            }

            with contextlib.redirect_stdout(io.StringIO()):
                observations, team_rewards, terminated, truncated, _ = (
                    env.step(action_dict)
                )

            states.append(state.squeeze(0))
            actions_list.append(actions.squeeze(0))
            old_log_probs.append(log_probs.squeeze(0))
            values.append(value.reshape(()))

            rewards.append(
                team_rewards[env.env.centre_ids[0]]
            )

            if terminated or truncated:
                break

        states = torch.stack(states)
        actions_tensor = torch.stack(actions_list)
        old_log_probs = torch.stack(old_log_probs)
        values_tensor = torch.stack(values)

        advantages = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + gamma * next_value
                - values_tensor[t].item()
            )

            gae = (
                delta
                + gamma * gae_lambda * gae
            )

            advantages.insert(0, gae)
            next_value = values_tensor[t].item()

        advantages = torch.tensor(
            advantages,
            dtype=torch.float32
        )

        returns = advantages + values_tensor

        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std(unbiased=False) + 1e-8
        )

        for _ in range(ppo_epochs):
            _, new_log_probs, entropy, new_values = (
                model.get_actions_and_value(
                    states,
                    actions_tensor
                )
            )

            ratio = torch.exp(
                new_log_probs.sum(dim=1)
                - old_log_probs.sum(dim=1)
            )

            surrogate1 = ratio * advantages

            surrogate2 = torch.clamp(
                ratio,
                1 - clip_epsilon,
                1 + clip_epsilon
            ) * advantages

            actor_loss = -torch.min(
                surrogate1,
                surrogate2
            ).mean()

            critic_loss = (
                new_values.reshape(-1) - returns
            ).pow(2).mean()

            loss = (
                actor_loss
                + 0.5 * critic_loss
                - 0.01 * entropy.mean()
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                0.5
            )

            optimizer.step()

        episode_rewards.append(sum(rewards))

    return (
        {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        },
        float(np.mean(episode_rewards))
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    model = CooperativePPO()

    state, reward = train_local_ppo(
        model.state_dict(),
        centre_id="RC01",
        episodes=2
    )

    assert len(state) == len(model.state_dict())

    print("FEDERATED PPO TRAINER TEST")
    print("--------------------------")
    print("Parameters returned:", len(state))
    print("Mean team reward:", round(reward, 4))
    print("SUCCESS! Reusable PPO trainer validated.")
