
from pathlib import Path
import contextlib
import io

import numpy as np
import torch

from cooperative_ppo_network import CooperativePPO
from federated_local_env import FederatedLocalEnv


BASE_PATH = Path(
    "data/processed/marl/federated/global_cooperative_ppo.pth"
)

OUTPUT_DIR = Path("data/processed/marl/federated")

EPISODES = 20
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
PPO_EPOCHS = 4


def train_centre(centre_id, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CooperativePPO()
    model.load_state_dict(
        torch.load(
            BASE_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0003
    )

    env = FederatedLocalEnv(centre_id)

    print(f"\nTraining {centre_id}")

    for episode in range(EPISODES):
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
                + GAMMA * next_value
                - values_tensor[t].item()
            )

            gae = (
                delta
                + GAMMA * GAE_LAMBDA * gae
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

        for _ in range(PPO_EPOCHS):
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
                1 - CLIP_EPSILON,
                1 + CLIP_EPSILON
            ) * advantages

            actor_loss = -torch.min(
                surrogate1,
                surrogate2
            ).mean()

            critic_loss = (
                new_values.reshape(-1) - returns
            ).pow(2).mean()

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
                0.5
            )

            optimizer.step()

        if (episode + 1) % 5 == 0:
            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"Team reward: {sum(rewards):.2f}"
            )

    output_path = OUTPUT_DIR / f"{centre_id}_ppo_trained.pth"

    torch.save(
        model.state_dict(),
        output_path
    )

    print(f"Saved: {output_path}")


print("REMAINING FEDERATED PPO LOCAL TRAINING")
print("--------------------------------------")

train_centre("RC02", seed=43)
train_centre("RC03", seed=44)

print("\nSUCCESS! All three local PPO models trained.")
