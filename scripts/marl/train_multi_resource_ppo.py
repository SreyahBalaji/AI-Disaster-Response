
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F

from multi_resource_cooperative_env import MultiResourceCooperativeEnv
from multi_resource_ppo_network import MultiResourceCooperativePPO


# ============================================================
# 1. TRAINING CONFIGURATION
# ============================================================

SEED = 42
EPISODES = 300

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
PPO_EPOCHS = 4

LEARNING_RATE = 3e-4
ENTROPY_COEFFICIENT = 0.01
VALUE_COEFFICIENT = 0.5
MAX_GRAD_NORM = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 2. INITIALIZE ENVIRONMENT AND MODEL
# ============================================================

env = MultiResourceCooperativeEnv()

model = MultiResourceCooperativePPO(
    observation_size=env.observation_size,
    action_size=env.action_size,
    num_agents=len(env.centre_ids),
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

output_dir = Path("data/processed/marl")
output_dir.mkdir(parents=True, exist_ok=True)

reward_history = []
success_history = []
fairness_history = []


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def to_tensor(observations):
    """Convert centre observations into a PPO input tensor."""

    return torch.tensor(
        np.stack([
            observations[centre_id]
            for centre_id in env.centre_ids
        ]),
        dtype=torch.float32
    ).unsqueeze(0)


def get_masks_tensor():
    """Convert current valid-action masks into a tensor."""

    masks = env.get_action_masks()

    return torch.tensor(
        np.stack([
            masks[centre_id]
            for centre_id in env.centre_ids
        ]),
        dtype=torch.bool
    ).unsqueeze(0)


# ============================================================
# 4. TRAINING
# ============================================================

print("MULTI-RESOURCE MASKED COOPERATIVE PPO TRAINING")
print("==============================================")
print("Episodes:", EPISODES)
print("Agents:", len(env.centre_ids))
print("Resources:", len(env.resources))
print("Observation size:", env.observation_size)
print("Action size:", env.action_size)
print("Action masking: ENABLED")
print("==============================================\n")

for episode in range(EPISODES):

    # Alternate between static and dynamic disaster scenarios.
    env.dynamic_demand_enabled = (episode % 2 == 1)

    observations = env.reset()

    # IMPORTANT: These must reset for every episode.
    states = []
    actions_history = []
    log_probs_history = []
    mask_history = []
    rewards = []
    values = []
    dones = []

    episode_reward = 0.0
    successful_allocations = 0
    final_fairness_gap = 0.0

    # --------------------------------------------------------
    # 4.1 COLLECT EXPERIENCE
    # --------------------------------------------------------

    while True:

        state = to_tensor(observations)
        action_masks = get_masks_tensor()

        # Select only valid actions.
        with torch.no_grad():

            actions, log_probs, _, value = (
                model.get_actions_and_value(
                    state,
                    action_masks=action_masks
                )
            )

        action_dict = {
            centre_id: int(actions[0, i].item())
            for i, centre_id in enumerate(env.centre_ids)
        }

        next_observations, reward_dict, terminated, truncated, info = (
            env.step(action_dict)
        )

        done = terminated or truncated

        # Cooperative team reward.
        team_reward = float(
            reward_dict[env.centre_ids[0]]
        )

        states.append(state.squeeze(0))
        actions_history.append(actions.squeeze(0))
        log_probs_history.append(log_probs.squeeze(0))
        mask_history.append(action_masks.squeeze(0))

        rewards.append(team_reward)
        values.append(float(value.item()))
        dones.append(done)

        episode_reward += team_reward

        successful_allocations += info[
            "successful_actions"
        ]

        final_fairness_gap = info["fairness_gap"]

        observations = next_observations

        if done:
            break

    # --------------------------------------------------------
    # 4.2 GENERALIZED ADVANTAGE ESTIMATION
    # --------------------------------------------------------

    # Bootstrap the value when the episode ends because
    # of the time limit rather than a terminal state.
    if truncated and not terminated:
        with torch.no_grad():
            bootstrap_value = float(
                model.critic(
                    to_tensor(observations).reshape(1, -1)
                ).item()
            )
    else:
        bootstrap_value = 0.0

    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):

        if t + 1 < len(rewards):
            next_value = values[t + 1]
        else:
            next_value = bootstrap_value

        # A time-limit truncation still allows bootstrapping.
        if t == len(rewards) - 1:
            nonterminal = 0.0 if terminated else 1.0
        else:
            nonterminal = 1.0

        delta = (
            rewards[t]
            + GAMMA * next_value * nonterminal
            - values[t]
        )

        gae = (
            delta
            + GAMMA * GAE_LAMBDA * nonterminal * gae
        )

        advantages.insert(0, gae)

    # --------------------------------------------------------
    # 4.3 CONVERT EXPERIENCE TO TENSORS
    # --------------------------------------------------------

    states_tensor = torch.stack(states)

    actions_tensor = torch.stack(actions_history)

    old_log_probs = torch.stack(
        log_probs_history
    ).detach()

    masks_tensor = torch.stack(mask_history)

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
            advantages_tensor
            - advantages_tensor.mean()
        ) / (
            advantages_tensor.std(unbiased=False)
            + 1e-8
        )

    # --------------------------------------------------------
    # 4.4 PPO OPTIMIZATION
    # --------------------------------------------------------

    for _ in range(PPO_EPOCHS):

        _, new_log_probs, entropy, new_values = (
            model.get_actions_and_value(
                states_tensor,
                actions_tensor,
                action_masks=masks_tensor
            )
        )

        # Joint log probability of all three agents.
        old_joint = old_log_probs.sum(dim=1)
        new_joint = new_log_probs.sum(dim=1)

        ratio = torch.exp(
            (new_joint - old_joint).clamp(-20, 20)
        )

        surrogate1 = (
            ratio * advantages_tensor
        )

        surrogate2 = (
            torch.clamp(
                ratio,
                1.0 - CLIP_EPSILON,
                1.0 + CLIP_EPSILON
            )
            * advantages_tensor
        )

        actor_loss = -torch.min(
            surrogate1,
            surrogate2
        ).mean()

        critic_loss = F.mse_loss(
            new_values,
            returns_tensor
        )

        entropy_bonus = entropy.mean()

        total_loss = (
            actor_loss
            + VALUE_COEFFICIENT * critic_loss
            - ENTROPY_COEFFICIENT * entropy_bonus
        )

        optimizer.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRAD_NORM
        )

        optimizer.step()

    # --------------------------------------------------------
    # 4.5 RECORD TRAINING METRICS
    # --------------------------------------------------------

    reward_history.append(
        float(episode_reward)
    )

    success_history.append(
        int(successful_allocations)
    )

    fairness_history.append(
        float(final_fairness_gap)
    )

    if (episode + 1) % 30 == 0:

        avg_reward = np.mean(
            reward_history[-30:]
        )

        avg_success = np.mean(
            success_history[-30:]
        )

        avg_fairness = np.mean(
            fairness_history[-30:]
        )

        print(
            f"Episode {episode + 1}/{EPISODES} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Avg Allocations: {avg_success:.1f} | "
            f"Avg Fairness Gap: {avg_fairness:.4f}"
        )


# ============================================================
# 5. SAVE TRAINED MODEL
# ============================================================

model_path = (
    output_dir / "aluva_multi_resource_masked_ppo.pth"
)

history_path = (
    output_dir / "masked_multi_resource_training_history.json"
)

torch.save(
    model.state_dict(),
    model_path
)

training_history = {
    "model": "Masked Multi-Resource Cooperative PPO",
    "episodes": EPISODES,
    "seed": SEED,
    "agents": env.centre_ids,
    "resources": env.resources,
    "observation_size": env.observation_size,
    "action_size": env.action_size,
    "action_masking": True,
    "reward_history": reward_history,
    "successful_allocations": success_history,
    "fairness_history": fairness_history,
    "scenario_types": [
        "static",
        "dynamic_ward9_surge"
    ]
}

with open(
    history_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        training_history,
        file,
        indent=2
    )


# ============================================================
# 6. TRAINING SUMMARY
# ============================================================

print("\n==============================================")
print("TRAINING COMPLETED")
print("==============================================")

print("Model saved:", model_path)
print("Training history saved:", history_path)

print(
    "Final 30-episode average reward:",
    round(float(np.mean(reward_history[-30:])), 2)
)

print(
    "Final 30-episode average allocations:",
    round(float(np.mean(success_history[-30:])), 2)
)

print(
    "Final 30-episode average fairness gap:",
    round(float(np.mean(fairness_history[-30:])), 4)
)

print(
    "\nSUCCESS: Masked multi-resource PPO training completed."
)
