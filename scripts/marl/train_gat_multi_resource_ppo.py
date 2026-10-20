
"""
RESPONDA — Phase 23
Train three-agent, GAT-informed multi-resource PPO.

Input:  62 features per agent
Output: 16 masked actions per agent

The original 38-feature checkpoint is not modified.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gat_informed_multi_resource_env import (
    GATInformedMultiResourceEnv,
)
from multi_resource_ppo_network import (
    MultiResourceCooperativePPO,
)


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    ROOT
    / "data"
    / "processed"
    / "marl"
    / "aluva_gat_multi_resource_ppo.pth"
)

DEVICE = torch.device("cpu")


def create_model():
    return MultiResourceCooperativePPO(
        observation_size=62,
        action_size=16,
        num_agents=3,
    ).to(DEVICE)


def tensors_from_env(env, observations):
    state = torch.as_tensor(
        np.stack(
            [
                observations[centre]
                for centre in env.centre_ids
            ]
        ),
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    masks = env.get_action_masks()

    mask = torch.as_tensor(
        np.stack(
            [
                masks[centre]
                for centre in env.centre_ids
            ]
        ),
        dtype=torch.bool,
        device=DEVICE,
    ).unsqueeze(0)

    return state, mask


def randomize_demand(env, rng):
    """
    Create varying simulated training scenarios.

    This does not modify stored datasets.
    """
    for ward in env.wards:
        for resource in env.resources:
            original = env.env.demand[ward][resource]

            if original > 0:
                multiplier = rng.uniform(0.6, 1.4)

                env.env.demand[ward][resource] = max(
                    1,
                    int(round(original * multiplier)),
                )


def collect_episode(model, rng, dynamic=False):
    env = GATInformedMultiResourceEnv()

    env.dynamic_demand_enabled = dynamic
    observations = env.reset()

    randomize_demand(env, rng)
    observations = env.get_observations()

    trajectory = []
    total_reward = 0.0

    while True:
        state, masks = tensors_from_env(
            env,
            observations,
        )

        with torch.no_grad():
            (
                actions,
                log_probs,
                _,
                values,
            ) = model.get_actions_and_value(
                state,
                action_masks=masks,
            )

        action_dict = {
            centre: int(actions[0, index].item())
            for index, centre in enumerate(
                env.centre_ids
            )
        }

        (
            next_observations,
            rewards,
            terminated,
            truncated,
            _,
        ) = env.step(action_dict)

        # The environment uses a shared team reward.
        team_reward = float(
            rewards[env.centre_ids[0]]
        )

        done = bool(terminated or truncated)

        trajectory.append({
            "state": state.squeeze(0),
            "mask": masks.squeeze(0),
            "actions": actions.squeeze(0),
            "log_prob": (
                log_probs.sum(dim=1).squeeze(0)
            ),
            "value": values.squeeze(0),
            "reward": team_reward,
            "done": done,
        })

        total_reward += team_reward
        observations = next_observations

        if done:
            break

    return trajectory, total_reward


def calculate_advantages(
    trajectory,
    gamma=0.99,
    gae_lambda=0.95,
):
    advantages = []
    next_advantage = 0.0
    next_value = 0.0

    for item in reversed(trajectory):
        value = float(item["value"].item())

        continuation = (
            0.0 if item["done"] else 1.0
        )

        delta = (
            item["reward"]
            + gamma * next_value * continuation
            - value
        )

        advantage = (
            delta
            + gamma
            * gae_lambda
            * continuation
            * next_advantage
        )

        advantages.append(advantage)
        next_advantage = advantage
        next_value = value

    advantages.reverse()

    returns = [
        advantage + float(item["value"].item())
        for advantage, item in zip(
            advantages,
            trajectory,
        )
    ]

    return advantages, returns


def update_model(
    model,
    optimizer,
    trajectories,
    epochs=4,
    batch_size=128,
):
    states = []
    masks = []
    actions = []
    old_log_probs = []
    advantages = []
    returns = []

    for trajectory in trajectories:
        episode_advantages, episode_returns = (
            calculate_advantages(trajectory)
        )

        for index, item in enumerate(trajectory):
            states.append(item["state"])
            masks.append(item["mask"])
            actions.append(item["actions"])
            old_log_probs.append(item["log_prob"])
            advantages.append(
                episode_advantages[index]
            )
            returns.append(
                episode_returns[index]
            )

    states = torch.stack(states)
    masks = torch.stack(masks)
    actions = torch.stack(actions)

    old_log_probs = torch.stack(
        old_log_probs
    ).detach()

    advantages = torch.tensor(
        advantages,
        dtype=torch.float32,
    )

    returns = torch.tensor(
        returns,
        dtype=torch.float32,
    )

    advantages = (
        advantages - advantages.mean()
    ) / (
        advantages.std(unbiased=False) + 1e-8
    )

    sample_count = states.shape[0]
    losses = []

    for _ in range(epochs):
        order = torch.randperm(sample_count)

        for start in range(
            0,
            sample_count,
            batch_size,
        ):
            indices = order[
                start:start + batch_size
            ]

            (
                _,
                new_log_probs,
                entropy,
                values,
            ) = model.get_actions_and_value(
                states[indices],
                actions=actions[indices],
                action_masks=masks[indices],
            )

            # Treat the three agents' joint action
            # as one team decision for PPO clipping.
            new_joint_log_prob = (
                new_log_probs.sum(dim=1)
            )

            ratio = torch.exp(
                new_joint_log_prob
                - old_log_probs[indices]
            )

            sampled_advantages = advantages[
                indices
            ]

            unclipped = (
                ratio * sampled_advantages
            )

            clipped = torch.clamp(
                ratio,
                0.8,
                1.2,
            ) * sampled_advantages

            actor_loss = -torch.min(
                unclipped,
                clipped,
            ).mean()

            critic_loss = F.mse_loss(
                values,
                returns[indices],
            )

            entropy_bonus = (
                entropy.mean()
            )

            loss = (
                actor_loss
                + 0.5 * critic_loss
                - 0.01 * entropy_bonus
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.5,
            )

            optimizer.step()

            losses.append(
                float(loss.item())
            )

    return float(np.mean(losses))


def select_baseline_actions(env):
    """Existing fairness-baseline decision rule."""
    masks = env.get_action_masks()
    actions = {}

    for centre_id in env.centre_ids:
        candidates = []

        for action, (ward, resource) in enumerate(
            env.action_map
        ):
            if not masks[centre_id][action]:
                continue

            delivered = (
                env.env.delivered[ward][resource]
            )

            remaining = (
                env.env.demand[ward][resource]
            )

            total = delivered + remaining

            satisfaction = (
                delivered / total
                if total > 0 else 1.0
            )

            priority = (
                (1.0 - satisfaction)
                * env.vulnerability[ward]
            )

            candidates.append(
                (priority, action)
            )

        if candidates:
            candidates.sort(reverse=True)
            actions[centre_id] = candidates[0][1]
        else:
            actions[centre_id] = env.wait_action

    return actions


def evaluate(model, dynamic, algorithm):
    env = GATInformedMultiResourceEnv()

    env.dynamic_demand_enabled = dynamic
    observations = env.reset()

    while True:
        if algorithm == "baseline":
            actions = select_baseline_actions(env)

        else:
            state, mask = tensors_from_env(
                env,
                observations,
            )

            with torch.no_grad():
                logits = model.actor(state)

                logits = logits.masked_fill(
                    ~mask,
                    torch.finfo(logits.dtype).min,
                )

                selected = logits.argmax(dim=-1)

            actions = {
                centre: int(
                    selected[0, index].item()
                )
                for index, centre in enumerate(
                    env.centre_ids
                )
            }

        (
            observations,
            _,
            terminated,
            truncated,
            _,
        ) = env.step(actions)

        if terminated or truncated:
            break

    coverage = []

    for ward in env.wards:
        for resource in env.resources:
            delivered = (
                env.env.delivered[ward][resource]
            )

            remaining = (
                env.env.demand[ward][resource]
            )

            total = delivered + remaining

            if total > 0:
                coverage.append(
                    delivered / total
                )

    return {
        "minimum_coverage": round(
            min(coverage) * 100,
            2,
        ),
        "average_coverage": round(
            np.mean(coverage) * 100,
            2,
        ),
        "steps": env.steps,
    }


def print_evaluation(model):
    print()
    print("=" * 56)
    print("GAT-PPO VS FAIRNESS BASELINE")
    print("=" * 56)

    for dynamic in (False, True):
        scenario = (
            "DYNAMIC"
            if dynamic
            else "STATIC"
        )

        for algorithm in (
            "baseline",
            "gat_ppo",
        ):
            metrics = evaluate(
                model,
                dynamic,
                algorithm,
            )

            print(
                f"{scenario:8} "
                f"{algorithm:9} "
                f"Min={metrics['minimum_coverage']:6.2f}% "
                f"Avg={metrics['average_coverage']:6.2f}% "
                f"Steps={metrics['steps']}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error(
            "--episodes must be positive"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_num_threads(2)

    model = create_model()

    if args.eval_only:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"Model not found: {CHECKPOINT}"
            )

        model.load_state_dict(
            torch.load(
                CHECKPOINT,
                map_location="cpu",
                weights_only=True,
            )
        )

        model.eval()
        print_evaluation(model)
        return

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    rng = random.Random(args.seed)

    all_trajectories = []
    rewards = []

    print("=" * 56)
    print("PHASE 23 — TRAIN GAT-INFORMED PPO")
    print("=" * 56)
    print("Observations: 62")
    print("Actions: 16")
    print("Agents: 3")
    print("Episodes:", args.episodes)
    print("Device:", DEVICE)
    print()

    for episode in range(
        1,
        args.episodes + 1,
    ):
        dynamic = episode % 2 == 0

        trajectory, reward = collect_episode(
            model,
            rng,
            dynamic=dynamic,
        )

        all_trajectories.append(
            trajectory
        )

        rewards.append(
            reward
        )

        should_update = (
            episode % 8 == 0
            or episode == args.episodes
        )

        if should_update:
            loss = update_model(
                model,
                optimizer,
                all_trajectories,
            )

            all_trajectories = []

        if (
            episode % 10 == 0
            or episode == args.episodes
        ):
            recent = rewards[-10:]

            print(
                f"Episode {episode:3}/{args.episodes} "
                f"| Mean reward: {np.mean(recent):9.2f} "
                f"| Loss: {loss:9.4f}",
                flush=True,
            )

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        CHECKPOINT,
    )

    metadata = {
        "model": "GAT-informed cooperative PPO",
        "observation_size": 62,
        "action_size": 16,
        "num_agents": 3,
        "wards": [3, 5, 9],
        "training_episodes": args.episodes,
        "seed": args.seed,
        "embedding_source": (
            "aluva_gat_embeddings.csv"
        ),
        "evaluation_status": (
            "Exploratory; not approved for dispatch"
        ),
    }

    CHECKPOINT.with_suffix(".json").write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("MODEL SAVED:", CHECKPOINT)

    model.eval()
    print_evaluation(model)

    print()
    print(
        "Training complete. These evaluations "
        "are preliminary, not publication results."
    )


if __name__ == "__main__":
    main()
