
"""
RESPONDA — Phase 23
Fairness-Aware GAT-PPO Training Experiment

Tests whether a reward based on ward-resource
coverage improves the existing GAT-PPO approach.

Does NOT overwrite previous models.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from gat_informed_multi_resource_env import (
    GATInformedMultiResourceEnv,
)

from train_gat_multi_resource_ppo import (
    create_model,
    tensors_from_env,
    randomize_demand,
    update_model,
    select_baseline_actions,
)


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    ROOT
    / "data"
    / "processed"
    / "marl"
    / "aluva_gat_fairness_ppo_v2.pth"
)


class FairnessGATEnv(GATInformedMultiResourceEnv):
    def coverage_score(self):
        """
        Sum vulnerability-weighted satisfaction.

        Resource types are compared using their
        individual satisfaction ratios, NOT by
        adding litres, meals and rescue cases.
        """

        total = 0.0

        for ward in self.wards:
            weight = 1.0 + self.vulnerability[ward]

            for resource in self.resources:
                delivered = self.env.delivered[
                    ward
                ][resource]

                remaining = self.env.demand[
                    ward
                ][resource]

                requested = delivered + remaining

                if requested <= 0:
                    continue

                total += (
                    weight * delivered / requested
                )

        return total

    def uncovered_ratios(self):
        gaps = []

        for ward in self.wards:
            for resource in self.resources:
                delivered = self.env.delivered[
                    ward
                ][resource]

                remaining = self.env.demand[
                    ward
                ][resource]

                requested = delivered + remaining

                if requested > 0:
                    gaps.append(
                        remaining / requested
                    )

        return gaps

    def step(self, actions):
        previous_score = self.coverage_score()

        (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = super().step(actions)

        new_score = self.coverage_score()

        base_reward = float(
            rewards[self.centre_ids[0]]
        )

        # Reward progress toward satisfying
        # each ward-resource need.
        progress_reward = (
            30.0
            * (new_score - previous_score)
        )

        # Retain a small part of the original
        # successful-action/team reward.
        shaped_reward = (
            0.10 * base_reward
            + progress_reward
        )

        # At episode end, penalize unequal
        # and unfinished resource coverage.
        if terminated or truncated:
            gaps = self.uncovered_ratios()

            if gaps:
                mean_gap = float(np.mean(gaps))
                worst_gap = float(max(gaps))

                shaped_reward -= (
                    15.0 * mean_gap
                    + 20.0 * worst_gap
                )

        info["original_team_reward"] = base_reward
        info["shaped_team_reward"] = shaped_reward

        team_rewards = {
            centre: shaped_reward
            for centre in self.centre_ids
        }

        return (
            observations,
            team_rewards,
            terminated,
            truncated,
            info,
        )


def collect_episode(model, rng, dynamic):
    env = FairnessGATEnv()

    env.dynamic_demand_enabled = dynamic
    observations = env.reset()

    randomize_demand(env, rng)
    observations = env.get_observations()

    trajectory = []
    episode_reward = 0.0

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

        chosen = {
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
        ) = env.step(chosen)

        reward = float(
            rewards[env.centre_ids[0]]
        )

        trajectory.append({
            "state": state.squeeze(0),
            "mask": masks.squeeze(0),
            "actions": actions.squeeze(0),
            "log_prob": (
                log_probs.sum(dim=1).squeeze(0)
            ),
            "value": values.squeeze(0),
            "reward": reward,
            "done": terminated or truncated,
        })

        episode_reward += reward
        observations = next_observations

        if terminated or truncated:
            break

    return trajectory, episode_reward


def evaluate(model, algorithm, dynamic):
    env = FairnessGATEnv()

    env.dynamic_demand_enabled = dynamic
    observations = env.reset()

    while True:
        if algorithm == "baseline":
            actions = select_baseline_actions(env)

        else:
            state, masks = tensors_from_env(
                env,
                observations,
            )

            with torch.no_grad():
                logits = model.actor(state)

                logits = logits.masked_fill(
                    ~masks,
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

    rows = []
    coverage = []

    for ward in env.wards:
        for resource in env.resources:
            delivered = env.env.delivered[
                ward
            ][resource]

            remaining = env.env.demand[
                ward
            ][resource]

            requested = delivered + remaining

            if requested <= 0:
                continue

            percentage = (
                100.0 * delivered / requested
            )

            coverage.append(percentage)

            rows.append({
                "ward": ward,
                "resource": resource,
                "coverage": round(
                    percentage, 2
                ),
                "unmet": int(remaining),
            })

    return {
        "minimum_coverage": round(
            min(coverage), 2
        ),
        "average_coverage": round(
            float(np.mean(coverage)), 2
        ),
        "steps": env.steps,
        "shortages": [
            row
            for row in rows
            if row["unmet"] > 0
        ],
    }


def compare(model):
    print()
    print("=" * 65)
    print("FAIRNESS-TRAINED GAT-PPO VS BASELINE")
    print("=" * 65)

    results = {}

    for dynamic in [False, True]:
        scenario = (
            "DYNAMIC" if dynamic else "STATIC"
        )

        results[scenario] = {}

        for method in ["baseline", "gat_ppo_v2"]:
            result = evaluate(
                model,
                method,
                dynamic,
            )

            results[scenario][method] = result

            print(
                f"{scenario:8} "
                f"{method:12} "
                f"Min={result['minimum_coverage']:6.2f}% "
                f"Avg={result['average_coverage']:6.2f}% "
                f"Steps={result['steps']}"
            )

            if dynamic:
                for shortage in result["shortages"]:
                    print(
                        "  Shortage:",
                        f"Ward {shortage['ward']}",
                        shortage["resource"],
                        f"unmet={shortage['unmet']}",
                    )

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error(
            "Episode count must be positive"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    model = create_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    rng = random.Random(args.seed)

    trajectories = []
    recent_rewards = []
    last_loss = 0.0

    print("=" * 65)
    print("PHASE 23 — FAIRNESS-AWARE GAT-PPO TRAINING")
    print("=" * 65)

    print("Observation features: 62")
    print("Actions: 16")
    print("Relief centres: 3")
    print("Episodes:", args.episodes)
    print("Seed:", args.seed)

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

        trajectories.append(trajectory)
        recent_rewards.append(reward)

        if (
            episode % 8 == 0
            or episode == args.episodes
        ):
            last_loss = update_model(
                model,
                optimizer,
                trajectories,
            )

            trajectories.clear()

        if (
            episode % 20 == 0
            or episode == args.episodes
        ):
            print(
                f"Episode {episode:3}/{args.episodes}"
                f" | Mean reward="
                f"{np.mean(recent_rewards[-20:]):8.2f}"
                f" | Loss={last_loss:9.3f}",
                flush=True,
            )

    model.eval()
    results = compare(model)

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        CHECKPOINT,
    )

    metadata = {
        "model": "Fairness-aware GAT-PPO v2",
        "observations": 62,
        "actions": 16,
        "episodes": args.episodes,
        "seed": args.seed,
        "reward": (
            "Ward-resource coverage improvement "
            "with terminal unmet-demand penalty"
        ),
        "evaluation": results,
        "approved_for_real_dispatch": False,
    }

    CHECKPOINT.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("Saved new model:", CHECKPOINT)
    print(
        "Original PPO and GAT-PPO checkpoints "
        "remain unchanged."
    )
    print()
    print(
        "Research note: This is one experimental "
        "run, not a validated performance claim."
    )


if __name__ == "__main__":
    main()
