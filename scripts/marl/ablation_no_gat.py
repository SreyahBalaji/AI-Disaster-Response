
"""
RESPONDA — Phase 23: No-GAT Ablation

Compare two fairness-trained PPO policies:
  1. GAT-PPO v2: 38 original + 24 GAT features
  2. No-GAT PPO: 38 original + 24 zeros

Both have the same network dimensions and
training procedure.

Existing checkpoints remain unchanged.
"""

import json
import random
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from train_gat_ppo_v2 import FairnessGATEnv
from train_gat_multi_resource_ppo import (
    create_model,
    tensors_from_env,
    randomize_demand,
    update_model,
    select_baseline_actions,
)
from benchmark_gat_models import generate_scenarios


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "processed" / "marl"

NO_GAT_PATH = (
    MODEL_DIR
    / "aluva_no_gat_ablation_seed42.pth"
)

GAT_V2_PATH = (
    MODEL_DIR
    / "aluva_gat_fairness_ppo_v2.pth"
)

EPISODES = 200
SEED = 42


class NoGATEnv(FairnessGATEnv):
    """Remove graph information without changing dimensions."""

    def get_gat_context(self):
        return np.zeros(
            24,
            dtype=np.float32,
        )


def collect_episode(model, rng, dynamic):
    env = NoGATEnv()
    env.dynamic_demand_enabled = dynamic

    observations = env.reset()
    randomize_demand(env, rng)
    observations = env.get_observations()

    trajectory = []
    total_reward = 0.0

    while True:
        state, masks = tensors_from_env(
            env, observations
        )

        with torch.no_grad():
            actions, log_probs, _, values = (
                model.get_actions_and_value(
                    state,
                    action_masks=masks,
                )
            )

        action_dict = {
            centre: int(actions[0, i].item())
            for i, centre in enumerate(
                env.centre_ids
            )
        }

        observations, rewards, terminated, truncated, _ = (
            env.step(action_dict)
        )

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
            "done": bool(terminated or truncated),
        })

        total_reward += reward

        if terminated or truncated:
            break

    return trajectory, total_reward


def train_no_gat():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    model = create_model()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    rng = random.Random(SEED)

    trajectories = []
    rewards = []
    loss = 0.0

    print("=" * 58)
    print("TRAINING NO-GAT ABLATION")
    print("=" * 58)
    print("Episodes:", EPISODES)
    print("Seed:", SEED)
    print("Observation dimensions: 62")
    print("Graph inputs: 24 zeros")
    print()

    for episode in range(1, EPISODES + 1):
        trajectory, reward = collect_episode(
            model,
            rng,
            dynamic=(episode % 2 == 0),
        )

        trajectories.append(trajectory)
        rewards.append(reward)

        if episode % 8 == 0 or episode == EPISODES:
            loss = update_model(
                model,
                optimizer,
                trajectories,
            )
            trajectories.clear()

        if episode % 20 == 0:
            print(
                f"Episode {episode:3}/{EPISODES}"
                f" | Reward={mean(rewards[-20:]):.2f}"
                f" | Loss={loss:.3f}",
                flush=True,
            )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        NO_GAT_PATH,
    )

    model.eval()

    print()
    print("Saved:", NO_GAT_PATH)

    return model


def load_gat_v2():
    if not GAT_V2_PATH.exists():
        raise FileNotFoundError(
            f"Missing GAT-PPO v2: {GAT_V2_PATH}"
        )

    model = create_model()

    model.load_state_dict(
        torch.load(
            GAT_V2_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()
    return model


def evaluate_scenario(scenario, method, model=None):
    env = (
        NoGATEnv()
        if method == "no_gat"
        else FairnessGATEnv()
    )

    env.dynamic_demand_enabled = scenario["dynamic"]
    env.surge_ward = scenario["surge_ward"]
    env.surge_step = scenario["surge_step"]
    env.surge_resources = scenario[
        "surge_resources"
    ].copy()

    env.reset()

    for ward in env.wards:
        for resource in env.resources:
            env.env.demand[ward][resource] = (
                scenario["initial"][ward][resource]
            )
            env.env.delivered[ward][resource] = 0

    observations = env.get_observations()

    while True:
        if method == "baseline":
            actions = select_baseline_actions(env)

        else:
            state, masks = tensors_from_env(
                env, observations
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
                    selected[0, i].item()
                )
                for i, centre in enumerate(
                    env.centre_ids
                )
            }

        observations, _, terminated, truncated, _ = (
            env.step(actions)
        )

        if terminated or truncated:
            break

    coverage = []

    for ward in env.wards:
        for resource in env.resources:
            delivered = env.env.delivered[
                ward
            ][resource]

            remaining = env.env.demand[
                ward
            ][resource]

            total = delivered + remaining

            if total > 0:
                coverage.append(
                    100.0 * delivered / total
                )

    return {
        "scenario": scenario["id"],
        "method": method,
        "dynamic": scenario["dynamic"],
        "minimum_coverage": min(coverage),
        "average_coverage": mean(coverage),
        "zero_coverage_items": sum(
            value <= 1e-6
            for value in coverage
        ),
    }


def main():
    no_gat = train_no_gat()
    gat_v2 = load_gat_v2()

    models = {
        "baseline": None,
        "no_gat": no_gat,
        "gat_v2": gat_v2,
    }

    scenarios = generate_scenarios()
    results = []

    print()
    print("=" * 66)
    print("MATCHED-SCENARIO GAT ABLATION")
    print("=" * 66)

    for scenario in scenarios:
        for method, model in models.items():
            result = evaluate_scenario(
                scenario,
                method,
                model,
            )
            results.append(result)

    summary = {}

    for dynamic, label in (
        (False, "STATIC"),
        (True, "DYNAMIC"),
    ):
        print()
        print(label)

        summary[label] = {}

        for method in models:
            rows = [
                row for row in results
                if row["dynamic"] == dynamic
                and row["method"] == method
            ]

            minimum = mean(
                row["minimum_coverage"]
                for row in rows
            )

            average = mean(
                row["average_coverage"]
                for row in rows
            )

            zero = mean(
                row["zero_coverage_items"]
                for row in rows
            )

            summary[label][method] = {
                "mean_minimum_coverage": round(
                    minimum, 2
                ),
                "mean_average_coverage": round(
                    average, 2
                ),
                "mean_zero_coverage_items": round(
                    zero, 2
                ),
            }

            print(
                f"{method:10}"
                f" Min={minimum:6.2f}%"
                f" Avg={average:6.2f}%"
                f" Zero={zero:.2f}"
            )

    output = (
        MODEL_DIR
        / "gat_ablation_seed42_results.json"
    )

    output.write_text(
        json.dumps(
            {
                "training_seed": SEED,
                "training_episodes": EPISODES,
                "comparison": (
                    "GAT-PPO v2 vs no-GAT PPO "
                    "with identical architecture "
                    "and fairness reward"
                ),
                "summary": summary,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Results saved:", output)
    print()
    print(
        "Single training-seed ablation only. "
        "Additional seeds and independent "
        "evaluation are required."
    )


if __name__ == "__main__":
    main()
