
"""
RESPONDA — Phase 27
Federated GAT-Informed Multi-Resource PPO

Three simulated federated participants:
RC01, RC02, RC03.

Each participant:
- Starts with the same global model
- Trains using its own disaster scenario
- Returns updated model weights

Server:
- Averages model weights using FedAvg
- Evaluates the global model against baseline
- Saves a separate experimental checkpoint

No changes to existing models or dashboard.
No real-world resource dispatch.
"""

import argparse
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
)

from benchmark_gat_models import (
    generate_scenarios,
    simulate,
)


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "marl"
    / "federated_multi_resource"
)

MODEL_PATH = (
    OUTPUT_DIR
    / "fedavg_gat_multi_resource_ppo.pth"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "fedavg_gat_multi_resource_results.json"
)

CENTRES = ["RC01", "RC02", "RC03"]

# Each participant receives a different
# simulated local disaster distribution.
LOCAL_SCENARIOS = {
    "RC01": {
        "surge_ward": 3,
        "surge_resources": {
            "water_litres": 300,
            "food_meals": 80,
            "rescue_cases": 15,
            "shelter_households": 5,
            "medical_cases": 4,
        },
    },
    "RC02": {
        "surge_ward": 5,
        "surge_resources": {
            "water_litres": 350,
            "food_meals": 180,
            "rescue_cases": 8,
            "shelter_households": 12,
            "medical_cases": 6,
        },
    },
    "RC03": {
        "surge_ward": 9,
        "surge_resources": {
            "water_litres": 500,
            "food_meals": 100,
            "rescue_cases": 5,
            "shelter_households": 20,
            "medical_cases": 15,
        },
    },
}


def clone_weights(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def create_local_environment(centre_id, rng):
    configuration = LOCAL_SCENARIOS[centre_id]

    env = FairnessGATEnv()

    env.dynamic_demand_enabled = True
    env.surge_step = 10

    env.surge_ward = configuration["surge_ward"]

    env.surge_resources = dict(
        configuration["surge_resources"]
    )

    env.reset()

    # Vary initial demand between local episodes
    # without changing the stored project datasets.
    randomize_demand(env, rng)

    return env


def collect_local_episode(model, centre_id, rng):
    env = create_local_environment(
        centre_id,
        rng,
    )

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
            agent: int(actions[0, index].item())
            for index, agent in enumerate(
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

        team_reward = float(
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
            "reward": team_reward,
            "done": bool(terminated or truncated),
        })

        total_reward += team_reward
        observations = next_observations

        if terminated or truncated:
            break

    return trajectory, total_reward


def train_local_model(
    global_weights,
    centre_id,
    episodes,
    seed,
):
    """
    Start from the global parameters, train on
    this participant's simulated scenarios,
    and return the updated parameters.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rng = random.Random(seed)

    model = create_model()
    model.load_state_dict(global_weights)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    trajectories = []
    rewards = []

    for _ in range(episodes):
        trajectory, reward = collect_local_episode(
            model,
            centre_id,
            rng,
        )

        trajectories.append(trajectory)
        rewards.append(reward)

    training_loss = update_model(
        model,
        optimizer,
        trajectories,
        epochs=4,
        batch_size=128,
    )

    return {
        "weights": clone_weights(model),
        "mean_reward": float(mean(rewards)),
        "training_loss": training_loss,
        "episodes": episodes,
    }


def federated_average(local_results):
    """
    FedAvg weighted by each participant's
    number of local training episodes.

    Here all three participants receive equal
    numbers of episodes, so the aggregation
    is an equal-weight average.
    """

    total_episodes = sum(
        result["episodes"]
        for result in local_results
    )

    averaged = {}

    first_weights = local_results[0]["weights"]

    for key, reference in first_weights.items():
        tensors = [
            result["weights"][key]
            for result in local_results
        ]

        for tensor in tensors:
            if tensor.shape != reference.shape:
                raise ValueError(
                    f"Incompatible parameter shape: {key}"
                )

        if torch.is_floating_point(reference):
            combined = torch.zeros_like(
                reference,
                dtype=torch.float64,
            )

            for result in local_results:
                weight = (
                    result["episodes"]
                    / total_episodes
                )

                combined += (
                    result["weights"][key].double()
                    * weight
                )

            averaged[key] = combined.to(
                reference.dtype
            )

        else:
            averaged[key] = reference.clone()

    return averaged


def evaluate_global_model(model):
    """
    Compare against the fairness baseline
    on the same six dynamic test scenarios.
    """

    scenarios = [
        scenario
        for scenario in generate_scenarios()
        if scenario["dynamic"]
    ]

    methods = {
        "baseline": None,
        "fedavg_gat_ppo": model,
    }

    summary = {}

    for method, candidate in methods.items():
        results = [
            simulate(
                scenario,
                method,
                candidate,
            )
            for scenario in scenarios
        ]

        summary[method] = {
            "scenarios": len(results),
            "mean_minimum_coverage": round(
                mean(
                    row["minimum_coverage"]
                    for row in results
                ),
                2,
            ),
            "mean_average_coverage": round(
                mean(
                    row["average_coverage"]
                    for row in results
                ),
                2,
            ),
            "mean_zero_coverage_items": round(
                mean(
                    row["zero_coverage_items"]
                    for row in results
                ),
                2,
            ),
        }

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--episodes-per-centre",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.rounds < 1:
        parser.error("--rounds must be positive")

    if args.episodes_per_centre < 1:
        parser.error(
            "--episodes-per-centre must be positive"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_num_threads(2)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_model = create_model()

    global_weights = clone_weights(global_model)

    round_history = []

    print("=" * 65)
    print("PHASE 27 — FEDERATED GAT MULTI-RESOURCE PPO")
    print("=" * 65)

    print("Participants:", ", ".join(CENTRES))
    print("Observation features: 62")
    print("Actions: 16")
    print("Resources: 5")
    print("Agents per local simulation: 3")
    print("Rounds:", args.rounds)

    print(
        "Episodes per participant per round:",
        args.episodes_per_centre,
    )

    print("Aggregation: FedAvg")
    print()

    for round_number in range(
        1,
        args.rounds + 1,
    ):
        print(
            f"ROUND {round_number}/{args.rounds}"
        )

        print("-" * 65)

        local_results = []

        for centre_index, centre_id in enumerate(
            CENTRES
        ):
            local_seed = (
                args.seed
                + round_number * 100
                + centre_index
            )

            result = train_local_model(
                global_weights=global_weights,
                centre_id=centre_id,
                episodes=args.episodes_per_centre,
                seed=local_seed,
            )

            local_results.append(result)

            print(
                f"{centre_id} "
                f"| Mean reward="
                f"{result['mean_reward']:.2f} "
                f"| Loss="
                f"{result['training_loss']:.3f}",
                flush=True,
            )

        global_weights = federated_average(
            local_results
        )

        # Verify that the global model accepts
        # the averaged parameter structure.
        global_model.load_state_dict(
            global_weights,
            strict=True,
        )

        # Verify the aggregation has not created
        # non-finite model parameters.
        for name, parameter in (
            global_model.state_dict().items()
        ):
            if torch.is_floating_point(parameter):
                if not torch.isfinite(parameter).all():
                    raise ValueError(
                        f"Non-finite global parameter: {name}"
                    )

        round_history.append({
            "round": round_number,
            "participants": {
                centre_id: {
                    "mean_reward": round(
                        result["mean_reward"], 4
                    ),
                    "training_loss": round(
                        result["training_loss"], 4
                    ),
                }
                for centre_id, result in zip(
                    CENTRES,
                    local_results,
                )
            },
            "aggregation_verified": True,
        })

        print(
            "PASS: Global model parameters aggregated"
        )
        print()

    global_model.eval()

    print("=" * 65)
    print("FEDERATED GLOBAL MODEL EVALUATION")
    print("=" * 65)

    evaluation = evaluate_global_model(
        global_model
    )

    for method, metrics in evaluation.items():
        print(
            f"{method:15} "
            f"Min={metrics['mean_minimum_coverage']:6.2f}% "
            f"Avg={metrics['mean_average_coverage']:6.2f}% "
            f"Zero={metrics['mean_zero_coverage_items']:.2f}"
        )

    torch.save(
        global_weights,
        MODEL_PATH,
    )

    # Verify the saved checkpoint can reload.
    verified_model = create_model()

    verified_model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )

    results = {
        "model": "Federated GAT-Informed Multi-Resource PPO",
        "observation_size": 62,
        "action_size": 16,
        "num_agents": 3,
        "participants": CENTRES,
        "rounds": args.rounds,
        "episodes_per_centre": args.episodes_per_centre,
        "training_seed": args.seed,
        "aggregation": "FedAvg",
        "local_data": "Separate simulated disaster scenarios",
        "privacy_guarantee": False,
        "synthetic_experiment": True,
        "round_history": round_history,
        "evaluation": evaluation,
        "approved_for_operational_allocation": False,
    }

    RESULTS_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("PASS: Saved global model reload verified")
    print()
    print("MODEL SAVED:", MODEL_PATH)
    print("RESULTS SAVED:", RESULTS_PATH)

    print()
    print(
        "Federated experiment completed. "
        "No active allocation policy was replaced."
    )


if __name__ == "__main__":
    main()
