
from pathlib import Path
import sys

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent
MARL_DIR = BASE_DIR / "scripts" / "marl"

if str(MARL_DIR) not in sys.path:
    sys.path.insert(0, str(MARL_DIR))

from multi_resource_cooperative_env import MultiResourceCooperativeEnv
from multi_resource_ppo_network import MultiResourceCooperativePPO


router = APIRouter(
    prefix="/api/multi-resource",
    tags=["Multi-Resource Allocation"]
)

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "marl"
    / "aluva_multi_resource_masked_ppo.pth"
)

model = None
model_error = None

try:
    model = MultiResourceCooperativePPO()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()
    print("Multi-resource PPO loaded successfully.")

except Exception as error:
    model = None
    model_error = str(error)
    print("Multi-resource PPO loading failed:", model_error)


def select_ppo_actions(env, observations):
    state = torch.tensor(
        np.stack([
            observations[centre_id]
            for centre_id in env.centre_ids
        ]),
        dtype=torch.float32
    ).unsqueeze(0)

    masks = env.get_action_masks()

    mask_tensor = torch.tensor(
        np.stack([
            masks[centre_id]
            for centre_id in env.centre_ids
        ]),
        dtype=torch.bool
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model.actor(state)

        logits = logits.masked_fill(
            ~mask_tensor,
            torch.finfo(logits.dtype).min
        )

        actions = torch.argmax(logits, dim=-1)

    return {
        centre_id: int(actions[0, i].item())
        for i, centre_id in enumerate(env.centre_ids)
    }


def select_baseline_actions(env):
    actions = {}
    masks = env.get_action_masks()

    for centre_id in env.centre_ids:
        candidates = []

        for action, (ward, resource) in enumerate(env.action_map):
            if not masks[centre_id][action]:
                continue

            delivered = env.env.delivered[ward][resource]
            remaining = env.env.demand[ward][resource]
            total = delivered + remaining

            satisfaction = (
                delivered / total if total > 0 else 1.0
            )

            priority = (
                (1.0 - satisfaction)
                * env.vulnerability[ward]
            )

            candidates.append((priority, action))

        if candidates:
            candidates.sort(reverse=True)
            actions[centre_id] = candidates[0][1]
        else:
            actions[centre_id] = env.wait_action

    return actions


def run_simulation(algorithm, dynamic):
    env = MultiResourceCooperativeEnv()
    env.dynamic_demand_enabled = dynamic
    observations = env.reset()

    successful_actions = 0

    while True:
        if algorithm == "ppo":
            actions = select_ppo_actions(env, observations)
        else:
            actions = select_baseline_actions(env)

        observations, _, terminated, truncated, info = env.step(
            actions
        )

        successful_actions += info["successful_actions"]

        if terminated or truncated:
            break

    resources = []

    for resource in env.resources:
        demand = sum(
            env.env.demand[ward][resource]
            + env.env.delivered[ward][resource]
            for ward in env.wards
        )

        delivered = sum(
            env.env.delivered[ward][resource]
            for ward in env.wards
        )

        resources.append({
            "resource": resource,
            "demand": demand,
            "delivered": delivered,
            "unmet": demand - delivered,
            "satisfaction_percent": (
                round(100 * delivered / demand, 2)
                if demand > 0 else None
            )
        })

    wards = []

    for ward in env.wards:
        ward_resources = []

        for resource in env.resources:
            delivered = env.env.delivered[ward][resource]
            remaining = env.env.demand[ward][resource]
            demand = delivered + remaining

            ward_resources.append({
                "resource": resource,
                "demand": demand,
                "delivered": delivered,
                "unmet": remaining,
                "satisfaction_percent": (
                    round(100 * delivered / demand, 2)
                    if demand > 0 else None
                )
            })

        wards.append({
            "ward_no": ward,
            "vulnerability_weight": env.vulnerability[ward],
            "resources": ward_resources
        })

    return {
        "algorithm": (
            "Masked Cooperative PPO"
            if algorithm == "ppo"
            else "Fairness-Aware Heuristic Baseline"
        ),
        "scenario": "dynamic" if dynamic else "static",
        "simulation_steps": env.steps,
        "successful_actions": successful_actions,
        "fairness_gap": round(info["fairness_gap"], 4),
        "vulnerability_weighted_unmet_ratio": round(
            info["vulnerability_weighted_unmet_ratio"], 4
        ),
        "resources": resources,
        "wards": wards,
        "events": [
            event for event in env.events
            if event["type"] == "new_disaster_report"
        ],
        "note": (
            "Research simulation only. No real resources "
            "are dispatched. Vulnerability weights are illustrative."
        )
    }


@router.get("/status")
def multi_resource_status():
    return {
        "model_loaded": model is not None,
        "model_file": MODEL_PATH.name,
        "model_error": model_error,
        "resources": [
            "water_litres",
            "food_meals",
            "rescue_cases",
            "shelter_households",
            "medical_cases"
        ],
        "algorithms": [
            "ppo",
            "baseline"
        ]
    }


@router.get("/compare")
def compare_allocations(dynamic: bool = False):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Multi-resource PPO unavailable: {model_error}"
        )

    try:
        ppo = run_simulation("ppo", dynamic)
        baseline = run_simulation("baseline", dynamic)

        return {
            "scenario": "dynamic" if dynamic else "static",
            "ppo": ppo,
            "baseline": baseline
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {error}"
        )
