
import contextlib
import io

import numpy as np
import torch

from fastapi import APIRouter, HTTPException

from cooperative_allocation_env import CooperativeAllocationEnv


router = APIRouter(prefix="/api", tags=["AI Allocation"])


def register_allocation_routes(model):

    def get_model():
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="AI model is not available"
            )
        return model

    def predict_actions(observations, centre_ids):
        state = torch.tensor(
            np.stack([
                observations[centre_id]
                for centre_id in centre_ids
            ]),
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            logits = get_model().actor(state)
            actions = torch.argmax(logits, dim=-1)

        return {
            centre_id: int(actions[0, index].item())
            for index, centre_id in enumerate(centre_ids)
        }

    @router.get("/allocation/predict")
    def predict_allocation():
        env = CooperativeAllocationEnv()
        observations = env.reset()

        actions = predict_actions(
            observations,
            env.centre_ids
        )

        recommendations = []

        for centre_id, action in actions.items():
            if action < len(env.wards):
                recommendations.append({
                    "centre_id": centre_id,
                    "ward_no": env.wards[action],
                    "resource": "water_litres",
                    "quantity": 10,
                    "action": "allocate"
                })
            else:
                recommendations.append({
                    "centre_id": centre_id,
                    "ward_no": None,
                    "resource": "water_litres",
                    "quantity": 0,
                    "action": "wait"
                })

        return {
            "model": "Federated Cooperative PPO",
            "scenario": "initial_static",
            "recommendations": recommendations,
            "note": "Prediction only; no real dispatch."
        }

    @router.get("/allocation/simulate")
    def simulate_allocation(dynamic: bool = False):
        get_model()

        env = CooperativeAllocationEnv()
        env.dynamic_demand_enabled = dynamic

        observations = env.reset()

        while True:
            actions = predict_actions(
                observations,
                env.centre_ids
            )

            with contextlib.redirect_stdout(io.StringIO()):
                observations, _, terminated, truncated, _ = (
                    env.step(actions)
                )

            if terminated or truncated:
                break

        ward_results = []
        satisfaction_ratios = []

        for ward in env.wards:
            delivered = float(
                env.env.delivered[ward]["water_litres"]
            )

            remaining = float(
                env.env.demand[ward]["water_litres"]
            )

            demand = delivered + remaining

            satisfaction = (
                delivered / demand if demand > 0 else 1.0
            )

            satisfaction_ratios.append(satisfaction)

            ward_results.append({
                "ward_no": ward,
                "demand_litres": demand,
                "delivered_litres": delivered,
                "unmet_litres": remaining,
                "satisfaction_percent": round(
                    satisfaction * 100, 2
                )
            })

        total_delivered = sum(
            ward["delivered_litres"]
            for ward in ward_results
        )

        total_demand = sum(
            ward["demand_litres"]
            for ward in ward_results
        )

        fairness_gap = (
            max(satisfaction_ratios)
            - min(satisfaction_ratios)
        )

        return {
            "model": "Federated Cooperative PPO",
            "scenario": (
                "dynamic_ward9_surge"
                if dynamic else "static"
            ),
            "simulation_steps": env.steps,
            "total_demand_litres": total_demand,
            "total_delivered_litres": total_delivered,
            "demand_satisfaction_percent": round(
                100 * total_delivered / total_demand, 2
            ),
            "fairness_gap": round(fairness_gap, 4),
            "ward_allocations": ward_results,
            "note": (
                "Simulated allocations only. "
                "No real-world resources dispatched."
            )
        }

    return router
