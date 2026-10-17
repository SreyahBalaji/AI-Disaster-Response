
from pathlib import Path
import contextlib
import io

import numpy as np
import torch

from cooperative_allocation_env import CooperativeAllocationEnv
from cooperative_ppo_network import CooperativePPO


MODELS = {
    "Original Dynamic PPO": Path(
        "data/processed/marl/aluva_dynamic_cooperative_ppo.pth"
    ),
    "Federated PPO (FedAvg)": Path(
        "data/processed/marl/federated/fedavg_cooperative_ppo.pth"
    ),
}

EPISODES = 100


def evaluate(model, dynamic):
    results = []

    for episode in range(EPISODES):
        torch.manual_seed(1000 + episode)

        env = CooperativeAllocationEnv()
        env.dynamic_demand_enabled = dynamic

        observations = env.reset()

        while True:
            state = torch.tensor(
                np.stack([
                    observations[centre_id]
                    for centre_id in env.centre_ids
                ]),
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                actions, _, _, _ = (
                    model.get_actions_and_value(state)
                )

            action_dict = {
                centre_id: int(actions[0, i].item())
                for i, centre_id in enumerate(env.centre_ids)
            }

            with contextlib.redirect_stdout(io.StringIO()):
                observations, _, terminated, truncated, _ = (
                    env.step(action_dict)
                )

            if terminated or truncated:
                break

        deliveries = {}
        fulfilment_ratios = []

        for ward in env.wards:
            delivered = env.env.delivered[ward]["water_litres"]
            remaining = env.env.demand[ward]["water_litres"]

            original_demand = delivered + remaining

            deliveries[ward] = delivered

            fulfilment_ratios.append(
                delivered / original_demand
                if original_demand > 0 else 1.0
            )

        total_delivered = sum(deliveries.values())
        total_demand = 1496 if dynamic else 996

        results.append({
            "delivered": total_delivered,
            "satisfaction": total_delivered / total_demand,
            "fairness_gap": (
                max(fulfilment_ratios)
                - min(fulfilment_ratios)
            ),
            "ward3": deliveries[3],
            "ward5": deliveries[5],
            "ward9": deliveries[9]
        })

    return {
        key: float(np.mean([result[key] for result in results]))
        for key in results[0]
    }


print("FEDERATED PPO EVALUATION")
print("------------------------")

for model_name, model_path in MODELS.items():
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = CooperativePPO()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    print(f"\n{model_name}")

    for dynamic in [False, True]:
        metrics = evaluate(model, dynamic)

        scenario = "Dynamic" if dynamic else "Static"

        print(f"\n{scenario} scenario:")

        print(
            f"Water delivered: "
            f"{metrics['delivered']:.2f} L"
        )

        print(
            f"Demand satisfaction: "
            f"{metrics['satisfaction']:.2%}"
        )

        print(
            f"Fairness gap: "
            f"{metrics['fairness_gap']:.4f}"
        )

        print(
            f"Ward 3: {metrics['ward3']:.2f} L | "
            f"Ward 5: {metrics['ward5']:.2f} L | "
            f"Ward 9: {metrics['ward9']:.2f} L"
        )

print("\nSUCCESS! Federated PPO evaluation completed.")
