
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
    "Single-round FedAvg PPO": Path(
        "data/processed/marl/federated/fedavg_cooperative_ppo.pth"
    ),
    "Multi-round FedAvg PPO": Path(
        "data/processed/marl/federated/multiround_federated_ppo.pth"
    )
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
        ratios = []

        for ward in env.wards:
            delivered = env.env.delivered[ward]["water_litres"]
            remaining = env.env.demand[ward]["water_litres"]

            demand = delivered + remaining

            deliveries[ward] = delivered

            ratios.append(
                delivered / demand if demand > 0 else 1.0
            )

        total_delivered = sum(deliveries.values())
        total_demand = 1496 if dynamic else 996

        results.append({
            "delivered": total_delivered,
            "satisfaction": total_delivered / total_demand,
            "fairness_gap": max(ratios) - min(ratios),
            "ward9": deliveries[9]
        })

    return {
        key: float(np.mean([r[key] for r in results]))
        for key in results[0]
    }


print("MULTI-ROUND FEDERATED PPO EVALUATION")
print("------------------------------------")

for name, path in MODELS.items():
    if not path.exists():
        raise FileNotFoundError(path)

    model = CooperativePPO()
    model.load_state_dict(
        torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    )
    model.eval()

    print(f"\n{name}")

    for dynamic in [False, True]:
        result = evaluate(model, dynamic)

        scenario = "Dynamic" if dynamic else "Static"

        print(
            f"{scenario}: "
            f"Delivered={result['delivered']:.2f} L | "
            f"Satisfaction={result['satisfaction']:.2%} | "
            f"Fairness gap={result['fairness_gap']:.4f} | "
            f"Ward 9={result['ward9']:.2f} L"
        )

print("\nSUCCESS! Multi-round evaluation completed.")
