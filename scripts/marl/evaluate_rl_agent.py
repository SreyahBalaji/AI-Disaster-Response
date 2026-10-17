
import torch

from resource_allocation_env import ResourceAllocationEnv
from agent_observation import get_agent_observation
from agent_action import AgentActionSpace
from policy_network import PolicyNetwork
from fairness_metrics import calculate_fairness
from fair_allocation_policy import fair_allocate


def evaluate_rl():
    env = ResourceAllocationEnv()
    actions = AgentActionSpace(env)

    model = PolicyNetwork()

    model.load_state_dict(
        torch.load(
            "data/processed/marl/aluva_rl_policy.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    for _ in range(100):
        observation = get_agent_observation(env, "RC01")

        state = torch.tensor(
            observation,
            dtype=torch.float32
        )

        with torch.no_grad():
            logits = model(state)
            action = int(torch.argmax(logits).item())

        actions.step("RC01", action)

    delivered = sum(
        ward["water_litres"]
        for ward in env.delivered.values()
    )

    fairness = calculate_fairness(env, "water_litres")

    return delivered, fairness


def evaluate_rule_based():
    env = ResourceAllocationEnv()

    # Match the RL agent's available water.
    env.centres[1]["resources"]["water_litres"] = 0
    env.centres[2]["resources"]["water_litres"] = 0

    fair_allocate(env, "water_litres", batch_size=10)

    delivered = sum(
        ward["water_litres"]
        for ward in env.delivered.values()
    )

    fairness = calculate_fairness(env, "water_litres")

    return delivered, fairness


if __name__ == "__main__":
    rl_delivered, rl_fairness = evaluate_rl()
    baseline_delivered, baseline_fairness = evaluate_rule_based()

    print("RL VS RULE-BASED EVALUATION")
    print("--------------------------------")

    print("\nTRAINED RL AGENT")
    print("Water delivered:", rl_delivered)
    print(
        "Fairness gap:",
        round(rl_fairness["fairness_gap"], 4)
    )

    print("\nRULE-BASED POLICY")
    print("Water delivered:", baseline_delivered)
    print(
        "Fairness gap:",
        round(baseline_fairness["fairness_gap"], 4)
    )

    print("\nSUCCESS! Evaluation completed.")
