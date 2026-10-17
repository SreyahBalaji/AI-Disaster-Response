
import torch

from resource_allocation_env import ResourceAllocationEnv
from agent_observation import get_agent_observation
from agent_action import AgentActionSpace
from action_masking import (
    get_valid_action_mask,
    mask_action_logits
)
from policy_network import PolicyNetwork
from fairness_metrics import calculate_fairness


def evaluate_policy(model_path):
    env = ResourceAllocationEnv()
    actions = AgentActionSpace(env)

    model = PolicyNetwork()
    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )
    )
    model.eval()

    with torch.no_grad():
        for _ in range(50):
            mask = get_valid_action_mask(env, "RC01")

            if mask[-1].item():
                break

            observation = get_agent_observation(
                env, "RC01"
            )

            state = torch.tensor(
                observation,
                dtype=torch.float32
            )

            logits = model(state)
            masked_logits = mask_action_logits(
                logits, mask
            )

            action = int(
                torch.argmax(masked_logits).item()
            )

            actions.step("RC01", action)

    delivered = sum(
        ward["water_litres"]
        for ward in env.delivered.values()
    )

    fairness = calculate_fairness(
        env, "water_litres"
    )

    return delivered, fairness["fairness_gap"]


if __name__ == "__main__":
    old_path = (
        "data/processed/marl/aluva_masked_rl_policy.pth"
    )
    improved_path = (
    "data/processed/marl/aluva_priority_rl_policy.pth"
)

    old_delivered, old_gap = evaluate_policy(old_path)

    new_delivered, new_gap = evaluate_policy(improved_path)

    print("RL POLICY COMPARISON")
    print("--------------------------------")

    print("\nORIGINAL MASKED RL")
    print("Water delivered:", old_delivered)
    print("Fairness gap:", round(old_gap, 4))

    print("\nPRIORITY-BASED RL")
    print("Water delivered:", new_delivered)
    print("Fairness gap:", round(new_gap, 4))

    print("\nFAIRNESS GAP CHANGE")
    print(round(old_gap - new_gap, 4))

    assert old_delivered <= 500
    assert new_delivered <= 500

    print("\nSUCCESS! RL policy comparison completed.")
