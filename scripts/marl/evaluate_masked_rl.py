
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


def evaluate(model):
    env = ResourceAllocationEnv()
    actions = AgentActionSpace(env)

    model.eval()

    with torch.no_grad():
        for _ in range(50):
            mask = get_valid_action_mask(env, "RC01")

            # Stop if no allocation action remains.
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

    return delivered, fairness


if __name__ == "__main__":
    # Same initial weights as the training run.
    torch.manual_seed(42)
    untrained_model = PolicyNetwork()

    trained_model = PolicyNetwork()
    trained_model.load_state_dict(
        torch.load(
            "data/processed/marl/aluva_masked_rl_policy.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    untrained_delivery, untrained_fairness = evaluate(
        untrained_model
    )

    trained_delivery, trained_fairness = evaluate(
        trained_model
    )

    print("MASKED RL EVALUATION")
    print("--------------------------------")

    print("\nUNTRAINED MASKED POLICY")
    print("Water delivered:", untrained_delivery)
    print(
        "Fairness gap:",
        round(untrained_fairness["fairness_gap"], 4)
    )

    print("\nTRAINED MASKED POLICY")
    print("Water delivered:", trained_delivery)
    print(
        "Fairness gap:",
        round(trained_fairness["fairness_gap"], 4)
    )

    assert untrained_delivery <= 500
    assert trained_delivery <= 500

    print("\nSUCCESS! Masked policy evaluation completed.")
