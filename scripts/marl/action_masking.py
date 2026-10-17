
import torch

from resource_allocation_env import ResourceAllocationEnv
from agent_observation import get_agent_observation
from policy_network import PolicyNetwork


def get_valid_action_mask(env, centre_id):
    centre = next(
        (c for c in env.centres if c["centre_id"] == centre_id),
        None
    )

    if centre is None:
        raise ValueError("Invalid centre ID")

    resource = "water_litres"
    has_stock = centre["resources"][resource] > 0

    ward_numbers = sorted(env.demand)

    valid_actions = [
        has_stock and env.demand[ward][resource] > 0
        for ward in ward_numbers
    ]

    # WAIT is valid only when no allocation is possible.
    valid_actions.append(not any(valid_actions))

    return torch.tensor(valid_actions, dtype=torch.bool)


def mask_action_logits(logits, valid_mask):
    if logits.shape != valid_mask.shape:
        raise ValueError("Logits and mask shapes must match")

    if not valid_mask.any():
        raise ValueError("No valid actions available")

    return logits.masked_fill(~valid_mask, -1e9)


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    model = PolicyNetwork()
    model.load_state_dict(
        torch.load(
            "data/processed/marl/aluva_rl_policy.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    observation = get_agent_observation(env, "RC01")
    state = torch.tensor(observation, dtype=torch.float32)

    with torch.no_grad():
        logits = model(state)

    mask = get_valid_action_mask(env, "RC01")
    masked_logits = mask_action_logits(logits, mask)

    action = int(torch.argmax(masked_logits).item())

    ward_numbers = sorted(env.demand)

    print("ACTION MASKING TEST")
    print("-----------------------------")
    print("Total actions:", len(mask))
    print("Valid actions:", int(mask.sum().item()))
    print("Selected action:", action)

    if action == len(ward_numbers):
        print("Selected: WAIT")
    else:
        selected_ward = ward_numbers[action]
        print("Selected ward:", selected_ward)
        print(
            "Remaining water demand:",
            env.demand[selected_ward]["water_litres"]
        )

    assert mask.shape == (27,)
    assert mask[action].item()
    assert mask.sum().item() > 0

    print("\nSUCCESS! Action masking works.")
