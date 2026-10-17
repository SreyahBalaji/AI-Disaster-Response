
import torch

from resource_allocation_env import ResourceAllocationEnv
from agent_observation import get_agent_observation
from agent_action import AgentActionSpace
from action_masking import (
    get_valid_action_mask,
    mask_action_logits
)
from policy_network import PolicyNetwork


env = ResourceAllocationEnv()
actions = AgentActionSpace(env)

model = PolicyNetwork()
model.load_state_dict(
    torch.load(
        "data/processed/marl/aluva_improved_rl_policy.pth",
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

        observation = get_agent_observation(env, "RC01")
        state = torch.tensor(observation, dtype=torch.float32)

        logits = model(state)
        masked_logits = mask_action_logits(logits, mask)

        action = int(torch.argmax(masked_logits).item())
        actions.step("RC01", action)


print("WARD-LEVEL ALLOCATION DIAGNOSIS")
print("--------------------------------")

for ward_no in sorted(env.demand):
    delivered = env.delivered[ward_no]["water_litres"]
    remaining = env.demand[ward_no]["water_litres"]

    requested = delivered + remaining

    if requested > 0:
        fulfilment = delivered / requested

        print(
            f"Ward {ward_no}: "
            f"Requested={requested} L | "
            f"Delivered={delivered} L | "
            f"Fulfilment={fulfilment:.2%}"
        )

print("\nSUCCESS! Ward allocations inspected.")
