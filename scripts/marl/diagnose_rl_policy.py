
from collections import Counter

import torch

from resource_allocation_env import ResourceAllocationEnv
from agent_observation import get_agent_observation
from agent_action import AgentActionSpace
from policy_network import PolicyNetwork


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

action_counts = Counter()
total_delivered = 0

for _ in range(100):
    observation = get_agent_observation(env, "RC01")
    state = torch.tensor(observation, dtype=torch.float32)

    with torch.no_grad():
        logits = model(state)
        action = int(torch.argmax(logits).item())

    _, reward, info = actions.step("RC01", action)

    action_counts[action] += 1
    total_delivered += info["allocated"]

print("RL POLICY DIAGNOSIS")
print("-----------------------------")

for action, count in action_counts.most_common():
    if action == 26:
        description = "WAIT"
    else:
        description = f"Ward {actions.ward_numbers[action]}"

    print(f"{description}: {count} times")

print("\nTotal water delivered:", total_delivered)
print(
    "RC01 remaining water:",
    env.centres[0]["resources"]["water_litres"]
)

print("\nSUCCESS! Policy diagnosis completed.")
