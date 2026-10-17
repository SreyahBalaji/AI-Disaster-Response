
import torch
from torch import nn

from resource_allocation_env import ResourceAllocationEnv
from agent_observation import get_agent_observation


class PolicyNetwork(nn.Module):
    def __init__(self, observation_size=53, action_size=27):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, observation):
        return self.network(observation)


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    observation = get_agent_observation(
        env,
        centre_id="RC01"
    )

    model = PolicyNetwork()

    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32
    )

    action_scores = model(observation_tensor)

    probabilities = torch.softmax(
        action_scores,
        dim=-1
    )

    print("RL POLICY NETWORK TEST")
    print("-----------------------------")
    print("Input features:", len(observation))
    print("Output actions:", len(action_scores))
    print(
        "Probability sum:",
        round(probabilities.sum().item(), 4)
    )

    assert action_scores.shape == (27,)
    assert torch.isfinite(action_scores).all()
    assert torch.all(probabilities >= 0)
    assert torch.allclose(
        probabilities.sum(),
        torch.tensor(1.0),
        atol=1e-6
    )

    print("\nSUCCESS! Policy network works.")
