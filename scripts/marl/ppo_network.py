import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOActorCritic(nn.Module):
    def __init__(self, observation_dim=7, action_dim=4):
        super().__init__()

        # Actor: selects an allocation action.
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

        # Critic: estimates future reward.
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, observation):
        action_logits = self.actor(observation)
        state_value = self.critic(observation)

        return action_logits, state_value

    def get_action_and_value(self, observation, action=None):
        logits, value = self.forward(observation)

        distribution = Categorical(logits=logits)

        if action is None:
            action = distribution.sample()

        log_probability = distribution.log_prob(action)
        entropy = distribution.entropy()

        return (
            action,
            log_probability,
            entropy,
            value.squeeze(-1)
        )


if __name__ == "__main__":
    torch.manual_seed(42)

    model = PPOActorCritic()

    observation = torch.tensor(
        [[0.48, 0.30, 0.216, 0.0, 0.0, 0.0, 0.50]],
        dtype=torch.float32
    )

    action, log_prob, entropy, value = (
        model.get_action_and_value(observation)
    )

    print("PPO ACTOR-CRITIC NETWORK TEST")
    print("--------------------------------")
    print("Selected action:", action.item())
    print("Log probability:", log_prob.item())
    print("Entropy:", entropy.item())
    print("Estimated value:", value.item())

    assert 0 <= action.item() < 4
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(value).all()

    print("\nSUCCESS! PPO networks validated.")
