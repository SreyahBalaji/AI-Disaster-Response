
import torch
import torch.nn as nn
from torch.distributions import Categorical


class CooperativePPO(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared actor: 7 features per relief centre.
        self.actor = nn.Sequential(
            nn.Linear(7, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4)
        )

        # Centralized critic: 7 features x 3 centres.
        self.critic = nn.Sequential(
            nn.Linear(21, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def get_actions_and_value(
        self,
        observations,
        actions=None
    ):
        # observations: [batch, 3 centres, 7 features]
        batch_size = observations.shape[0]

        actor_input = observations.reshape(-1, 7)

        logits = self.actor(actor_input)
        distribution = Categorical(logits=logits)

        if actions is None:
            actions = distribution.sample()
        else:
            actions = actions.reshape(-1)

        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        actions = actions.reshape(batch_size, 3)
        log_probs = log_probs.reshape(batch_size, 3)
        entropy = entropy.reshape(batch_size, 3)

        # Joint observation for centralized value estimation.
        critic_input = observations.reshape(batch_size, 21)

        value = self.critic(critic_input).squeeze(-1)

        return actions, log_probs, entropy, value


if __name__ == "__main__":
    torch.manual_seed(42)

    model = CooperativePPO()

    observations = torch.tensor(
        [[
            [0.48, 0.30, 0.216, 0.0, 0.0, 0.0, 0.5],
            [0.48, 0.30, 0.216, 0.0, 0.0, 0.0, 0.4],
            [0.48, 0.30, 0.216, 0.0, 0.0, 0.0, 0.4]
        ]],
        dtype=torch.float32
    )

    actions, log_probs, entropy, value = (
        model.get_actions_and_value(observations)
    )

    print("COOPERATIVE PPO NETWORK TEST")
    print("--------------------------------")
    print("Actions:", actions)
    print("Log probabilities:", log_probs)
    print("Entropy:", entropy)
    print("Centralized value:", value)

    assert actions.shape == (1, 3)
    assert log_probs.shape == (1, 3)
    assert entropy.shape == (1, 3)
    assert value.shape == (1,)

    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(value).all()

    print("\nSUCCESS! Cooperative PPO network validated.")
