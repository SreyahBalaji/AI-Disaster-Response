
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MultiResourceCooperativePPO(nn.Module):
    def __init__(
        self,
        observation_size=38,
        action_size=16,
        num_agents=3
    ):
        super().__init__()

        self.observation_size = observation_size
        self.action_size = action_size
        self.num_agents = num_agents

        self.actor = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(observation_size * num_agents, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def get_actions_and_value(
        self,
        observations,
        actions=None,
        action_masks=None
    ):
        if observations.ndim != 3:
            raise ValueError(
                "Expected observations: [batch, agents, features]"
            )

        batch_size, agents, features = observations.shape

        if (
            agents != self.num_agents
            or features != self.observation_size
        ):
            raise ValueError("Incorrect observation dimensions")

        actor_input = observations.reshape(-1, features)
        logits = self.actor(actor_input)

        if action_masks is not None:
            expected_shape = (
                batch_size,
                self.num_agents,
                self.action_size
            )

            if tuple(action_masks.shape) != expected_shape:
                raise ValueError("Incorrect action mask dimensions")

            mask = action_masks.reshape(
                -1, self.action_size
            ).to(device=logits.device, dtype=torch.bool)

            if not mask.any(dim=-1).all():
                raise ValueError(
                    "Every agent must have at least one valid action"
                )

            logits = logits.masked_fill(
                ~mask,
                torch.finfo(logits.dtype).min
            )

        distribution = Categorical(logits=logits)

        if actions is None:
            actions = distribution.sample()
        else:
            actions = actions.reshape(-1).long()

        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        actions = actions.reshape(batch_size, agents)
        log_probs = log_probs.reshape(batch_size, agents)
        entropy = entropy.reshape(batch_size, agents)

        values = self.critic(
            observations.reshape(batch_size, -1)
        ).squeeze(-1)

        return actions, log_probs, entropy, values


if __name__ == "__main__":
    model = MultiResourceCooperativePPO()

    observations = torch.randn(2, 3, 38)

    masks = torch.zeros(
        2, 3, 16,
        dtype=torch.bool
    )

    # Only action 2 and WAIT are valid.
    masks[:, :, 2] = True
    masks[:, :, 15] = True

    actions, log_probs, entropy, values = (
        model.get_actions_and_value(
            observations,
            action_masks=masks
        )
    )

    assert actions.shape == (2, 3)
    assert values.shape == (2,)
    assert torch.all(
        (actions == 2) | (actions == 15)
    )

    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()

    print("SUCCESS: Masked PPO network validated.")
