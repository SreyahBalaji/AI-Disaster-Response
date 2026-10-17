
import torch
import torch.nn as nn


class LocalReliefModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":

    model = LocalReliefModel()

    # Five sample resource-related features.
    sample_input = torch.tensor(
        [[0.8, 0.6, 0.4, 0.7, 0.5]],
        dtype=torch.float32
    )

    output = model(sample_input)

    print("LOCAL RELIEF MODEL TEST")
    print("----------------------------")
    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)

    assert output.shape == (1, 1)

    print("\nSUCCESS! Local model initialized.")
