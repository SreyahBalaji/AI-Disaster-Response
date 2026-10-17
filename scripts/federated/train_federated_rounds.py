
from pathlib import Path

import torch
from local_model import LocalReliefModel
from federated_server import federated_average

torch.manual_seed(42)

centres = ["RC01", "RC02", "RC03"]
rounds = 5
local_epochs = 20

output = Path("data/processed/federated")
output.mkdir(parents=True, exist_ok=True)

# Fixed synthetic datasets for a reproducible experiment.
datasets = {}

for index, centre_id in enumerate(centres):
    x = torch.rand(40, 5)
    x[:, 0] = torch.clamp(x[:, 0] + index * 0.1, 0, 1)

    y = (
        0.5 * x[:, 0:1]
        + 0.3 * x[:, 1:2]
        + 0.2 * x[:, 2:3]
    )

    datasets[centre_id] = (x, y)

global_model = LocalReliefModel()
loss_function = torch.nn.MSELoss()

print("MULTI-ROUND FEDERATED TRAINING")
print("--------------------------------")

for round_number in range(1, rounds + 1):
    local_weights = []

    for centre_id in centres:
        local_model = LocalReliefModel()
        local_model.load_state_dict(global_model.state_dict())

        optimizer = torch.optim.Adam(
            local_model.parameters(), lr=0.01
        )

        x, y = datasets[centre_id]

        for _ in range(local_epochs):
            optimizer.zero_grad()
            predictions = local_model(x)
            loss = loss_function(predictions, y)
            loss.backward()
            optimizer.step()

        local_weights.append(local_model.state_dict())

    global_weights = federated_average(local_weights)
    global_model.load_state_dict(global_weights)

    with torch.no_grad():
        losses = [
            loss_function(global_model(x), y).item()
            for x, y in datasets.values()
        ]

    average_loss = sum(losses) / len(losses)

    print(
        f"Round {round_number}/{rounds} | "
        f"Global training MSE: {average_loss:.6f}"
    )

torch.save(
    global_model.state_dict(),
    output / "aluva_multiround_global_model.pth"
)

print("\nSUCCESS! Multi-round federated training completed.")
