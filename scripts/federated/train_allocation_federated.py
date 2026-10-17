
import json
from pathlib import Path

import torch

from local_model import LocalReliefModel
from federated_server import federated_average

torch.manual_seed(42)

data_dir = Path("data/processed/federated")

with open(
    data_dir / "aluva_allocation_training_data.json",
    encoding="utf-8"
) as file:
    records = json.load(file)

centres = ["RC01", "RC02", "RC03"]
datasets = {}

for centre_id in centres:
    centre_records = [
        record for record in records
        if record["centre_id"] == centre_id
    ]

    assert len(centre_records) > 0

    x = torch.tensor(
        [record["features"] for record in centre_records],
        dtype=torch.float32
    )

    y = torch.tensor(
        [[record["target"]] for record in centre_records],
        dtype=torch.float32
    )

    datasets[centre_id] = (x, y)

global_model = LocalReliefModel()
loss_function = torch.nn.MSELoss()

print("ALLOCATION FEDERATED TRAINING")
print("--------------------------------")

for round_number in range(1, 6):
    local_weights = []

    for centre_id in centres:
        local_model = LocalReliefModel()
        local_model.load_state_dict(
            global_model.state_dict()
        )

        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=0.01
        )

        x, y = datasets[centre_id]

        for _ in range(20):
            optimizer.zero_grad()
            loss = loss_function(local_model(x), y)
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

    print(
        f"Round {round_number}/5 | "
        f"Training MSE: {sum(losses) / len(losses):.6f}"
    )

model_path = data_dir / "aluva_allocation_federated_model.pth"
torch.save(global_model.state_dict(), model_path)

print("\nModel saved:", model_path)
print("SUCCESS! Allocation federated training completed.")
