
import torch
from pathlib import Path
from local_model import LocalReliefModel

torch.manual_seed(42)

output_dir = Path("data/processed/federated")
output_dir.mkdir(parents=True, exist_ok=True)

centres = ["RC01", "RC02", "RC03"]

# Shared starting weights for federated training.
initial_model = LocalReliefModel()
initial_weights = initial_model.state_dict()

for index, centre_id in enumerate(centres):
    model = LocalReliefModel()
    model.load_state_dict(initial_weights)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01
    )
    loss_function = torch.nn.MSELoss()

    # Synthetic data, different for each centre.
    x = torch.rand(40, 5)
    x[:, 0] = torch.clamp(
        x[:, 0] + index * 0.1, 0, 1
    )

    # Illustrative target, not actual disaster outcomes.
    y = (
        0.5 * x[:, 0:1]
        + 0.3 * x[:, 1:2]
        + 0.2 * x[:, 2:3]
    )

    for epoch in range(100):
        predictions = model(x)
        loss = loss_function(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model_path = output_dir / f"{centre_id}_local_model.pth"
    torch.save(model.state_dict(), model_path)

    print(
        f"{centre_id} | Final loss: {loss.item():.6f} "
        f"| Saved: {model_path}"
    )

print("\nSUCCESS! Three local models trained.")
