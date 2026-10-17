
import json
from pathlib import Path

results = {
    "experiment": "Federated Learning - Aluva Prototype",
    "participating_centres": ["RC01", "RC02", "RC03"],
    "aggregation_method": "FedAvg",
    "training_rounds": 5,
    "dataset": "Synthetic resource-related features",
    "test_mse": {
        "untrained": 1.385763,
        "federated": 0.000723,
        "centralized": 0.000801
    },
    "limitations": [
        "Synthetic training and evaluation data",
        "Centralized comparison not fully controlled",
        "No real-world disaster prediction validation",
        "No differential privacy or secure aggregation",
        "Not yet integrated with MARL"
    ]
}

output = Path("data/processed/federated")
output.mkdir(parents=True, exist_ok=True)

file_path = output / "federated_evaluation.json"

with open(file_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("FEDERATED EVALUATION RESULTS")
print("----------------------------")
print("Saved:", file_path)
print("\nSUCCESS! Federated evaluation results recorded.")
