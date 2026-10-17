
import json
from pathlib import Path

results = {
    "experiment": "Single-agent water allocation",
    "scenario": "Simulated Aluva disaster demand",
    "available_water_litres": 500,
    "total_demand_litres": 996,
    "models": {
        "original_masked_rl": {
            "delivered_litres": 500,
            "fairness_gap": 1.0
        },
        "priority_based_rl": {
            "delivered_litres": 500,
            "fairness_gap": 1.0
        }
    },
    "conclusion": (
        "Priority-based RL did not improve fairness "
        "over the original masked RL policy."
    ),
    "limitations": [
        "Single relief-centre agent",
        "Water allocation only",
        "One deterministic evaluation",
        "No transport constraints",
        "No vulnerability-aware priorities"
    ]
}

output = Path("data/processed/marl")
output.mkdir(parents=True, exist_ok=True)

file_path = output / "aluva_rl_evaluation.json"

with open(file_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("RL EVALUATION RESULTS")
print("-----------------------------")
print("Results saved:", file_path)
print("\nSUCCESS! Evaluation results recorded.")
