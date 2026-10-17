
import json
from pathlib import Path

OUTPUT = Path(
    "data/processed/aluva/aluva_relief_centres.json"
)

# Simulated relief centres (not real-world locations).
centres = [
    {
        "centre_id": "RC01",
        "name": "Simulated Relief Centre 1",
        "ward_no": 3,
        "resources": {
            "water_litres": 500,
            "food_meals": 350,
            "rescue_cases": 40,
            "shelter_households": 20,
            "medical_cases": 20
        }
    },
    {
        "centre_id": "RC02",
        "name": "Simulated Relief Centre 2",
        "ward_no": 13,
        "resources": {
            "water_litres": 400,
            "food_meals": 300,
            "rescue_cases": 40,
            "shelter_households": 20,
            "medical_cases": 20
        }
    },
    {
        "centre_id": "RC03",
        "name": "Simulated Relief Centre 3",
        "ward_no": 23,
        "resources": {
            "water_litres": 400,
            "food_meals": 300,
            "rescue_cases": 40,
            "shelter_households": 20,
            "medical_cases": 20
        }
    }
]

assert len(centres) == 3
assert len({c["centre_id"] for c in centres}) == 3
assert all(1 <= c["ward_no"] <= 26 for c in centres)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(centres, file, indent=2)

print("RELIEF CENTRE CONFIGURATION")
print("-----------------------------")

for centre in centres:
    print(
        f"{centre['centre_id']} | "
        f"Ward {centre['ward_no']} | "
        f"{centre['name']}"
    )

print("\nTOTAL INITIAL RESOURCES")

for resource in centres[0]["resources"]:
    total = sum(
        centre["resources"][resource]
        for centre in centres
    )
    print(f"{resource}: {total}")

print("\nSUCCESS! Relief centres created.")
print("Saved:", OUTPUT)
