
from pathlib import Path
import json

INPUT = Path(
    "data/processed/crisis_reports/aluva_resource_demand.json"
)

OUTPUT = Path(
    "data/processed/crisis_reports/aluva_cumulative_demand.json"
)

with open(INPUT, encoding="utf-8") as file:
    events = json.load(file)

RESOURCE_FIELDS = [
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases"
]

WARD_COUNT = 26

# Start with zero demand in all wards.
ward_demand = {
    ward: {resource: 0 for resource in RESOURCE_FIELDS}
    for ward in range(1, WARD_COUNT + 1)
}

snapshots = []

events = sorted(
    events,
    key=lambda event: event["simulation_minute"]
)

for event in events:
    ward = event["ward_no"]

    if ward not in ward_demand:
        raise ValueError(f"Unknown ward: {ward}")

    # Add newly reported demand.
    for resource in RESOURCE_FIELDS:
        ward_demand[ward][resource] += event["demand"][resource]

    # Save the full system state after this event.
    snapshot = {
        "simulation_minute": event["simulation_minute"],
        "trigger_report_id": event["report_id"],
        "updated_ward": ward,
        "ward_demand": {
            str(ward_id): demand.copy()
            for ward_id, demand in ward_demand.items()
        }
    }

    snapshots.append(snapshot)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(snapshots, file, indent=2)

print("Total snapshots:", len(snapshots))
print("Wards per snapshot:", len(snapshots[0]["ward_demand"]))

print("\nCumulative demand after each report:")

for snapshot in snapshots:
    totals = {
        resource: sum(
            ward[resource]
            for ward in snapshot["ward_demand"].values()
        )
        for resource in RESOURCE_FIELDS
    }

    print(
        f"Minute {snapshot['simulation_minute']:>3} | "
        f"{snapshot['trigger_report_id']} | "
        f"Water: {totals['water_litres']} | "
        f"Food: {totals['food_meals']} | "
        f"Rescue: {totals['rescue_cases']} | "
        f"Shelter: {totals['shelter_households']} | "
        f"Medical: {totals['medical_cases']}"
    )

print("\nSUCCESS! Saved:", OUTPUT)
