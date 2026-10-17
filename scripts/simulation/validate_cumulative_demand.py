
from pathlib import Path
import json

BASE = Path("data/processed/crisis_reports")

with open(BASE / "aluva_resource_demand.json", encoding="utf-8") as f:
    events = json.load(f)

with open(BASE / "aluva_cumulative_demand.json", encoding="utf-8") as f:
    snapshots = json.load(f)

RESOURCES = [
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases"
]

events.sort(key=lambda e: e["simulation_minute"])

assert len(events) == len(snapshots) == 10

expected_state = {
    str(ward): {resource: 0 for resource in RESOURCES}
    for ward in range(1, 27)
}

seen_reports = set()
previous_minute = -1

for event, snapshot in zip(events, snapshots):
    report_id = event["report_id"]
    minute = event["simulation_minute"]
    ward = str(event["ward_no"])

    assert report_id not in seen_reports
    seen_reports.add(report_id)

    assert minute > previous_minute
    previous_minute = minute

    assert snapshot["trigger_report_id"] == report_id
    assert snapshot["simulation_minute"] == minute
    assert snapshot["updated_ward"] == event["ward_no"]

    for resource in RESOURCES:
        expected_state[ward][resource] += event["demand"][resource]

    actual_state = snapshot["ward_demand"]

    assert set(actual_state.keys()) == set(expected_state.keys())

    for ward_id in expected_state:
        assert actual_state[ward_id] == expected_state[ward_id], (
            f"Incorrect demand in ward {ward_id} "
            f"after report {report_id}"
        )

    print(f"{report_id} | Minute {minute} | PASS")

final_totals = {
    resource: sum(
        ward[resource]
        for ward in expected_state.values()
    )
    for resource in RESOURCES
}

expected_totals = {
    "water_litres": 996,
    "food_meals": 660,
    "rescue_cases": 112,
    "shelter_households": 52,
    "medical_cases": 52
}

assert final_totals == expected_totals

print("\nVALIDATION RESULTS")
print("Reports validated:", len(seen_reports))
print("Snapshots validated:", len(snapshots))
print("Ward states per snapshot:", len(expected_state))
print("Final demand totals:", final_totals)

print("\nSUCCESS! All temporal demand updates validated.")
