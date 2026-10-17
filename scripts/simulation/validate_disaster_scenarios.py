
from pathlib import Path
import json
import math

BASE = Path("data/processed/crisis_reports")

with open(BASE / "aluva_resource_demand.json", encoding="utf-8") as f:
    original_events = json.load(f)

with open(BASE / "aluva_disaster_scenarios.json", encoding="utf-8") as f:
    scenarios = json.load(f)

EXPECTED_MULTIPLIERS = {
    "low": 0.5,
    "moderate": 1.0,
    "severe": 2.0
}

EXPECTED_TOTALS = {
    "low": [498, 330, 57, 26, 26],
    "moderate": [996, 660, 112, 52, 52],
    "severe": [1992, 1320, 224, 104, 104]
}

RESOURCES = [
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases"
]

assert set(scenarios) == set(EXPECTED_MULTIPLIERS)

for name, multiplier in EXPECTED_MULTIPLIERS.items():
    scenario = scenarios[name]
    events = scenario["events"]

    assert scenario["multiplier"] == multiplier
    assert scenario["data_type"] == "synthetic"
    assert len(events) == len(original_events) == 10

    for original, scaled in zip(original_events, events):
        assert scaled["report_id"] == original["report_id"]
        assert scaled["ward_no"] == original["ward_no"]
        assert scaled["simulation_minute"] == original["simulation_minute"]
        assert scaled["unquantified_resources"] == original["unquantified_resources"]

        for resource in RESOURCES:
            expected = math.ceil(
                original["demand"][resource] * multiplier
            )

            assert scaled["demand"][resource] == expected, (
                f"{name}: {scaled['report_id']} "
                f"incorrect {resource}"
            )

    totals = [
        sum(event["demand"][resource] for event in events)
        for resource in RESOURCES
    ]

    assert totals == EXPECTED_TOTALS[name], (
        f"{name}: unexpected totals {totals}"
    )

    print(f"{name.upper()} | 10 events | PASS")

print("\nVALIDATION RESULTS")
print("Scenarios validated:", len(scenarios))
print("Total scenario events validated:", 30)
print("Resource categories:", len(RESOURCES))

print("\nSUCCESS! Phase 3 scenario validation passed.")
