
import json
from pathlib import Path

INPUT = Path(
    "data/processed/crisis_reports/aluva_resource_demand.json"
)

with open(INPUT, encoding="utf-8") as file:
    events = json.load(file)

# Expected values for the controlled synthetic scenario.
expected = {
    "R001": {"water_litres": 480, "rescue_cases": 40},
    "R002": {"water_litres": 300, "food_meals": 300},
    "R003": {"rescue_cases": 15},
    "R004": {"medical_cases": 12},
    "R005": {"shelter_households": 30},
    "R006": {"food_meals": 240},
    "R007": {"rescue_cases": 35},
    "R008": {"water_litres": 216},
    "R009": {"food_meals": 120, "medical_cases": 40},
    "R010": {"rescue_cases": 22, "shelter_households": 22}
}

resource_fields = [
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases"
]

assert len(events) == 10
assert len({e["report_id"] for e in events}) == 10

for event in events:
    report_id = event["report_id"]
    assert report_id in expected

    for resource in resource_fields:
        actual = event["demand"][resource]
        required = expected[report_id].get(resource, 0)

        assert actual == required, (
            f"{report_id}: {resource} "
            f"expected {required}, got {actual}"
        )

    assert not event["unquantified_resources"]

    print(report_id, "PASS")

r004 = next(e for e in events if e["report_id"] == "R004")

assert r004["affected_families"] is None
assert r004["affected_patients"] == 12
assert r004["demand"]["medical_cases"] == 12

totals = {
    resource: sum(e["demand"][resource] for e in events)
    for resource in resource_fields
}

print("\nTOTAL SIMULATED DEMAND")

for resource, quantity in totals.items():
    print(f"{resource}: {quantity}")

print("\nSUCCESS! All demand calculations validated.")
