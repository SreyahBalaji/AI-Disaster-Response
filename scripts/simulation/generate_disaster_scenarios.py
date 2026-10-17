
from pathlib import Path
import json
import math

INPUT = Path(
    "data/processed/crisis_reports/aluva_resource_demand.json"
)

OUTPUT = Path(
    "data/processed/crisis_reports/aluva_disaster_scenarios.json"
)

with open(INPUT, encoding="utf-8") as file:
    events = json.load(file)

SCENARIOS = {
    "low": 0.5,
    "moderate": 1.0,
    "severe": 2.0
}

RESOURCE_FIELDS = [
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases"
]

scenarios = {}

for scenario_name, multiplier in SCENARIOS.items():
    scenario_events = []

    for event in events:
        scaled_demand = {
            resource: math.ceil(
                event["demand"][resource] * multiplier
            )
            for resource in RESOURCE_FIELDS
        }

        scenario_events.append({
            "report_id": event["report_id"],
            "ward_no": event["ward_no"],
            "simulation_minute": event["simulation_minute"],
            "demand": scaled_demand,
            "unquantified_resources": event["unquantified_resources"]
        })

    scenarios[scenario_name] = {
        "multiplier": multiplier,
        "data_type": "synthetic",
        "events": scenario_events
    }

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(scenarios, file, indent=2)

for name, scenario in scenarios.items():
    totals = {
        resource: sum(
            event["demand"][resource]
            for event in scenario["events"]
        )
        for resource in RESOURCE_FIELDS
    }

    print(f"\n{name.upper()} SCENARIO")
    print("Events:", len(scenario["events"]))

    for resource, total in totals.items():
        print(f"{resource}: {total}")

print("\nSUCCESS! Saved:", OUTPUT)
