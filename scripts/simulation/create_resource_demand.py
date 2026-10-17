
from pathlib import Path
import json

INPUT = Path(
    "data/processed/crisis_reports/aluva_report_timeline.json"
)

OUTPUT = Path(
    "data/processed/crisis_reports/aluva_resource_demand.json"
)

with open(INPUT, encoding="utf-8") as file:
    reports = json.load(file)

PEOPLE_PER_FAMILY = 4
WATER_LITRES_PER_PERSON = 3
FOOD_MEALS_PER_PERSON = 3

RESOURCE_FIELDS = {
    "water": "water_litres",
    "food": "food_meals",
    "rescue": "rescue_cases",
    "shelter": "shelter_households",
    "medicine": "medical_cases"
}

demand_events = []

for report in reports:
    families = report["affected_families"]
    patients = report.get("affected_patients")
    resources = report["requested_resources"]

    people = (
        families * PEOPLE_PER_FAMILY
        if families is not None
        else None
    )

    demand = {
        "water_litres": 0,
        "food_meals": 0,
        "rescue_cases": 0,
        "shelter_households": 0,
        "medical_cases": 0
    }

    if people is not None:
        if "water" in resources:
            demand["water_litres"] = (
                people * WATER_LITRES_PER_PERSON
            )

        if "food" in resources:
            demand["food_meals"] = (
                people * FOOD_MEALS_PER_PERSON
            )

        if "rescue" in resources:
            demand["rescue_cases"] = families

        if "shelter" in resources:
            demand["shelter_households"] = families

        if "medicine" in resources:
            demand["medical_cases"] = people

    # Explicit patient counts override estimated medical cases.
    if "medicine" in resources and patients is not None:
        demand["medical_cases"] = patients

    unknown_resources = []

    for resource in resources:
        if resource not in RESOURCE_FIELDS:
            raise ValueError(f"Unknown resource: {resource}")

        if people is None and not (
            resource == "medicine" and patients is not None
        ):
            unknown_resources.append(resource)

    demand_events.append({
        "report_id": report["report_id"],
        "ward_no": report["detected_ward_no"],
        "simulation_minute": report["simulation_minute"],
        "affected_families": families,
        "affected_patients": patients,
        "estimated_people": people,
        "demand": demand,
        "unquantified_resources": unknown_resources,
        "data_type": "synthetic"
    })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(demand_events, file, indent=4, ensure_ascii=False)

print("Demand events created:", len(demand_events))

for event in demand_events:
    print(
        event["report_id"],
        "| Ward:", event["ward_no"],
        "| Demand:", event["demand"],
        "| Unknown:", event["unquantified_resources"]
    )

print("\nSUCCESS! Saved:", OUTPUT)
