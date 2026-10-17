
from pathlib import Path
import json
import re

import geopandas as gpd

BASE = Path("data")
INPUT = BASE / "raw/crisis_reports/aluva_sample_reports.json"
WARDS = BASE / "processed/aluva/aluva_wards_clean.geojson"
OUTPUT = BASE / "processed/crisis_reports/aluva_extracted_reports.json"

with open(INPUT, encoding="utf-8") as file:
    reports = json.load(file)

wards = gpd.read_file(WARDS)

ward_lookup = {
    str(row["Ward_Name"]).lower(): int(row["Ward_No"])
    for _, row in wards.iterrows()
}

aliases = {
    "mangalapuzha seminary": 1,
    "desam kadavu": 3,
    "manappuram": 5,
    "kadathukadavu": 8,
    "palace": 9,
    "thaluk hospital": 20,
    "sastha temple": 21,
    "market": 23,
    "priyadarsini": 24,
    "thandikkal": 26,
}

ward_lookup.update(aliases)

resource_keywords = {
    "water": ["drinking water", "clean water", "water supplies"],
    "food": ["food", "food supplies"],
    "medicine": ["medicine", "medical assistance"],
    "rescue": ["rescue", "evacuation", "rescue boats"],
    "shelter": ["shelter", "temporary shelter"]
}


def extract_information(text):
    lower_text = text.lower()

    detected_ward = None

    for name, number in sorted(
        ward_lookup.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):
        if name in lower_text:
            detected_ward = number
            break

    family_match = re.search(
        r"\b(\d+)\s+famil(?:y|ies)\b",
        lower_text
    )

    affected_families = (
        int(family_match.group(1))
        if family_match
        else None
    )

    patient_match = re.search(
        r"\b(\d+)\s+patients?\b",
        lower_text
    )

    affected_patients = (
        int(patient_match.group(1))
        if patient_match
        else None
    )

    resources = []

    for resource, keywords in resource_keywords.items():
        if any(keyword in lower_text for keyword in keywords):
            resources.append(resource)

    return {
        "detected_ward_no": detected_ward,
        "affected_families": affected_families,
        "affected_patients": affected_patients,
        "requested_resources": resources
    }


extracted = []

for report in reports:
    result = extract_information(report["text"])

    extracted.append({
        "report_id": report["report_id"],
        "text": report["text"],
        **result,
        "source_type": "synthetic",
        "extraction_method": "rule_based_baseline"
    })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(extracted, file, indent=4, ensure_ascii=False)

print("Reports processed:", len(extracted))

for report in extracted:
    print(
        report["report_id"],
        "| Ward:", report["detected_ward_no"],
        "| Families:", report["affected_families"],
        "| Patients:", report["affected_patients"],
        "| Resources:", report["requested_resources"]
    )

print("\nSUCCESS! Saved:", OUTPUT)
