
import json
from pathlib import Path

PATH = Path(
    "data/processed/aluva/aluva_relief_centres.json"
)

with open(PATH, "r", encoding="utf-8") as file:
    centres = json.load(file)

required_resources = {
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases"
}

assert len(centres) == 3

centre_ids = set()
centre_wards = set()

for centre in centres:
    centre_id = centre["centre_id"]
    ward = centre["ward_no"]
    resources = centre["resources"]

    assert centre_id not in centre_ids
    centre_ids.add(centre_id)

    assert isinstance(ward, int)
    assert 1 <= ward <= 26
    assert ward not in centre_wards
    centre_wards.add(ward)

    assert set(resources) == required_resources

    for quantity in resources.values():
        assert isinstance(quantity, int)
        assert quantity >= 0

print("RELIEF CENTRE VALIDATION")
print("-----------------------------")
print("Total centres:", len(centres))
print("Unique centre IDs: PASS")
print("Valid ward locations: PASS")
print("Resource inventory: PASS")
print("Non-negative quantities: PASS")

print("\nSUCCESS! Relief centre validation passed.")
