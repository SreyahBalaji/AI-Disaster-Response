
import json
from pathlib import Path

path = Path(
    "data/processed/federated/aluva_allocation_training_data.json"
)

with open(path, "r", encoding="utf-8") as file:
    records = json.load(file)

print("ALLOCATION DATA VALIDATION")
print("----------------------------")

assert len(records) == 9

centres = set()

for record in records:
    centres.add(record["centre_id"])

    assert len(record["features"]) == 5
    assert all(
        isinstance(value, (int, float))
        and 0 <= value <= 1
        for value in record["features"]
    )

    assert 0 <= record["target"] <= 1

    assert record["ward_no"] in range(1, 27)

assert centres == {"RC01", "RC02", "RC03"}

print("Records validated:", len(records))
print("Relief centres:", sorted(centres))
print("Features per record: 5")
print("Targets within [0, 1]: Yes")

print("\nSUCCESS! Allocation dataset validated.")
