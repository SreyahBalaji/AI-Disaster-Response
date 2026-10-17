
import json
from pathlib import Path

BASE = Path("data")

INPUT = BASE / "processed/crisis_reports/aluva_extracted_reports.json"

with open(INPUT, encoding="utf-8") as file:
    predictions = json.load(file)

# Independently defined expected answers for our
# 10 synthetic test reports.
expected = {
    "R001": (3, 40, {"water", "rescue"}),
    "R002": (5, 25, {"water", "food"}),
    "R003": (1, 15, {"rescue"}),
    "R004": (20, None, {"medicine"}),
    "R005": (8, 30, {"shelter"}),
    "R006": (23, 20, {"food"}),
    "R007": (24, 35, {"rescue"}),
    "R008": (9, 18, {"water"}),
    "R009": (21, 10, {"food", "medicine"}),
    "R010": (26, 22, {"rescue", "shelter"}),
}

assert len(predictions) == len(expected)
assert len({p["report_id"] for p in predictions}) == len(predictions)

ward_correct = 0
count_correct = 0
resource_correct = 0

for report in predictions:
    report_id = report["report_id"]

    assert report_id in expected, f"Unexpected report: {report_id}"

    ward, families, resources = expected[report_id]

    ward_ok = report["detected_ward_no"] == ward
    count_ok = report["affected_families"] == families
    resources_ok = set(report["requested_resources"]) == resources

    ward_correct += ward_ok
    count_correct += count_ok
    resource_correct += resources_ok

    print(
        report_id,
        "| Ward:", "PASS" if ward_ok else "FAIL",
        "| Count:", "PASS" if count_ok else "FAIL",
        "| Resources:", "PASS" if resources_ok else "FAIL"
    )

total = len(predictions)

print("\nVALIDATION RESULTS")
print(f"Ward accuracy: {ward_correct}/{total}")
print(f"Count exact-match accuracy: {count_correct}/{total}")
print(f"Resource-set exact-match accuracy: {resource_correct}/{total}")

if all(x == total for x in [
    ward_correct, count_correct, resource_correct
]):
    print("\nSUCCESS! All synthetic extraction tests passed.")
else:
    print("\nWARNING: Some extraction tests failed.")
