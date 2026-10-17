
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json

INPUT = Path(
    "data/processed/crisis_reports/aluva_extracted_reports.json"
)

OUTPUT = Path(
    "data/processed/crisis_reports/aluva_report_timeline.json"
)

with open(INPUT, encoding="utf-8") as file:
    reports = json.load(file)

# Fixed simulation start for reproducible experiments.
start_time = datetime(
    2026, 8, 15, 6, 0,
    tzinfo=timezone.utc
)

# Simulated arrival offsets, in minutes.
arrival_offsets = [0, 5, 12, 20, 28, 40, 52, 65, 80, 95]

assert len(reports) == len(arrival_offsets)

timeline = []

for report, offset in zip(reports, arrival_offsets):
    event = {
        **report,
        "event_time": (
            start_time + timedelta(minutes=offset)
        ).isoformat(),
        "simulation_minute": offset,
        "time_type": "synthetic"
    }

    timeline.append(event)

timeline.sort(key=lambda x: x["simulation_minute"])

assert len(timeline) == 10
assert all(
    timeline[i]["simulation_minute"]
    <= timeline[i + 1]["simulation_minute"]
    for i in range(len(timeline) - 1)
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(timeline, file, indent=4, ensure_ascii=False)

print("Total timeline events:", len(timeline))
print("First event:", timeline[0]["event_time"])
print("Last event:", timeline[-1]["event_time"])

print("\nReport arrival timeline:")

for event in timeline:
    print(
        f"Minute {event['simulation_minute']:>3} | "
        f"{event['report_id']} | "
        f"Ward {event['detected_ward_no']} | "
        f"Resources: {event['requested_resources']}"
    )

print("\nSUCCESS! Saved:", OUTPUT)
