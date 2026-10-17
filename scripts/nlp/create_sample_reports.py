
from pathlib import Path
import json

OUTPUT = Path(
    "data/raw/crisis_reports/aluva_sample_reports.json"
)

reports = [
    {
        "report_id": "R001",
        "ward_no": 3,
        "text": "Floodwater entered houses in Desam Kadavu. 40 families need drinking water and rescue assistance."
    },
    {
        "report_id": "R002",
        "ward_no": 5,
        "text": "Heavy flooding reported in Manappuram. 25 families urgently require food and clean water."
    },
    {
        "report_id": "R003",
        "ward_no": 1,
        "text": "Residents near Mangalapuzha Seminary report rising water levels. 15 families need evacuation."
    },
    {
        "report_id": "R004",
        "ward_no": 20,
        "text": "Flooding near Thaluk Hospital has disrupted access. 12 patients require medical assistance."
    },
    {
        "report_id": "R005",
        "ward_no": 8,
        "text": "Kadathukadavu residents report flooded homes. 30 families need temporary shelter."
    },
    {
        "report_id": "R006",
        "ward_no": 23,
        "text": "Floodwater is rising near Market ward. 20 families need food supplies."
    },
    {
        "report_id": "R007",
        "ward_no": 24,
        "text": "Heavy rain and flooding in Priyadarsini. 35 families request rescue boats."
    },
    {
        "report_id": "R008",
        "ward_no": 9,
        "text": "Palace ward reports water entering residential streets. 18 families need drinking water."
    },
    {
        "report_id": "R009",
        "ward_no": 21,
        "text": "Sastha Temple area is affected by flooding. 10 families need medicine and food."
    },
    {
        "report_id": "R010",
        "ward_no": 26,
        "text": "Flooding reported in Thandikkal. 22 families need evacuation and shelter."
    }
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(reports, file, indent=4, ensure_ascii=False)

print("Total reports created:", len(reports))
print("Unique wards:", len(set(r["ward_no"] for r in reports)))
print("Data type: SYNTHETIC")
print("SUCCESS! Saved:", OUTPUT)
