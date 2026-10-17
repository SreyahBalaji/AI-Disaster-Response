
from pathlib import Path
import csv

from fastapi import APIRouter, HTTPException


router = APIRouter(
    prefix="/api",
    tags=["Ward Data"]
)

BASE_DIR = Path(__file__).resolve().parent.parent

PRIORITY_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "aluva"
    / "aluva_ward_priority.csv"
)


@router.get("/wards/priority")
def get_ward_priorities():
    if not PRIORITY_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Ward priority dataset not found"
        )

    wards = []

    with open(
        PRIORITY_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            record = {}

            for key, value in row.items():
                if value is None or value == "":
                    record[key] = None
                    continue

                try:
                    record[key] = float(value)
                except ValueError:
                    record[key] = value

            wards.append(record)

    return {
        "location": "Aluva, Kerala",
        "total_wards": len(wards),
        "priority_data": wards,
        "note": (
            "Priority scores are based on flood "
            "exposure and estimated population. "
            "They are not validated social "
            "vulnerability measurements."
        )
    }
