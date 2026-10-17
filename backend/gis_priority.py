
import csv
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException


router = APIRouter(
    prefix="/api",
    tags=["GIS Priority"]
)

BASE_DIR = Path(__file__).resolve().parent.parent

GEOJSON_FILE = (
    BASE_DIR / "data" / "processed"
    / "aluva" / "aluva_wards_clean.geojson"
)

PRIORITY_FILE = (
    BASE_DIR / "data" / "processed"
    / "aluva" / "aluva_ward_priority.csv"
)


@router.get("/gis/priority-map")
def get_priority_map():

    if not GEOJSON_FILE.exists() or not PRIORITY_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Required GIS dataset not found"
        )

    with open(GEOJSON_FILE, encoding="utf-8") as file:
        geojson = json.load(file)

    priority_lookup = {}

    with open(
        PRIORITY_FILE,
        encoding="utf-8-sig",
        newline=""
    ) as file:
        for row in csv.DictReader(file):
            ward_no = int(float(row["ward_no"]))
            priority_lookup[ward_no] = row

    features = []

    for feature in geojson["features"]:
        properties = feature["properties"]
        ward_no = int(properties["Ward_No"])

        priority = priority_lookup.get(ward_no)

        if priority is None:
            raise HTTPException(
                status_code=500,
                detail=f"Missing priority for Ward {ward_no}"
            )

        score = float(priority["flood_priority_score"])

        if score >= 0.45:
            category = "high"
        elif score >= 0.25:
            category = "moderate"
        else:
            category = "low"

        features.append({
            "type": "Feature",
            "properties": {
                "ward_no": ward_no,
                "ward_name": properties["Ward_Name"],
                "flood_priority_score": score,
                "priority_category": category,
                "flood_exposure_100yr": float(
                    priority["flood_exposure_100yr"]
                ),
                "population_score": float(
                    priority["population_score"]
                )
            },
            "geometry": feature["geometry"]
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }
