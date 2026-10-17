
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException


router = APIRouter(
    prefix="/api",
    tags=["GIS Map"]
)

BASE_DIR = Path(__file__).resolve().parent.parent

GEOJSON_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "aluva"
    / "aluva_wards_clean.geojson"
)


@router.get("/gis/wards")
def get_ward_boundaries():

    if not GEOJSON_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Aluva ward GeoJSON file not found"
        )

    try:
        with open(
            GEOJSON_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            geojson = json.load(file)

    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read GeoJSON: {error}"
        )

    if geojson.get("type") != "FeatureCollection":
        raise HTTPException(
            status_code=500,
            detail="Invalid GeoJSON FeatureCollection"
        )

    features = geojson.get("features", [])

    if not isinstance(features, list):
        raise HTTPException(
            status_code=500,
            detail="GeoJSON features must be a list"
        )

    return {
        "type": "FeatureCollection",
        "features": features
    }


@router.get("/gis/status")
def gis_status():

    if not GEOJSON_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="GeoJSON file not found"
        )

    with open(
        GEOJSON_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        geojson = json.load(file)

    features = geojson.get("features", [])

    return {
        "location": "Aluva, Kerala",
        "geojson_file": GEOJSON_FILE.name,
        "total_features": len(features),
        "geometry_types": sorted({
            feature.get("geometry", {}).get("type")
            for feature in features
            if isinstance(feature.get("geometry"), dict)
            and feature["geometry"].get("type")
        }),
        "sample_properties": (
            features[0].get("properties", {})
            if features else {}
        )
    }
