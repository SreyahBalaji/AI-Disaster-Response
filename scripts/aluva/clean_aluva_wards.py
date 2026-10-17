
import geopandas as gpd
from pathlib import Path

INPUT = Path("data/processed/aluva/aluva_wards.geojson")
OUTPUT = Path("data/processed/aluva/aluva_wards_clean.geojson")

wards = gpd.read_file(INPUT)

# Work in metres
crs_original = wards.crs
wards = wards.to_crs(wards.estimate_utm_crs())

# Deterministic order: lower ward number gets priority
wards["Ward_No"] = wards["Ward_No"].astype(int)
wards = wards.sort_values("Ward_No").reset_index(drop=True)

original_union = wards.geometry.union_all()
original_area = original_union.area

# Remove areas already assigned to earlier wards
assigned = None
cleaned = []

for _, ward in wards.iterrows():
    geom = ward.geometry

    if assigned is not None:
        geom = geom.difference(assigned)

    if geom.is_empty:
        raise ValueError(
            f"Ward {ward['Ward_No']} became empty."
        )

    if not geom.is_valid:
        geom = gpd.GeoSeries(
            [geom], crs=wards.crs
        ).make_valid().iloc[0]

    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            f"Ward {ward['Ward_No']} has unexpected geometry."
        )

    cleaned.append(geom)

    assigned = geom if assigned is None else assigned.union(geom)

wards["geometry"] = cleaned

# Validate the cleaned result
assert wards.geometry.is_valid.all()
assert len(wards) == 26

clean_union = wards.geometry.union_all()
area_difference = original_union.symmetric_difference(
    clean_union
).area

print("Ward count:", len(wards))
print("Original union area (km²):", round(original_area / 1e6, 6))
print("Clean union area (km²):", round(clean_union.area / 1e6, 6))
print("Area difference (m²):", round(area_difference, 4))

if area_difference > 0.01:
    raise ValueError("Cleaning unexpectedly changed coverage.")

wards = wards.to_crs(crs_original)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
wards.to_file(OUTPUT, driver="GeoJSON")

print("SUCCESS! Cleaned ward boundaries saved:", OUTPUT)
