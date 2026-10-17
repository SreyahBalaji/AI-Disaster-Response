
import geopandas as gpd
import pandas as pd
from pathlib import Path

path = Path("data/processed/aluva/aluva_wards.geojson")
wards = gpd.read_file(path)

# Validate ward identities
wards["Ward_No"] = pd.to_numeric(
    wards["Ward_No"], errors="raise"
).astype(int)

expected = set(range(1, 27))
actual = set(wards["Ward_No"])

assert len(wards) == 26, "Expected 26 wards"
assert actual == expected, "Missing or unexpected ward numbers"
assert wards["Ward_No"].is_unique, "Duplicate ward numbers"
assert wards.geometry.notna().all(), "Missing geometries"
assert wards.geometry.is_valid.all(), "Invalid geometries"
assert wards.geometry.geom_type.isin(
    ["Polygon", "MultiPolygon"]
).all(), "Non-polygon geometry found"

# Use a projected CRS for accurate area calculations
projected = wards.to_crs(wards.estimate_utm_crs())

areas = projected.geometry.area
print("Total ward area (km²):", round(areas.sum() / 1e6, 3))

# Detect overlapping ward interiors
overlaps = []

for i in range(len(projected)):
    for j in range(i + 1, len(projected)):
        intersection = projected.geometry.iloc[i].intersection(
            projected.geometry.iloc[j]
        )
        overlap_m2 = intersection.area

        if overlap_m2 > 1:
            overlaps.append((
                int(wards.iloc[i]["Ward_No"]),
                int(wards.iloc[j]["Ward_No"]),
                round(overlap_m2, 2)
            ))

print("Overlapping ward pairs (>1 m²):", len(overlaps))

for overlap in overlaps[:10]:
    print("Overlap:", overlap)

# Check area covered by unique polygons
union_area = projected.geometry.union_all().area
print("Union area (km²):", round(union_area / 1e6, 3))

print("Ward numbers:", sorted(actual))
print("CRS:", wards.crs)

if overlaps:
    print("WARNING: Significant ward overlaps need investigation.")
else:
    print("SUCCESS: Ward identities and overlap checks passed.")

# Detailed overlap investigation
print("\n--- OVERLAP INVESTIGATION ---")

total_overlap = areas.sum() - union_area
overlap_percentage = (
    total_overlap / union_area * 100
    if union_area > 0 else 0
)

print("Excess overlapping area (m²):", round(total_overlap, 2))
print("Excess overlap percentage:", round(overlap_percentage, 4))

print("\nAll overlapping ward pairs:")
for ward_a, ward_b, overlap_m2 in sorted(
    overlaps, key=lambda item: item[2], reverse=True
):
    print(
        f"Ward {ward_a} vs Ward {ward_b}: "
        f"{overlap_m2:.2f} m²"
    )

print("\nWard areas:")
for _, ward in projected.iterrows():
    print(
        f"Ward {int(ward['Ward_No'])}: "
        f"{ward.geometry.area / 1e6:.4f} km²"
    )
