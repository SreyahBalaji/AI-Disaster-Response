
import geopandas as gpd
from pathlib import Path

INPUT = Path("data/raw/aluva/Ernakulam_Wards.geojson")
OUTPUT = Path("data/processed/aluva/aluva_wards.geojson")

# Load ward boundaries
wards = gpd.read_file(INPUT)

print("Total records:", len(wards))
print("CRS:", wards.crs)
print("\nAvailable columns:")
print(wards.columns.tolist())

print("\nFirst 5 records:")
print(wards.drop(columns="geometry").head().to_string(index=False))

# Search all text columns for Aluva
text_columns = wards.select_dtypes(
    include=["object", "string"]
).columns

matches = wards.index.to_series().map(lambda _: False)

for col in text_columns:
    matches |= (
        wards[col]
        .astype(str)
        .str.contains(r"\baluva\b", case=False, na=False)
    )

aluva = wards.loc[matches].copy()

print("\nAluva matching records:", len(aluva))

if aluva.empty:
    raise ValueError(
        "No Aluva records found. Inspect the printed columns."
    )

if aluva.crs is None:
    raise ValueError("Ward dataset has no CRS.")

if aluva.geometry.isna().any():
    raise ValueError("Missing ward geometries.")

# Check invalid geometries
invalid_count = (~aluva.geometry.is_valid).sum()
print("Invalid geometries before repair:", invalid_count)

# Repair invalid geometries
if invalid_count > 0:
    aluva["geometry"] = aluva.geometry.make_valid()

# Validate after repair
remaining_invalid = (~aluva.geometry.is_valid).sum()

print("Invalid geometries after repair:", remaining_invalid)

if remaining_invalid > 0:
    raise ValueError("Some ward geometries are still invalid.")

# Ensure all geometries remain polygons
if not aluva.geometry.geom_type.isin(
    ["Polygon", "MultiPolygon"]
).all():
    raise ValueError(
        "Repair produced non-polygon geometries. Manual inspection needed."
    )

# Check duplicate ward numbers
if aluva["Ward_No"].duplicated().any():
    raise ValueError("Duplicate ward numbers detected.")

aluva = aluva.to_crs("EPSG:4326")

print("\nExtracted ward records:")
print(aluva.drop(columns="geometry").to_string(index=False))

if len(aluva) != 26:
    print(
        f"\nWARNING: Expected 26 wards, found {len(aluva)}."
        "\nCheck whether the source uses a different ward year"
        " or contains Aluva references in other municipalities."
    )
else:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    aluva.to_file(OUTPUT, driver="GeoJSON")
    print("\nSUCCESS! Saved:", OUTPUT)
