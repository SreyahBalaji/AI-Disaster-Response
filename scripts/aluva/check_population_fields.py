
import geopandas as gpd

path = "data/processed/aluva/aluva_wards_clean.geojson"
wards = gpd.read_file(path)

print("Available columns:")
print(wards.columns.tolist())

print("\nPopulation-related columns:")
matches = [
    col for col in wards.columns
    if any(term in col.lower() for term in
           ["population", "pop", "census", "household"])
]

print(matches if matches else "None found")

print("\nWard names and numbers:")
print(
    wards[["Ward_No", "Ward_Name"]]
    .sort_values("Ward_No")
    .to_string(index=False)
)
