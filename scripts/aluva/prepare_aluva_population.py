
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

WARDS_FILE = Path(
    "data/processed/aluva/aluva_wards_clean.geojson"
)
CENSUS_FILE = Path(
    "data/processed/aluva/aluva_census_2011.csv"
)
OUTPUT_FILE = Path(
    "data/processed/aluva/aluva_population.csv"
)

# Load verified datasets
wards = gpd.read_file(WARDS_FILE)
census = pd.read_csv(CENSUS_FILE)

town = census[census["Level"] == "TOWN"]

if len(town) != 1:
    raise ValueError("Expected exactly one Aluva town record.")

population_total = int(town["TOT_P"].iloc[0])

# Calculate ward areas in square metres
projected = wards.to_crs(wards.estimate_utm_crs())

areas = projected.geometry.area.to_numpy()
weights = areas / areas.sum()

# Allocate population using largest remainder method
raw_population = population_total * weights
population = np.floor(raw_population).astype(int)

remaining = population_total - population.sum()

remainders = raw_population - population
priority = np.argsort(-remainders, kind="stable")

population[priority[:remaining]] += 1

result = pd.DataFrame({
    "ward_no": wards["Ward_No"].astype(int),
    "ward_name": wards["Ward_Name"],
    "area_km2": np.round(areas / 1e6, 6),
    "population_2011_estimated": population,
    "data_type": "semi-synthetic",
    "method": "area_proportional_allocation"
})

result = result.sort_values("ward_no")

# Validation
assert len(result) == 26
assert result["ward_no"].is_unique
assert result["population_2011_estimated"].sum() == population_total
assert (result["population_2011_estimated"] >= 0).all()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False)

print("Ward count:", len(result))
print("Official 2011 population:", population_total)
print(
    "Allocated population:",
    result["population_2011_estimated"].sum()
)
print("\nWard-level population estimates:")
print(result.to_string(index=False))

print("\nSUCCESS! Saved:", OUTPUT_FILE)
