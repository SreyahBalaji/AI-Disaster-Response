
from pathlib import Path

import pandas as pd

BASE = Path("data/processed/aluva")

population = pd.read_csv(BASE / "aluva_population.csv")
flood = pd.read_csv(BASE / "aluva_flood_exposure.csv")
edges = pd.read_csv(BASE / "aluva_graph_edges.csv")

periods = [10, 25, 50, 100, 200, 500]

# One flood exposure feature per return period.
flood_wide = flood.pivot(
    index="ward_no",
    columns="return_period_years",
    values="flood_exposure_proxy"
)

assert set(flood_wide.columns) == set(periods)

flood_wide = flood_wide[periods]
flood_wide.columns = [
    f"flood_exposure_{period}yr"
    for period in periods
]

flood_wide = flood_wide.reset_index()

# Merge by ward number, not row position.
features = population.merge(
    flood_wide,
    on="ward_no",
    how="inner",
    validate="one_to_one"
)

features = features.sort_values("ward_no").reset_index(drop=True)

# Validate all 26 graph nodes have features.
expected_wards = set(range(1, 27))

assert len(features) == 26
assert set(features["ward_no"]) == expected_wards

assert set(edges["source_ward"]).union(
    set(edges["target_ward"])
).issubset(expected_wards)

assert features["population_2011_estimated"].sum() == 22428

flood_columns = [
    f"flood_exposure_{period}yr"
    for period in periods
]

assert features[flood_columns].notna().all().all()
assert features[flood_columns].ge(0).all().all()
assert features[flood_columns].le(1).all().all()

output = BASE / "aluva_ward_features.csv"
features.to_csv(output, index=False)

print("Total wards:", len(features))
print("Total columns:", len(features.columns))
print("Total estimated population:",
      features["population_2011_estimated"].sum())
print("Flood feature columns:", flood_columns)
print("Missing values:", features.isna().sum().sum())

print("\nPreview:")
print(features.head().to_string(index=False))

print("\nSUCCESS! Saved:", output)
