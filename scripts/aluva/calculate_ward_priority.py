
from pathlib import Path

import pandas as pd

input_path = Path(
    "data/processed/aluva/aluva_ward_features.csv"
)

output_path = Path(
    "data/processed/aluva/aluva_ward_priority.csv"
)

df = pd.read_csv(input_path)

required = [
    "ward_no",
    "ward_name",
    "population_2011_estimated",
    "flood_exposure_100yr"
]

assert all(column in df.columns for column in required)
assert len(df) == 26

max_population = df["population_2011_estimated"].max()

df["population_score"] = (
    df["population_2011_estimated"] / max_population
)

df["flood_priority_score"] = (
    0.6 * df["flood_exposure_100yr"]
    + 0.4 * df["population_score"]
)

df["flood_priority_score"] = df[
    "flood_priority_score"
].round(4)

assert df["flood_priority_score"].between(0, 1).all()

result = df[
    [
        "ward_no",
        "ward_name",
        "population_score",
        "flood_exposure_100yr",
        "flood_priority_score"
    ]
].sort_values(
    "flood_priority_score",
    ascending=False
)

result.to_csv(output_path, index=False)

print("ALUVA FLOOD PRIORITY SCORES")
print("--------------------------------")
print(result.head(5).to_string(index=False))
print("\nSaved:", output_path)
print("\nSUCCESS! Ward priority scores calculated.")
