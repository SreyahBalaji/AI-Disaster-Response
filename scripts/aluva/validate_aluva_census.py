
import pandas as pd

path = "data/processed/aluva/aluva_census_2011.csv"

df = pd.read_csv(
    path,
    dtype={"Ward": str, "Town/Village": str}
)

town = df[df["Level"] == "TOWN"]
wards = df[df["Level"] == "WARD"]

assert len(town) == 1, "Expected one municipality record"
assert len(wards) == 23, "Expected 23 historical wards"

# Check historical ward identifiers
expected = {f"{i:04d}" for i in range(1, 24)}
actual = set(wards["Ward"])

assert actual == expected, "Missing or duplicate ward identifiers"
assert wards["Ward"].is_unique, "Duplicate ward numbers"

for column in ["TOT_P", "TOT_M", "TOT_F", "No_HH"]:
    town_total = int(town[column].iloc[0])
    ward_total = int(wards[column].sum())

    print(f"{column}: Municipality={town_total}, Wards={ward_total}")

    assert town_total == ward_total, (
        f"{column} totals do not match"
    )

assert (df["TOT_P"] == df["TOT_M"] + df["TOT_F"]).all(), (
    "Male and female population totals do not match"
)

print("\nSUCCESS: All census totals validated.")
