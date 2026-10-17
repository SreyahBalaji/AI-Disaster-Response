
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(
    "data/processed/aluva/aluva_flood_exposure.csv"
)

df = pd.read_csv(PATH)

periods = [10, 25, 50, 100, 200, 500]

assert len(df) == 156
assert df["ward_no"].nunique() == 26
assert sorted(df["return_period_years"].unique()) == periods

assert not df.duplicated(
    ["ward_no", "return_period_years"]
).any()

assert df["flood_exposure_proxy"].notna().all()
assert df["flood_exposure_proxy"].between(0, 1).all()

assert (
    df["hazard_pixel_count"] <= df["ward_pixel_count"]
).all()

pivot = df.pivot(
    index="ward_no",
    columns="return_period_years",
    values="flood_exposure_proxy"
)[periods]

# Exposure should not decrease for a longer return period.
differences = np.diff(pivot.to_numpy(), axis=1)

violations = np.argwhere(differences < -1e-9)

print("Wards:", len(pivot))
print("Return periods:", periods)
print("Total records:", len(df))
print("Monotonicity violations:", len(violations))

for row, col in violations:
    ward = pivot.index[row]
    print(
        f"Ward {ward}: "
        f"{periods[col]} -> {periods[col + 1]} years"
    )

print("\nPixel count summary:")
print(
    df.groupby("return_period_years")[
        "ward_pixel_count"
    ].agg(["min", "max"])
)

if len(violations) == 0:
    print("\nSUCCESS! All flood exposure checks passed.")
else:
    print("\nWARNING: Review monotonicity violations.")
