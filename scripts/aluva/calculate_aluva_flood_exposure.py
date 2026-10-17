
from pathlib import Path
import re

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

WARDS_PATH = Path(
    "data/processed/aluva/aluva_wards_clean.geojson"
)
FLOOD_DIR = Path("data/raw/flood")
OUTPUT = Path(
    "data/processed/aluva/aluva_flood_exposure.csv"
)

wards = gpd.read_file(WARDS_PATH)
rasters = list(FLOOD_DIR.rglob("*.tif"))

assert len(wards) == 26
assert len(rasters) == 6

results = []

for raster_path in rasters:
    match = re.search(r"Flood_(\d+)_yr", raster_path.name)
    if not match:
        raise ValueError(f"Unknown raster: {raster_path.name}")

    period = int(match.group(1))

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Missing CRS: {raster_path}")

        # Read masked raster to identify valid hazard cells.
        data = src.read(1, masked=True)
        valid_hazard = ~np.ma.getmaskarray(data)

        projected_wards = wards.to_crs(src.crs)

        for _, ward in projected_wards.iterrows():
            # Pixel centres inside the ward boundary.
            ward_pixels = geometry_mask(
                [ward.geometry],
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True,
                all_touched=False
            )

            total_pixels = int(ward_pixels.sum())
            hazard_pixels = int(
                (ward_pixels & valid_hazard).sum()
            )

            exposure = (
                hazard_pixels / total_pixels
                if total_pixels > 0
                else np.nan
            )

            results.append({
                "ward_no": int(ward["Ward_No"]),
                "ward_name": ward["Ward_Name"],
                "return_period_years": period,
                "ward_pixel_count": total_pixels,
                "hazard_pixel_count": hazard_pixels,
                "flood_exposure_proxy": exposure,
                "method": "valid_hazard_pixel_fraction"
            })

df = pd.DataFrame(results)
df = df.sort_values(
    ["ward_no", "return_period_years"]
)

assert len(df) == 26 * 6
assert (df["hazard_pixel_count"] <= df["ward_pixel_count"]).all()

valid = df["flood_exposure_proxy"].dropna()
assert valid.between(0, 1).all()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("Ward count:", df["ward_no"].nunique())
print("Return periods:", sorted(df["return_period_years"].unique()))
print("Total records:", len(df))
print("Wards with zero sampled pixels:",
      df.loc[df["ward_pixel_count"] == 0, "ward_no"].unique().tolist())

print("\nExposure summary:")
print(
    df.groupby("return_period_years")[
        "flood_exposure_proxy"
    ].agg(["count", "mean", "min", "max"])
)

print("\nSUCCESS! Saved:", OUTPUT)
