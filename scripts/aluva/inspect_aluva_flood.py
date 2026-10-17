
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask

FLOOD_DIR = Path("data/raw/flood")
WARDS_FILE = Path(
    "data/processed/aluva/aluva_wards_clean.geojson"
)

wards = gpd.read_file(WARDS_FILE)

rasters = sorted(FLOOD_DIR.rglob("*.tif"))

assert len(rasters) == 6, (
    f"Expected 6 flood rasters, found {len(rasters)}"
)

# Combine Aluva's wards into one study-area geometry
aluva_boundary = wards.geometry.union_all()

for raster_path in rasters:
    print("\n" + "=" * 55)
    print("Raster:", raster_path.name)

    with rasterio.open(raster_path) as src:
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("NoData:", src.nodata)
        print("Bands:", src.count)
        print("Data type:", src.dtypes[0])
        print("Bounds:", src.bounds)
        print("Descriptions:", src.descriptions)

        print("Band 1 tags:", src.tags(1))

        # Match ward coordinates to raster coordinates
        boundary = gpd.GeoSeries(
            [aluva_boundary],
            crs=wards.crs
        ).to_crs(src.crs)

        try:
            data, transform = mask(
                src,
                boundary.geometry,
                crop=True,
                filled=False,
                indexes=1
            )
        except ValueError as error:
            print("ERROR: Aluva does not overlap raster.")
            print(error)
            continue

        valid = data.compressed()

        print("Unmasked pixels:", len(valid))

        if len(valid) > 0:
            values, counts = np.unique(
                valid, return_counts=True
            )

            print("Unique values:", len(values))

            if len(values) <= 20:
                print(
                    "Value counts:",
                    list(zip(values.tolist(), counts.tolist()))
                )
            else:
                print("Minimum:", float(valid.min()))
                print("Maximum:", float(valid.max()))
                print("Sample values:", values[:10].tolist())

print("\nINSPECTION COMPLETED")
