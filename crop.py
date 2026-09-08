import rasterio
from rasterio.windows import Window
from pathlib import Path
import numpy as np

# Input SAR
input_file = r"C:\Users\palak\processing\SAR_common.tif"

# Output folder
output_dir = Path(r"C:\Users\palak\processing\SAR_128x128")
output_dir.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 128
MIN_VALID_PERCENT = 90   # keep patches with >=90% valid pixels

with rasterio.open(input_file) as src:

    print("SAR size:", src.width, "x", src.height)
    print("Pixel size:", src.res)
    print("CRS:", src.crs)

    nodata = src.nodata
    patch_id = 0

    for row in range(0, src.height - PATCH_SIZE + 1, PATCH_SIZE):

        for col in range(0, src.width - PATCH_SIZE + 1, PATCH_SIZE):

            window = Window(
                col,
                row,
                PATCH_SIZE,
                PATCH_SIZE
            )

            data = src.read(1, window=window)

            # Determine valid pixels
            if nodata is not None:
                valid = data != nodata
            else:
                valid = data > 0

            valid_percent = 100 * np.count_nonzero(valid) / data.size

            # Skip mostly empty patches
            if valid_percent < MIN_VALID_PERCENT:
                continue

            # Preserve exact geographic transform
            transform = src.window_transform(window)

            profile = src.profile.copy()

            profile.update({
                "driver": "GTiff",
                "width": PATCH_SIZE,
                "height": PATCH_SIZE,
                "transform": transform,
                "compress": "lzw"
            })

            output_file = output_dir / f"SAR_{patch_id:05d}.tif"

            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(data, 1)

            patch_id += 1

print(f"\nCreated {patch_id} SAR patches.")
print("Each patch: 128 × 128 pixels ≈ 2.3 × 2.3 km")
print("All patches remain georeferenced GeoTIFFs.")