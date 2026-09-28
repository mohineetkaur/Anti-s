import os
import math
import requests
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import (
    transform_bounds,
    calculate_default_transform,
    reproject,
    Resampling
)


# ============================================================
# SETTINGS
# ============================================================

SAR_PATH = r"C:\Users\palak\Downloads\sar_grd_db.tif"

OUTPUT_DIR = r"C:\Users\palak\Downloads\sar_rgb_output"

ZOOM = 18

TILE_SIZE = 256
MAX_WORKERS = 8

GOOGLE_URL = (
    "https://mt1.google.com/vt/"
    "lyrs=s&x={x}&y={y}&z={z}"
)


# ============================================================
# SAR REFERENCE
# ============================================================

def get_sar_reference(path):

    with rasterio.open(path) as sar:

        return {
            "crs": sar.crs,
            "bounds": sar.bounds,
            "width": sar.width,
            "height": sar.height,
            "transform": sar.transform,
            "res": sar.res
        }


# ============================================================
# SAR BOUNDS → EPSG:3857
# ============================================================

def sar_to_mercator(bounds, crs):

    return transform_bounds(
        crs,
        "EPSG:3857",
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top
    )


# ============================================================
# WEB MERCATOR → TILE COORDINATES
# ============================================================

def mercator_to_tile(x, y, zoom):

    origin = 20037508.342789244
    world = 40075016.685578488

    n = 2 ** zoom

    tx = int(
        math.floor(
            (x + origin) / (world / n)
        )
    )

    ty = int(
        math.floor(
            (origin - y) / (world / n)
        )
    )

    tx = max(0, min(tx, n - 1))
    ty = max(0, min(ty, n - 1))

    return tx, ty


# ============================================================
# DOWNLOAD ONE TILE
# ============================================================

def download_tile(x, y, zoom, path):

    url = GOOGLE_URL.format(
        x=x,
        y=y,
        z=zoom
    )

    try:

        r = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent":
                "Mozilla/5.0"
            }
        )

        if r.status_code != 200:

            return False, (
                f"{x},{y}: HTTP {r.status_code}"
            )

        # Make sure we actually received an image
        if len(r.content) < 1000:

            return False, (
                f"{x},{y}: suspiciously small response"
            )

        with open(path, "wb") as f:
            f.write(r.content)

        return True, None

    except Exception as e:

        return False, f"{x},{y}: {e}"


# ============================================================
# DOWNLOAD ALL TILES
# ============================================================

def download_tiles(
    xmin,
    xmax,
    ymin,
    ymax,
    zoom,
    tile_dir
):

    os.makedirs(
        tile_dir,
        exist_ok=True
    )

    tasks = []

    for y in range(ymin, ymax + 1):

        for x in range(xmin, xmax + 1):

            path = os.path.join(
                tile_dir,
                f"{zoom}_{x}_{y}.jpg"
            )

            tasks.append(
                (x, y, path)
            )

    total = len(tasks)

    print(
        f"\nGoogle tiles required: {total}"
    )

    downloaded = []
    failures = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                download_tile,
                x,
                y,
                zoom,
                path
            ): (x, y, path)

            for x, y, path in tasks
        }

        done = 0

        for future in as_completed(futures):

            x, y, path = futures[future]

            success, error = future.result()

            done += 1

            if success:

                downloaded.append(
                    (x, y, path)
                )

            else:

                failures.append(error)

            print(
                f"\rDownloading: "
                f"{done}/{total}",
                end=""
            )

    print()

    print(
        f"Successful: {len(downloaded)}"
    )

    print(
        f"Failed: {len(failures)}"
    )

    if failures:

        print("\nFirst failures:")

        for error in failures[:10]:

            print(" ", error)

        raise RuntimeError(
            "Some Google tiles failed. "
            "Stopping instead of creating black areas."
        )

    return downloaded


# ============================================================
# STITCH TILES WITH THEIR REAL GEOGRAPHIC EXTENT
# ============================================================

def stitch_tiles(
    tiles,
    xmin,
    xmax,
    ymin,
    ymax,
    zoom,
    output_path
):

    width = (
        (xmax - xmin + 1)
        * TILE_SIZE
    )

    height = (
        (ymax - ymin + 1)
        * TILE_SIZE
    )

    mosaic = Image.new(
        "RGB",
        (width, height)
    )

    for x, y, path in tiles:

        tile = Image.open(
            path
        ).convert("RGB")

        px = (
            (x - xmin)
            * TILE_SIZE
        )

        py = (
            (y - ymin)
            * TILE_SIZE
        )

        mosaic.paste(
            tile,
            (px, py)
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # These are the ACTUAL geographic bounds of the
    # complete Google tile mosaic.
    # --------------------------------------------------------

    origin = 20037508.342789244

    world = 40075016.685578488

    n = 2 ** zoom

    tile_world_size = world / n

    mosaic_min_x = (
        -origin
        + xmin * tile_world_size
    )

    mosaic_max_x = (
        -origin
        + (xmax + 1)
        * tile_world_size
    )

    mosaic_max_y = (
        origin
        - ymin * tile_world_size
    )

    mosaic_min_y = (
        origin
        - (ymax + 1)
        * tile_world_size
    )

    mosaic.save(
        output_path,
        quality=95
    )

    print(
        "\nActual Google mosaic bounds:"
    )

    print(
        mosaic_min_x,
        mosaic_min_y,
        mosaic_max_x,
        mosaic_max_y
    )

    return (
        mosaic,
        (
            mosaic_min_x,
            mosaic_min_y,
            mosaic_max_x,
            mosaic_max_y
        )
    )


# ============================================================
# SAVE REAL GOOGLE MOSAIC AS GEOTIFF
# ============================================================

def save_google_geotiff(
    image,
    bounds,
    output_path
):

    min_x, min_y, max_x, max_y = bounds

    width, height = image.size

    transform = from_bounds(
        min_x,
        min_y,
        max_x,
        max_y,
        width,
        height
    )

    rgb = np.asarray(
        image,
        dtype=np.uint8
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": transform,
        "compress": "deflate"
    }

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            rgb[:, :, 0],
            1
        )

        dst.write(
            rgb[:, :, 1],
            2
        )

        dst.write(
            rgb[:, :, 2],
            3
        )

    print(
        f"\nGoogle GeoTIFF saved:\n"
        f"{output_path}"
    )


# ============================================================
# CROP + REPROJECT TO EXACT SAR FOOTPRINT
# ============================================================

def create_sar_aligned_rgb(
    google_tif,
    sar_path,
    output_path
):

    print("\n========== ALIGNING RGB TO SAR ==========")

    with rasterio.open(sar_path) as sar:

        sar_crs = sar.crs
        sar_bounds = sar.bounds

        print("Target CRS:", sar_crs)
        print("Target SAR bounds:", sar_bounds)

        with rasterio.open(google_tif) as src:

            # ------------------------------------------------
            # Source RGB resolution is in meters because
            # Google mosaic is EPSG:3857.
            # ------------------------------------------------

            src_res_x = abs(src.res[0])
            src_res_y = abs(src.res[1])

            # ------------------------------------------------
            # Convert the source RGB resolution approximately
            # from meters/pixel to degrees/pixel.
            #
            # At the SAR latitude (~37.34 degrees):
            #   longitude degree is ~88-90 km
            #   latitude degree is ~111 km
            # ------------------------------------------------

            center_lat = (
                sar_bounds.bottom +
                sar_bounds.top
            ) / 2.0

            meters_per_degree_lat = 111320.0

            meters_per_degree_lon = (
                111320.0 *
                math.cos(
                    math.radians(center_lat)
                )
            )

            dst_res_x = (
                src_res_x /
                meters_per_degree_lon
            )

            dst_res_y = (
                src_res_y /
                meters_per_degree_lat
            )

            print(
                "RGB source resolution:",
                src_res_x,
                src_res_y,
                "m/pixel"
            )

            print(
                "RGB target resolution:",
                dst_res_x,
                dst_res_y,
                "degrees/pixel"
            )

            # ------------------------------------------------
            # Build output transform directly from the
            # EXACT SAR geographic footprint.
            # ------------------------------------------------

            dst_width = max(
                1,
                int(
                    math.ceil(
                        (
                            sar_bounds.right -
                            sar_bounds.left
                        ) / dst_res_x
                    )
                )
            )

            dst_height = max(
                1,
                int(
                    math.ceil(
                        (
                            sar_bounds.top -
                            sar_bounds.bottom
                        ) / dst_res_y
                    )
                )
            )

            dst_transform = from_bounds(
                sar_bounds.left,
                sar_bounds.bottom,
                sar_bounds.right,
                sar_bounds.top,
                dst_width,
                dst_height
            )

            print(
                "Output RGB size:",
                dst_width,
                "x",
                dst_height
            )

            # ------------------------------------------------
            # Output profile
            # ------------------------------------------------

            profile = src.profile.copy()

            profile.update(
                driver="GTiff",
                crs=sar_crs,
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
                count=3,
                dtype="uint8",
                compress="deflate"
            )

            # ------------------------------------------------
            # Reproject RGB → SAR CRS
            # ------------------------------------------------

            with rasterio.open(
                output_path,
                "w",
                **profile
            ) as dst:

                for band in range(1, 4):

                    reproject(
                        source=rasterio.band(
                            src,
                            band
                        ),

                        destination=rasterio.band(
                            dst,
                            band
                        ),

                        src_transform=src.transform,
                        src_crs=src.crs,

                        dst_transform=dst_transform,
                        dst_crs=sar_crs,

                        resampling=Resampling.bilinear,

                        src_nodata=0,
                        dst_nodata=0
                    )

    # --------------------------------------------------------
    # VERIFY FINAL FILE
    # --------------------------------------------------------

    with rasterio.open(
        output_path
    ) as result:

        print(
            "\n========== FINAL RGB =========="
        )

        print(
            "CRS:",
            result.crs
        )

        print(
            "Bounds:",
            result.bounds
        )

        print(
            "Size:",
            result.width,
            "x",
            result.height
        )

        print(
            "Resolution:",
            result.res
        )

        print(
            "Data type:",
            result.dtypes
        )
# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("       SAR-REFERENCED RGB EXTRACTION")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Read SAR
    # --------------------------------------------------------

    sar = get_sar_reference(
        SAR_PATH
    )

    print(
        "\n========== SAR =========="
    )

    print(
        "CRS:",
        sar["crs"]
    )

    print(
        "Bounds:",
        sar["bounds"]
    )

    print(
        "Size:",
        sar["width"],
        "x",
        sar["height"]
    )

    # --------------------------------------------------------
    # 2. SAR footprint → Web Mercator
    # --------------------------------------------------------

    sar_mercator = sar_to_mercator(
        sar["bounds"],
        sar["crs"]
    )

    print(
        "\nSAR footprint in EPSG:3857:"
    )

    print(
        sar_mercator
    )

    # --------------------------------------------------------
    # 3. Determine Google tiles
    # --------------------------------------------------------

    min_x, min_y, max_x, max_y = (
        sar_mercator
    )

    xmin, ymax = mercator_to_tile(
        min_x,
        min_y,
        ZOOM
    )

    xmax, ymin = mercator_to_tile(
        max_x,
        max_y,
        ZOOM
    )

    print(
        "\n========== TILE RANGE =========="
    )

    print(
        "X:",
        xmin,
        "to",
        xmax
    )

    print(
        "Y:",
        ymin,
        "to",
        ymax
    )

    # --------------------------------------------------------
    # 4. Download
    # --------------------------------------------------------

    tile_dir = os.path.join(
        OUTPUT_DIR,
        "google_tiles"
    )

    tiles = download_tiles(
        xmin,
        xmax,
        ymin,
        ymax,
        ZOOM,
        tile_dir
    )

    # --------------------------------------------------------
    # 5. Stitch using ACTUAL tile bounds
    # --------------------------------------------------------

    mosaic_path = os.path.join(
        OUTPUT_DIR,
        "google_mosaic.jpg"
    )

    mosaic, actual_bounds = stitch_tiles(
        tiles,
        xmin,
        xmax,
        ymin,
        ymax,
        ZOOM,
        mosaic_path
    )

    # --------------------------------------------------------
    # 6. Save correctly georeferenced Google mosaic
    # --------------------------------------------------------

    google_tif = os.path.join(
        OUTPUT_DIR,
        "google_mosaic_3857.tif"
    )

    save_google_geotiff(
        mosaic,
        actual_bounds,
        google_tif
    )

    # --------------------------------------------------------
    # 7. Crop + reproject to SAR footprint
    # --------------------------------------------------------

    final_rgb = os.path.join(
        OUTPUT_DIR,
        "rgb_sar_aligned.tif"
    )

    create_sar_aligned_rgb(
        google_tif,
        SAR_PATH,
        final_rgb
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "                  COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nFinal RGB:"
    )

    print(
        final_rgb
    )


if __name__ == "__main__":
    main()