#!/usr/bin/env python3
"""
GIS Data Blender
----------------
A Python application to download ortho imagery and SRTM elevation data for a specified
lat/long and area, stitch them together, and generate a 3D mesh in OBJ and FBX formats.

Features:
1. Downloads ortho imagery (Google Maps, Mapbox, or OSM) and stitches them using VRTs.
2. Downloads corresponding Mapzen/SRTM elevation tiles.
3. Combines both into a single 4-band GeoTIFF (R, G, B, Elevation).
4. Generates a textured 3D mesh in OBJ (+ MTL) and FBX formats.
5. Keeps texture georeferencing intact via GeoTIFF tags and World files (.tfw, .pgw).
6. Embeds/provides lat/long metadata inside FBX properties, OBJ headers, and JSON.
7. Supports both CLI arguments and a step-by-step interactive wizard.
"""

import os
import sys
import math
import json
import shutil
import argparse
import subprocess
import requests
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter

# Set Dotnet Globalization Invariant mode to prevent ICU errors in Aspose.3D
os.environ["DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"] = "1"

# Import spatial and 3D libraries
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except ImportError:
    print("Error: 'rasterio' is not installed. Please install it using: pip install rasterio")
    sys.exit(1)

try:
    import trimesh
except ImportError:
    print("Error: 'trimesh' is not installed. Please install it using: pip install trimesh")
    sys.exit(1)

try:
    import aspose.threed as a3d
    from aspose.threed import Scene, FileFormat
    from aspose.threed.formats import FbxSaveOptions
except ImportError:
    print("Warning: 'aspose-3d' is not installed. FBX conversion will be skipped.")
    a3d = None

# Web Mercator projection constant
M = 20037508.342789244

def latlon_to_mercator(lat, lon):
    """Convert latitude/longitude (WGS84) to Web Mercator (EPSG:3857) meters."""
    x = lon * M / 180.0
    lat_rad = math.radians(lat)
    y = math.log(math.tan((math.pi / 4.0) + (lat_rad / 2.0))) * M / math.pi
    return x, y

def mercator_to_latlon(x, y):
    """Convert Web Mercator (EPSG:3857) meters to latitude/longitude (WGS84)."""
    lon = x * 180.0 / M
    lat_rad = 2.0 * math.atan(math.exp(y * math.pi / M)) - (math.pi / 2.0)
    lat = math.degrees(lat_rad)
    return lat, lon

def get_tile_bounds(lat, lon, size_km, zoom):
    """
    Calculate the bounding box in Mercator meters and the corresponding
    tile coordinates range [xmin, xmax, ymin, ymax] for a given zoom level.
    """
    cx, cy = latlon_to_mercator(lat, lon)
    lat_rad = math.radians(lat)
    scale = 1.0 / math.cos(lat_rad)
    
    # Calculate half-size in Mercator meters, adjusted for latitude distortion
    dx = (size_km * 1000.0 / 2.0) * scale
    dy = (size_km * 1000.0 / 2.0) * scale
    
    min_x = cx - dx
    max_x = cx + dx
    min_y = cy - dy
    max_y = cy + dy
    
    tile_size_m = (2.0 * M) / (2**zoom)
    
    xmin = int(math.floor((min_x + M) / tile_size_m))
    xmax = int(math.floor((max_x + M) / tile_size_m))
    ymin = int(math.floor((M - max_y) / tile_size_m))
    ymax = int(math.floor((M - min_y) / tile_size_m))
    
    # Clip tile ranges to valid Mercator limits
    max_tiles = 2**zoom
    xmin = max(0, min(xmin, max_tiles - 1))
    xmax = max(0, min(xmax, max_tiles - 1))
    ymin = max(0, min(ymin, max_tiles - 1))
    ymax = max(0, min(ymax, max_tiles - 1))
    
    return (min_x, min_y, max_x, max_y), (xmin, xmax, ymin, ymax)

def calculate_tri(dem):
    """Calculate Terrain Ruggedness Index (TRI) using the 8-neighbor RMS method."""
    shifts = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    diff_sq_sum = np.zeros_like(dem, dtype=np.float32)
    for dy, dx in shifts:
        shifted = np.roll(np.roll(dem, dy, axis=0), dx, axis=1)
        diff_sq_sum += (dem - shifted) ** 2
    tri = np.sqrt(diff_sq_sum / 8.0)
    # Clear borders to avoid wrap-around artifacts
    tri[0, :] = 0; tri[-1, :] = 0; tri[:, 0] = 0; tri[:, -1] = 0
    return tri

def calculate_slope(dem, res_x, res_y):
    """Calculate slope in degrees using central differences."""
    gy, gx = np.gradient(dem)
    gy /= res_y
    gx /= res_x
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    return np.degrees(slope_rad)

def compute_autocorrelation_profile(dem):
    """Compute the average 1D autocorrelation profile along rows and columns."""
    def autocorrelation_1d(signal):
        n = len(signal)
        mean = np.mean(signal)
        var = np.var(signal)
        if var < 1e-5:
            return np.ones(n)
        xp = signal - mean
        r = np.correlate(xp, xp, mode='full')
        r = r[n-1:] / (var * np.arange(n, 0, -1))
        return r

    h, w = dem.shape
    step_r = max(1, h // 20)
    step_c = max(1, w // 20)
    profiles = []
    
    for r in range(0, h, step_r):
        profiles.append(autocorrelation_1d(dem[r, :]))
    for c in range(0, w, step_c):
        profiles.append(autocorrelation_1d(dem[:, c]))
        
    max_len = max(len(p) for p in profiles)
    padded_profiles = []
    for p in profiles:
        if len(p) < max_len:
            padded_profiles.append(np.pad(p, (0, max_len - len(p)), mode='constant', constant_values=0))
        else:
            padded_profiles.append(p)
            
    return np.mean(padded_profiles, axis=0)

def calculate_autocorrelation_length(dem, res):
    """Calculate the profile autocorrelation length (lag where correlation drops below 1/e)."""
    profile = compute_autocorrelation_profile(dem)
    idx = np.where(profile < 0.368)[0]
    if idx.size > 0:
        return float(idx[0] * res)
    return float(len(profile) * res)

def calculate_entropy(dem):
    """Calculate the Shannon 2D Entropy of the DEM using 256 elevation bins."""
    d_min = np.min(dem)
    d_max = np.max(dem)
    if d_max - d_min < 1e-5:
        return 0.0
    normalized = ((dem - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    hist, _ = np.histogram(normalized, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def calculate_local_relief(dem, window_size_m, res):
    """Calculate local relief (difference between max and min) in a moving window."""
    w_size = int(round(window_size_m / res))
    if w_size % 2 == 0:
        w_size += 1
    w_size = max(3, w_size)
    local_max = maximum_filter(dem, size=w_size)
    local_min = minimum_filter(dem, size=w_size)
    return local_max - local_min

def calculate_curvature(dem, res_x, res_y):
    """Calculate curvature (Laplacian of elevation) using second derivatives."""
    gy, gx = np.gradient(dem)
    _, gxx = np.gradient(gx)
    gyy, _ = np.gradient(gy)
    gxx /= (res_x ** 2)
    gyy /= (res_y ** 2)
    return gxx + gyy

def calculate_ups(dem):
    """Calculate the Unique Profile Score (UPS) using normalized cross-correlation of sliding patches."""
    h, w = dem.shape
    target_h, target_w = 50, 50
    scale_h = max(1, h // target_h)
    scale_w = max(1, w // target_w)
    downsampled = dem[::scale_h, ::scale_w]
    
    dh, dw = downsampled.shape
    if dh < 5 or dw < 5:
        return np.ones_like(dem, dtype=np.float32)
        
    from numpy.lib.stride_tricks import sliding_window_view
    try:
        patches = sliding_window_view(downsampled, (5, 5))
    except ValueError:
        return np.ones_like(dem, dtype=np.float32)
        
    H_out, W_out = patches.shape[0], patches.shape[1]
    flat_patches = patches.reshape(-1, 25)
    
    means = flat_patches.mean(axis=1, keepdims=True)
    stds = flat_patches.std(axis=1, keepdims=True)
    stds[stds < 1e-5] = 1.0
    std_patches = (flat_patches - means) / stds
    
    corr_matrix = np.dot(std_patches, std_patches.T) / 25.0
    
    N = H_out * W_out
    mask = np.zeros((N, N), dtype=bool)
    for y in range(H_out):
        for x in range(W_out):
            idx_i = y * W_out + x
            ymin, ymax = max(0, y - 2), min(H_out, y + 3)
            xmin, xmax = max(0, x - 2), min(W_out, x + 3)
            for yn in range(ymin, ymax):
                for xn in range(xmin, xmax):
                    idx_j = yn * W_out + xn
                    mask[idx_i, idx_j] = True
                    
    masked_corr = corr_matrix.copy()
    masked_corr[mask] = -2.0
    
    max_corr = np.max(masked_corr, axis=1)
    unique_score = 1.0 - max_corr
    unique_score = np.clip(unique_score, 0.0, 1.0)
    
    ups_downsampled = unique_score.reshape(H_out, W_out)
    
    from scipy.ndimage import zoom
    zoom_factors = (h / H_out, w / W_out)
    ups_full = zoom(ups_downsampled, zoom_factors, order=1)
    
    if ups_full.shape[0] != h or ups_full.shape[1] != w:
        pad_h = max(0, h - ups_full.shape[0])
        pad_w = max(0, w - ups_full.shape[1])
        ups_full = np.pad(ups_full, ((0, pad_h), (0, pad_w)), mode='edge')[:h, :w]
        
    return np.clip(ups_full, 0.0, 1.0)

def calculate_tsi(std_dev, tri_mean, entropy, auto_len, ups_mean):
    """Compute the TERCOM Suitability Index (TSI) from 0 to 100 based on terrain stats."""
    n_std = 1.0 - math.exp(-std_dev / 30.0)
    n_tri = 1.0 - math.exp(-tri_mean / 5.0)
    n_ent = min(1.0, max(0.0, entropy / 8.0))
    n_auto = math.exp(-auto_len / 1000.0)
    n_ups = min(1.0, max(0.0, ups_mean))
    
    # Sub-component weights
    w_std, w_tri, w_auto, w_ups, w_ent = 0.25, 0.20, 0.20, 0.20, 0.15
    tsi = 100.0 * (w_std * n_std + w_tri * n_tri + w_auto * n_auto + w_ups * n_ups + w_ent * n_ent)
    return min(100.0, max(0.0, tsi)), (n_std, n_tri, n_ent, n_auto, n_ups)

def generate_terrain_dashboard(dem, slope, tri, relief_1k, curvature, ups, slope_dist, curvature_dist, autocorrelation_profile, tsi, sub_scores, output_path, lat, lon, size_km):
    """Generate a 9-panel PNG dashboard visualization summarizing the advanced terrain analysis."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    plt.suptitle(f"Terrain Analysis Report - Lat={lat:.5f}, Lon={lon:.5f} ({size_km:.1f}x{size_km:.1f} km)", fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Elevation Map
    im1 = axes[0, 0].imshow(dem, cmap='terrain')
    axes[0, 0].set_title("1. Elevation Map (meters)", fontsize=12, fontweight='bold')
    fig.colorbar(im1, ax=axes[0, 0])
    
    # 2. Slope Map
    im2 = axes[0, 1].imshow(slope, cmap='magma')
    axes[0, 1].set_title("2. Slope Map (degrees)", fontsize=12, fontweight='bold')
    fig.colorbar(im2, ax=axes[0, 1])
    
    # 3. TRI Map
    im3 = axes[0, 2].imshow(tri, cmap='viridis')
    axes[0, 2].set_title("3. Terrain Ruggedness Index (TRI)", fontsize=12, fontweight='bold')
    fig.colorbar(im3, ax=axes[0, 2])
    
    # 4. Local Relief Map
    im4 = axes[1, 0].imshow(relief_1k, cmap='plasma')
    axes[1, 0].set_title("4. Local Relief (1km Window)", fontsize=12, fontweight='bold')
    fig.colorbar(im4, ax=axes[1, 0])
    
    # 5. Curvature Map
    v_ext = max(0.001, float(np.percentile(np.abs(curvature), 95)))
    im5 = axes[1, 1].imshow(curvature, cmap='coolwarm', vmin=-v_ext, vmax=v_ext)
    axes[1, 1].set_title("5. Curvature Map (Laplacian)", fontsize=12, fontweight='bold')
    fig.colorbar(im5, ax=axes[1, 1])
    
    # 6. UPS Map
    im6 = axes[1, 2].imshow(ups, cmap='inferno', vmin=0, vmax=1)
    axes[1, 2].set_title("6. Unique Profile Score (UPS)", fontsize=12, fontweight='bold')
    fig.colorbar(im6, ax=axes[1, 2])
    
    # 7. Gradient Magnitude Histogram
    gy, gx = np.gradient(dem)
    grad_mag = np.sqrt(gx**2 + gy**2)
    axes[2, 0].hist(grad_mag.flatten(), bins=50, color='royalblue', alpha=0.7, edgecolor='black')
    axes[2, 0].set_title("7. Gradient Magnitude Histogram (rise/run)", fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel("Gradient Magnitude")
    axes[2, 0].set_ylabel("Pixel Count")
    
    # 8. Autocorrelation Profile
    lags = np.arange(len(autocorrelation_profile)) * (size_km * 1000.0 / len(autocorrelation_profile))
    axes[2, 1].plot(lags, autocorrelation_profile, color='darkgreen', linewidth=2)
    axes[2, 1].axhline(0.368, color='red', linestyle='--', label='1/e Threshold')
    axes[2, 1].set_title("8. Autocorrelation Profile", fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel("Lag Distance (meters)")
    axes[2, 1].set_ylabel("Correlation Coefficient")
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    # 9. Summary & TSI Panel
    axes[2, 2].axis('off')
    entropy_val = calculate_entropy(dem)
    
    # Determine classification recommendation
    std_dev = np.std(dem)
    max_val = np.max(dem)
    p95 = np.percentile(dem, 95)
    p05 = np.percentile(dem, 5)
    base_range = p95 - p05
    top_range = max_val - p95
    peakiness = top_range / base_range if base_range > 0.5 else top_range
    
    if std_dev < 1.5:
        recommendation = "Extremely Flat (poor for TERCOM)"
    elif peakiness > 1.2:
        if base_range < 5.0:
            recommendation = "Single Isolated Peak / Outlier"
        else:
            recommendation = "Prominent Peak / Localized Outlier"
    else:
        recommendation = "Varying Altitude Region (Ideal!)"
        
    summary_text = (
        f"=========================================\n"
        f"          TERRAIN SUMMARY REPORT         \n"
        f"=========================================\n"
        f"Min / Max Elev  : {np.min(dem):.1f} / {np.max(dem):.1f} m\n"
        f"Elevation Range : {np.max(dem) - np.min(dem):.1f} m\n"
        f"Elevation StdDev: {std_dev:.2f} m\n"
        f"Mean Slope      : {slope_dist['mean']:.2f}°\n"
        f"Slope Std Dev   : {slope_dist['std']:.2f}°\n"
        f"Slope 95th %ile : {slope_dist['p95']:.2f}°\n"
        f"Mean TRI        : {np.mean(tri):.2f} m\n"
        f"Mean Relief (1k): {np.mean(relief_1k):.2f} m\n"
        f"Entropy (2D)    : {entropy_val:.2f} bits\n"
        f"Convex Area %   : {curvature_dist['convex_pct']:.1f}%\n"
        f"Concave Area %  : {curvature_dist['concave_pct']:.1f}%\n"
        f"-----------------------------------------\n"
        f"TERCOM SUITABILITY INDEX (TSI)\n"
        f"SCORE: {tsi:.1f} / 100.0\n"
        f"-----------------------------------------\n"
        f"Sub-scores (normalized 0.0-1.0):\n"
        f"  - Elevation Var  : {sub_scores[0]:.2f}\n"
        f"  - Ruggedness (TRI): {sub_scores[1]:.2f}\n"
        f"  - Autocorrelation: {sub_scores[3]:.2f}\n"
        f"  - Profile Unique : {sub_scores[4]:.2f}\n"
        f"  - Shannon Entropy: {sub_scores[2]:.2f}\n"
        f"Classification  : {recommendation}\n"
        f"========================================="
    )
    
    axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes, fontsize=9,
                    family='monospace', verticalalignment='top', bbox=dict(boxstyle='round,pad=1', facecolor='whitesmoke', edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_sar_reference(sar_path):
    """Read SAR georeferencing information to use as the RGB spatial reference."""
    with rasterio.open(sar_path) as sar:
        return {
            "crs": sar.crs,
            "bounds": sar.bounds,
            "width": sar.width,
            "height": sar.height,
            "transform": sar.transform,
            "res": sar.res,
        }

def sar_bounds_to_mercator(sar_bounds, sar_crs):
    """Convert SAR bounds to EPSG:3857 for Google tile selection."""

    from rasterio.warp import transform_bounds

    min_x, min_y, max_x, max_y = transform_bounds(
        sar_crs,
        "EPSG:3857",
        sar_bounds.left,
        sar_bounds.bottom,
        sar_bounds.right,
        sar_bounds.top
    )

    return min_x, min_y, max_x, max_y




def check_gdal():
    """Verify that gdalwarp is available on the system."""
    if not shutil.which("gdalwarp"):
        print("Error: 'gdalwarp' CLI tool is not found. Please install GDAL on your system.")
        sys.exit(1)

def download_tile(url, filepath):
    """Download a single tile image using requests with retries."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    last_error = "Unknown error"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'wb') as f:
                    f.write(r.content)
                return True, None
            elif r.status_code == 404:
                return False, "HTTP 404 Not Found"
            else:
                last_error = f"HTTP status code {r.status_code}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
    return False, last_error

def download_tiles_parallel(tile_tasks, max_workers=10, desc="Downloading"):
    """Download a list of tiles in parallel and display a text progress bar."""
    total = len(tile_tasks)
    downloaded = 0
    results = []
    errors = {}
    
    print(f"{desc}: 0% [|] 0/{total}", end="", flush=True)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_tile, url, path): (x, y, path) for url, path, x, y in tile_tasks}
        
        for future in as_completed(futures):
            x, y, path = futures[future]
            success, error_msg = future.result()
            downloaded += 1
            if success:
                results.append({'x': x, 'y': y, 'path': path})
            else:
                errors[error_msg] = errors.get(error_msg, 0) + 1
            
            percent = int((downloaded / total) * 100)
            bar = "#" * (percent // 5) + "-" * (20 - (percent // 5))
            print(f"\r{desc}: {percent}% [{bar}] {downloaded}/{total}", end="", flush=True)
            
    print()  # new line
    return results, errors

def generate_vrt(vrt_path, tile_width, tile_height, tile_xmin, tile_xmax, tile_ymin, tile_ymax, zoom, tiles, is_dem=False):
    """Generate a GDAL VRT file referencing downloaded tiles at their spatial coordinates."""
    nx = tile_xmax - tile_xmin + 1
    ny = tile_ymax - tile_ymin + 1
    total_w = nx * tile_width
    total_h = ny * tile_height
    
    tile_size_m = (2.0 * M) / (2**zoom)
    global_xmin = -M + tile_xmin * tile_size_m
    global_ymax = M - tile_ymin * tile_size_m
    
    res_x = tile_size_m / tile_width
    res_y = tile_size_m / tile_height
    
    lines = []
    lines.append(f'<VRTDataset rasterXSize="{total_w}" rasterYSize="{total_h}">')
    
    srs_wkt = (
        'PROJCS["WGS 84 / Pseudo-Mercator",'
        'GEOGCS["WGS 84",'
        'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Mercator_1SP"],'
        'PARAMETER["central_meridian",0],'
        'PARAMETER["scale_factor",1],'
        'PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],'
        'UNIT["metre",1],'
        'AXIS["Easting",EAST],'
        'AXIS["Northing",NORTH],'
        'AUTHORITY["EPSG","3857"]]'
    )
    lines.append(f'  <SRS dataAxisToSRSAxisMapping="1,2">{srs_wkt}</SRS>')
    lines.append(f'  <GeoTransform>{global_xmin:.12f}, {res_x:.12f}, 0.0, {global_ymax:.12f}, 0.0, {-res_y:.12f}</GeoTransform>')
    
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    
    if is_dem:
        lines.append('  <VRTRasterBand dataType="Int16" band="1">')
        for tile in tiles:
            tx = tile['x']
            ty = tile['y']
            tpath = tile['path']
            rel_path = os.path.relpath(os.path.abspath(tpath), vrt_dir)
            dst_x = (tx - tile_xmin) * tile_width
            dst_y = (ty - tile_ymin) * tile_height
            
            lines.append('    <SimpleSource>')
            lines.append(f'      <SourceFilename relativeToVRT="1">{rel_path}</SourceFilename>')
            lines.append('      <SourceBand>1</SourceBand>')
            lines.append(f'      <SrcRect xOff="0" yOff="0" xSize="{tile_width}" ySize="{tile_height}"/>')
            lines.append(f'      <DstRect xOff="{dst_x}" yOff="{dst_y}" xSize="{tile_width}" ySize="{tile_height}"/>')
            lines.append('    </SimpleSource>')
        lines.append('  </VRTRasterBand>')
    else:
        band_names = ['Red', 'Green', 'Blue']
        for b in range(1, 4):
            lines.append(f'  <VRTRasterBand dataType="Byte" band="{b}">')
            lines.append(f'    <ColorInterp>{band_names[b-1]}</ColorInterp>')
            for tile in tiles:
                tx = tile['x']
                ty = tile['y']
                tpath = tile['path']
                rel_path = os.path.relpath(os.path.abspath(tpath), vrt_dir)
                dst_x = (tx - tile_xmin) * tile_width
                dst_y = (ty - tile_ymin) * tile_height
                
                lines.append('    <SimpleSource>')
                lines.append(f'      <SourceFilename relativeToVRT="1">{rel_path}</SourceFilename>')
                lines.append(f'      <SourceBand>{b}</SourceBand>')
                lines.append(f'      <SrcRect xOff="0" yOff="0" xSize="{tile_width}" ySize="{tile_height}"/>')
                lines.append(f'      <DstRect xOff="{dst_x}" yOff="{dst_y}" xSize="{tile_width}" ySize="{tile_height}"/>')
                lines.append('    </SimpleSource>')
            lines.append('  </VRTRasterBand>')
            
    lines.append('</VRTDataset>')
    
    with open(vrt_path, 'w') as f:
        f.write('\n'.join(lines))

def warp_raster(vrt_path, tif_path, bbox, res):
    """Crop and warp a VRT dataset to the exact bounding box and resolution using gdalwarp."""
    min_x, min_y, max_x, max_y = bbox
    cmd = [
        'gdalwarp',
        '-overwrite',
        '-te', str(min_x), str(min_y), str(max_x), str(max_y),
        '-tr', str(res), str(res),
        '-r', 'bilinear',
        '-t_srs', 'EPSG:3857',
        vrt_path,
        tif_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdalwarp failed: {result.stderr}")

def write_world_file(transform, filepath):
    """Write standard 6-line GIS World File (.tfw or .pgw) representing the affine transform."""
    a = transform.a
    b = transform.b
    d = transform.d
    e = transform.e
    c = transform.c
    f = transform.f
    
    # Shift upper-left coordinate from pixel outer corner to pixel center
    x_center = c + a / 2.0
    y_center = f + e / 2.0
    
    content = f"{a:.12f}\n{b:.12f}\n{d:.12f}\n{e:.12f}\n{x_center:.12f}\n{y_center:.12f}\n"
    with open(filepath, 'w') as f_out:
        f_out.write(content)

def build_3d_mesh(final_tif_path, bbox, mesh_res, z_scale, output_dir, lat, lon, size_km, zoom):
    """Generate 3D OBJ & FBX models from the combined 4-band GeoTIFF and embed/provide georeferencing metadata."""
    print("Building 3D Mesh...")
    
    obj_path = os.path.join(output_dir, "map.obj")
    mtl_path = os.path.join(output_dir, "map.mtl")
    fbx_path = os.path.join(output_dir, "map.fbx")
    texture_png_path = os.path.join(output_dir, "map_texture.png")
    metadata_json_path = os.path.join(output_dir, "map_metadata.json")
    
    min_x, min_y, max_x, max_y = bbox
    width_m = max_x - min_x
    height_m = max_y - min_y
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    
    # 1. Read and resample data to the target mesh resolution
    with rasterio.open(final_tif_path) as src:
        data = src.read(
            out_shape=(4, mesh_res, mesh_res),
            resampling=Resampling.bilinear
        )
        # Read full resolution image for texturing
        r_full = src.read(1)
        g_full = src.read(2)
        b_full = src.read(3)
        img_transform = src.transform
        # Read full resolution elevation for accurate min/max measurements
        elev_full = src.read(4)
        
    elev = data[3]
    
    # Measure min and max heights from full resolution elevation data
    # Filter out common nodata value (-32768) and extreme values
    valid_elev = elev_full[(elev_full != -32768) & (elev_full > -1000) & (~np.isnan(elev_full))]
    if valid_elev.size > 0:
        # Filter outliers using Tukey's IQR Method (k=3)
        q25, q75 = np.percentile(valid_elev, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 3.0 * iqr
        upper_bound = q75 + 3.0 * iqr
        filtered_elev = valid_elev[(valid_elev >= lower_bound) & (valid_elev <= upper_bound)]
        if filtered_elev.size == 0:
            filtered_elev = valid_elev
            
        min_elev = float(np.min(filtered_elev))
        max_elev = float(np.max(filtered_elev))
        p05 = float(np.percentile(filtered_elev, 5))
        p25 = float(np.percentile(filtered_elev, 25))
        p50 = float(np.percentile(filtered_elev, 50))
        p75 = float(np.percentile(filtered_elev, 75))
        p95 = float(np.percentile(filtered_elev, 95))
        mean_elev = float(np.mean(filtered_elev))
        std_dev = float(np.std(filtered_elev))
        
        base_range = p95 - p05
        top_range = max_elev - p95
        peakiness = top_range / base_range if base_range > 0.5 else top_range
        
        # Clean 2D DEM for spatial calculations
        median_val = p50
        cleaned_dem = elev_full.copy().astype(np.float32)
        cleaned_dem[(cleaned_dem == -32768) | (cleaned_dem <= -1000) | (np.isnan(cleaned_dem)) | (cleaned_dem < lower_bound) | (cleaned_dem > upper_bound)] = median_val
    else:
        min_elev = max_elev = p05 = p25 = p50 = p75 = p95 = mean_elev = std_dev = peakiness = 0.0
        cleaned_dem = elev_full.copy().astype(np.float32)

    # Compute Advanced Terrain Metrics
    tri_map = calculate_tri(cleaned_dem)
    pixel_size = width_m / src.width
    
    slope_map = calculate_slope(cleaned_dem, pixel_size, pixel_size)
    curvature_map = calculate_curvature(cleaned_dem, pixel_size, pixel_size)
    relief_1k = calculate_local_relief(cleaned_dem, 1000.0, pixel_size)
    relief_500m = calculate_local_relief(cleaned_dem, 500.0, pixel_size)
    ups_map = calculate_ups(cleaned_dem)
    
    tri_mean = float(np.mean(tri_map))
    slope_dist = {
        "mean": float(np.mean(slope_map)),
        "std": float(np.std(slope_map)),
        "p95": float(np.percentile(slope_map, 95))
    }
    curvature_std = float(np.std(curvature_map))
    convex_pct = float(np.sum(curvature_map < 0) / curvature_map.size * 100)
    concave_pct = float(np.sum(curvature_map > 0) / curvature_map.size * 100)
    curvature_dist = {
        "mean": float(np.mean(curvature_map)),
        "std": curvature_std,
        "convex_pct": convex_pct,
        "concave_pct": concave_pct
    }
    mean_relief_1k = float(np.mean(relief_1k))
    mean_relief_500m = float(np.mean(relief_500m))
    entropy_val = calculate_entropy(cleaned_dem)
    auto_len = calculate_autocorrelation_length(cleaned_dem, pixel_size)
    ups_mean = float(np.mean(ups_map))
    
    # TSI Calculation
    tsi_score, sub_scores = calculate_tsi(std_dev, tri_mean, entropy_val, auto_len, ups_mean)
    
    # Save the high-res texture image (PNG)
    rgb_full = np.stack([r_full, g_full, b_full], axis=-1).astype('uint8')
    tex_img = Image.fromarray(rgb_full)
    tex_img.save(texture_png_path)
    
    # Write world file for PNG texture (.pgw)
    write_world_file(img_transform, os.path.join(output_dir, "map_texture.pgw"))
    
    # Write world file for TIF texture (.tfw)
    write_world_file(img_transform, os.path.join(output_dir, "map_texture.tfw"))
    
    # Calculate Lat/Long bounding box
    min_lat, min_lon = mercator_to_latlon(min_x, min_y)
    max_lat, max_lon = mercator_to_latlon(max_x, max_y)
    
    # Generate Visual Report
    report_png_path = os.path.join(output_dir, "terrain_analysis_report.png")
    autocorr_profile = compute_autocorrelation_profile(cleaned_dem)
    print("Generating Terrain Analysis Visual Report...")
    generate_terrain_dashboard(
        cleaned_dem, slope_map, tri_map, relief_1k, curvature_map, ups_map,
        slope_dist, curvature_dist, autocorr_profile, tsi_score, sub_scores,
        report_png_path, lat, lon, size_km
    )
    print(f"Terrain Analysis Visual Report saved to: {os.path.abspath(report_png_path)}")
    
    # Print profile summary to console
    print("\n=========================================")
    print("          ELEVATION PROFILE ANALYSIS     ")
    print("=========================================")
    print(f"Min Elevation      : {min_elev:.2f} meters")
    print(f"Max Elevation      : {max_elev:.2f} meters")
    print(f"Elevation Range    : {max_elev - min_elev:.2f} meters")
    print(f"Mean Elevation     : {mean_elev:.2f} meters")
    print(f"Median (50th)      : {p50:.2f} meters")
    print(f"Std Deviation      : {std_dev:.2f} meters")
    print("-----------------------------------------")
    print("Percentile Distribution (meters):")
    print(f"  5th %ile : {p05:.2f} m")
    print(f" 25th %ile : {p25:.2f} m")
    print(f" 75th %ile : {p75:.2f} m")
    print(f" 95th %ile : {p95:.2f} m")
    print("-----------------------------------------")
    print("Advanced Terrain Metrics:")
    print(f"  Mean TRI (Ruggedness)      : {tri_mean:.2f} meters")
    print(f"  Slope (Mean/Std/95th)      : {slope_dist['mean']:.2f}° / {slope_dist['std']:.2f}° / {slope_dist['p95']:.2f}°")
    print(f"  Profile Autocorr Length    : {auto_len:.2f} meters")
    print(f"  2D Shannon Entropy         : {entropy_val:.2f} bits")
    print(f"  Mean Relief (500m / 1km)   : {mean_relief_500m:.2f} m / {mean_relief_1k:.2f} m")
    print(f"  Curvature (Mean/Std)       : {curvature_dist['mean']:.6f} / {curvature_dist['std']:.6f}")
    print(f"  Convex / Concave Terrain   : {convex_pct:.1f}% / {concave_pct:.1f}%")
    print(f"  Unique Profile Score (UPS) : {ups_mean:.4f}")
    print("-----------------------------------------")
    print("TERCOM NAV SUITABILITY:")
    print(f"  TSI Score                  : {tsi_score:.1f} / 100.0")
    
    # Terrain Profile Recommendation
    if std_dev < 1.5:
        recommendation = "Extremely Flat (poor for TERCOM)"
    elif peakiness > 1.2:
        if base_range < 5.0:
            recommendation = "Single Isolated Peak / Outlier (poor for TERCOM)"
        else:
            recommendation = "Prominent Peak / Localized Outlier (moderate suitability)"
    else:
        recommendation = "Varying Altitude Region (Ideal/High Suitability for TERCOM!)"
        
    print(f"  Terrain Classification     : {recommendation}")
    print("=========================================\n")
    
    # Build complete metadata structure
    metadata = {
        "center_lat": lat,
        "center_lon": lon,
        "size_km": size_km,
        "zoom_level": zoom,
        "spatial_reference": "EPSG:3857 (Web Mercator)",
        "bounds_mercator": {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y
        },
        "bounds_latlon": {
            "min_lat": min_lat,
            "min_lon": min_lon,
            "max_lat": max_lat,
            "max_lon": max_lon
        },
        "mesh_offset_mercator": {
            "x_offset": cx,
            "y_offset": cy
        },
        "z_scale": z_scale,
        "pixel_size_meters": pixel_size,
        "mesh_resolution": mesh_res,
        "elevation_stats": {
            "min_elevation_meters": min_elev,
            "max_elevation_meters": max_elev,
            "elevation_range_meters": max_elev - min_elev,
            "elevation_std_dev": std_dev
        },
        "advanced_terrain_metrics": {
            "terrain_ruggedness_index_tri_mean": tri_mean,
            "slope_degrees": slope_dist,
            "profile_autocorrelation_length_meters": auto_len,
            "shannon_entropy_2d_bits": entropy_val,
            "mean_local_relief_500m_meters": mean_relief_500m,
            "mean_local_relief_1km_meters": mean_relief_1k,
            "curvature": curvature_dist,
            "unique_profile_score_ups_mean": ups_mean,
            "tercom_suitability_index_tsi": tsi_score
        },
        "reconstruct_instructions": (
            "To resolve actual Web Mercator coordinate from vertex coordinate (vx, vy, vz): "
            "Mercator X = vx + x_offset, Mercator Y = vy + y_offset, Elevation Z = vz / z_scale. "
            "Then project back from EPSG:3857 to EPSG:4326 for Latitude/Longitude."
        )
    }
    
    # Save metadata JSON file
    with open(metadata_json_path, 'w') as f_meta:
        json.dump(metadata, f_meta, indent=2)
    print(f"Metadata JSON saved to: {metadata_json_path}")
    
    # 2. Build vertices, UV coordinates, and face topology
    vertices = []
    uvs = []
    
    for r in range(mesh_res):
        for c in range(mesh_res):
            # Centered around (0, 0)
            vx = (c / (mesh_res - 1) - 0.5) * width_m
            vy = (0.5 - r / (mesh_res - 1)) * height_m
            vz = elev[r, c] * z_scale
            vertices.append([vx, vy, vz])
            
            u = c / (mesh_res - 1)
            v = 1.0 - r / (mesh_res - 1)
            uvs.append([u, v])
            
    faces = []
    for r in range(mesh_res - 1):
        for c in range(mesh_res - 1):
            v00 = r * mesh_res + c
            v01 = v00 + 1
            v10 = (r + 1) * mesh_res + c
            v11 = v10 + 1
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])
            
    # 3. Create mesh in trimesh and apply textures
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(uvs), image=tex_img)
    
    # Export OBJ
    print("Saving OBJ mesh...")
    mesh.export(obj_path)
    
    # Prepend georeferencing metadata as comments at the top of the OBJ file
    with open(obj_path, 'r') as f:
        original_obj = f.read()
    
    comments = (
        "# GIS Georeference Metadata\n"
        f"# Latitude: {lat:.6f}\n"
        f"# Longitude: {lon:.6f}\n"
        f"# Map Size: {size_km:.2f} km\n"
        f"# Zoom Level: {zoom}\n"
        f"# Projection: EPSG:3857 (Web Mercator)\n"
        f"# Bounds Min X (Mercator): {min_x:.4f}\n"
        f"# Bounds Min Y (Mercator): {min_y:.4f}\n"
        f"# Bounds Max X (Mercator): {max_x:.4f}\n"
        f"# Bounds Max Y (Mercator): {max_y:.4f}\n"
        f"# Mesh Origin Offset X (Mercator): {cx:.4f}\n"
        f"# Mesh Origin Offset Y (Mercator): {cy:.4f}\n"
        f"# Vertical Scale (Z-Scale): {z_scale:.4f}\n"
        "# To resolve actual Web Mercator coordinate from vertex coordinate (vx, vy, vz):\n"
        "#   Mercator X = vx + Offset X\n"
        "#   Mercator Y = vy + Offset Y\n"
        "#   Elevation Z = vz / Z-Scale\n\n"
    )
    with open(obj_path, 'w') as f:
        f.write(comments + original_obj)
        
    # Adjust files created by trimesh to clean up names and point to 'map_texture.png'
    default_mtl = os.path.join(output_dir, "material.mtl")
    default_tex = os.path.join(output_dir, "material_0.png")
    
    if os.path.exists(default_mtl):
        with open(default_mtl, 'r') as f:
            mtl_content = f.read()
        mtl_content = mtl_content.replace('material_0.png', 'map_texture.png')
        with open(mtl_path, 'w') as f:
            f.write(mtl_content)
        os.remove(default_mtl)
        
    if os.path.exists(default_tex):
        os.remove(default_tex)
        
    if os.path.exists(obj_path):
        with open(obj_path, 'r') as f:
            obj_content = f.read()
        obj_content = obj_content.replace('material.mtl', 'map.mtl')
        with open(obj_path, 'w') as f:
            f.write(obj_content)
            
    print(f"OBJ model saved to: {obj_path}")
    
    # Convert OBJ to FBX using Aspose.3D and embed metadata properties
    if a3d:
        print("Converting mesh to FBX and embedding metadata...")
        try:
            scene = Scene.from_file(obj_path)
            root = scene.root_node
            
            # Embed metadata directly as custom properties on the root node
            root.set_property("Latitude", lat)
            root.set_property("Longitude", lon)
            root.set_property("MapSizeKm", size_km)
            root.set_property("ZoomLevel", zoom)
            root.set_property("SpatialReference", "EPSG:3857")
            root.set_property("MinX", min_x)
            root.set_property("MinY", min_y)
            root.set_property("MaxX", max_x)
            root.set_property("MaxY", max_y)
            root.set_property("OffsetX", cx)
            root.set_property("OffsetY", cy)
            root.set_property("ZScale", z_scale)
            
            # Create a material and texture to embed in the FBX file
            from aspose.threed.shading import LambertMaterial, Texture
            mat = LambertMaterial()
            mat.name = "map_material"
            tex = Texture()
            tex.file_name = "map_texture.png"
            
            # Read texture bytes for embedding
            if os.path.exists(texture_png_path):
                try:
                    with open(texture_png_path, 'rb') as f_tex:
                        tex.content = f_tex.read()
                except Exception as e:
                    print(f"Warning: Could not read texture file for embedding: {e}")
            
            mat.set_texture(LambertMaterial.MAP_DIFFUSE, tex)
            
            # Apply material to children (geometry nodes)
            for child in root.child_nodes:
                child.material = mat
            
            options = FbxSaveOptions(FileFormat.FBX7500_BINARY)
            options.embed_textures = True
            scene.save(fbx_path, options)
            print(f"FBX model saved to: {fbx_path}")
        except Exception as e:
            print(f"Warning: FBX conversion failed: {e}")
    else:
        print("FBX conversion skipped (aspose-3d package is not available).")

def run_pipeline(lat, lon, size_km, zoom, provider, mapbox_token, mesh_res, z_scale, out_dir, keep_temp, height_only=False):
    print("=========================================")
    print("          GIS DATA BLENDER PIPELINE      ")
    print("=========================================")
    print(f"Center Coordinate: Lat={lat:.5f}, Lon={lon:.5f}")
    print(f"Map Extent       : {size_km:.2f} km x {size_km:.2f} km")
    print(f"Zoom Level       : {zoom}")
    print(f"Imagery Provider : {provider.upper()}")
    print(f"Mesh Grid Res    : {mesh_res}x{mesh_res} vertices")
    print(f"Vertical Scale   : {z_scale:.2f}x")
    print(f"Output Directory : {out_dir}")
    print("=========================================")
    
    check_gdal()
    # -----------------------------------------------------------------------------------    
    # 1. Use SAR footprint as the geographic reference
    sar_path = r"C:\Users\palak\Downloads\sar_grd_db.tif"

    sar_ref = get_sar_reference(sar_path)

    print("\n========== SAR REFERENCE ==========")
    print("CRS:", sar_ref["crs"])
    print("Bounds:", sar_ref["bounds"])
    print("Size:", sar_ref["width"], "x", sar_ref["height"])

    # Convert exact SAR footprint to Web Mercator
    bbox = sar_bounds_to_mercator(
        sar_ref["bounds"],
        sar_ref["crs"]
    )

    print("========== RGB DOWNLOAD FOOTPRINT ==========")
    print("Web Mercator bounds:", bbox)

    # Calculate Google tile range directly from SAR footprint
    min_x, min_y, max_x, max_y = bbox

    tile_size_m = (2.0 * M) / (2 ** zoom)

    xmin = int(math.floor((min_x + M) / tile_size_m))
    xmax = int(math.floor((max_x + M) / tile_size_m))

    ymin = int(math.floor((M - max_y) / tile_size_m))
    ymax = int(math.floor((M - min_y) / tile_size_m))

    # DEM tile range
    dem_zoom = min(zoom, 14)

    dem_tile_size_m = (2.0 * M) / (2 ** dem_zoom)

    d_xmin = int(math.floor((min_x + M) / dem_tile_size_m))
    d_xmax = int(math.floor((max_x + M) / dem_tile_size_m))

    d_ymin = int(math.floor((M - max_y) / dem_tile_size_m))
    d_ymax = int(math.floor((M - min_y) / dem_tile_size_m))


    # -----------------------------------------------------------------------------------
    
    nx_img = xmax - xmin + 1
    ny_img = ymax - ymin + 1
    nx_dem = d_xmax - d_xmin + 1
    ny_dem = d_ymax - d_ymin + 1
    
    total_img_tiles = nx_img * ny_img
    total_dem_tiles = nx_dem * ny_dem
    
    if not height_only:
        print(f"Imagery tile grid: {nx_img}x{ny_img} ({total_img_tiles} tiles)")
    print(f"Elevation tile grid: {nx_dem}x{ny_dem} ({total_dem_tiles} tiles)")
    
    if not height_only and total_img_tiles > 500:
        print(f"Warning: Large tile count ({total_img_tiles} tiles). This might consume substantial bandwidth.")
        if sys.stdin.isatty():
            confirm = input("Do you want to proceed? (y/n): ")
            if confirm.lower() != 'y':
                print("Pipeline cancelled by user.")
                return
    
    # Prepare directories
    temp_dir = os.path.join(out_dir, "temp_tiles")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    if height_only:
        # Gather DEM Tile Tasks
        dem_tasks = []
        for x in range(d_xmin, d_xmax+1):
            for y in range(d_ymin, d_ymax+1):
                url = f"https://elevation-tiles-prod.s3.amazonaws.com/geotiff/{dem_zoom}/{x}/{y}.tif"
                path = os.path.join(temp_dir, "elevation", f"{dem_zoom}_{x}_{y}.tif")
                dem_tasks.append((url, path, x, y))
                
        # Download in parallel
        print("Downloading Elevation Tiles...")
        dem_downloaded, dem_errors = download_tiles_parallel(dem_tasks, max_workers=10, desc="Elevation")
        
        if not dem_downloaded:
            print("Error: Failed to download any elevation tiles.")
            if dem_errors:
                print("Download errors encountered:")
                for err, count in dem_errors.items():
                    print(f"  - {err} ({count} times)")
            return
            
        vrt_dem = os.path.join(out_dir, "elevation.vrt")
        print("Generating Virtual Dataset (VRT) structures...")
        generate_vrt(vrt_dem, 512, 512, d_xmin, d_xmax, d_ymin, d_ymax, dem_zoom, dem_downloaded, is_dem=True)
        
        # Calculate native resolution of DEM
        tile_size_m_dem = (2.0 * M) / (2**dem_zoom)
        res_dem = tile_size_m_dem / 512.0
        
        dem_warped = os.path.join(out_dir, "dem_cropped.tif")
        print(f"Stitching & cropping DEM to boundaries (Resampling to {res_dem:.4f} m/pixel)...")
        warp_raster(vrt_dem, dem_warped, bbox, res_dem)
        
        with rasterio.open(dem_warped) as src_dem:
            elev_full = src_dem.read(1)
            
        valid_elev = elev_full[(elev_full != -32768) & (elev_full > -1000) & (~np.isnan(elev_full))]
        if valid_elev.size > 0:
            # Filter outliers using Tukey's IQR Method (k=3)
            q25, q75 = np.percentile(valid_elev, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 3.0 * iqr
            upper_bound = q75 + 3.0 * iqr
            filtered_elev = valid_elev[(valid_elev >= lower_bound) & (valid_elev <= upper_bound)]
            if filtered_elev.size == 0:
                filtered_elev = valid_elev
                
            min_elev = float(np.min(filtered_elev))
            max_elev = float(np.max(filtered_elev))
            p05 = float(np.percentile(filtered_elev, 5))
            p25 = float(np.percentile(filtered_elev, 25))
            p50 = float(np.percentile(filtered_elev, 50))
            p75 = float(np.percentile(filtered_elev, 75))
            p95 = float(np.percentile(filtered_elev, 95))
            mean_elev = float(np.mean(filtered_elev))
            std_dev = float(np.std(filtered_elev))
            
            base_range = p95 - p05
            top_range = max_elev - p95
            peakiness = top_range / base_range if base_range > 0.5 else top_range
            
            # Clean 2D DEM for spatial calculations
            median_val = p50
            cleaned_dem = elev_full.copy().astype(np.float32)
            cleaned_dem[(cleaned_dem == -32768) | (cleaned_dem <= -1000) | (np.isnan(cleaned_dem)) | (cleaned_dem < lower_bound) | (cleaned_dem > upper_bound)] = median_val
        else:
            min_elev = max_elev = p05 = p25 = p50 = p75 = p95 = mean_elev = std_dev = peakiness = 0.0
            cleaned_dem = elev_full.copy().astype(np.float32)

        # Compute Advanced Terrain Metrics
        tri_map = calculate_tri(cleaned_dem)
        slope_map = calculate_slope(cleaned_dem, res_dem, res_dem)
        curvature_map = calculate_curvature(cleaned_dem, res_dem, res_dem)
        relief_1k = calculate_local_relief(cleaned_dem, 1000.0, res_dem)
        relief_500m = calculate_local_relief(cleaned_dem, 500.0, res_dem)
        ups_map = calculate_ups(cleaned_dem)
        
        tri_mean = float(np.mean(tri_map))
        slope_dist = {
            "mean": float(np.mean(slope_map)),
            "std": float(np.std(slope_map)),
            "p95": float(np.percentile(slope_map, 95))
        }
        curvature_std = float(np.std(curvature_map))
        convex_pct = float(np.sum(curvature_map < 0) / curvature_map.size * 100)
        concave_pct = float(np.sum(curvature_map > 0) / curvature_map.size * 100)
        curvature_dist = {
            "mean": float(np.mean(curvature_map)),
            "std": curvature_std,
            "convex_pct": convex_pct,
            "concave_pct": concave_pct
        }
        mean_relief_1k = float(np.mean(relief_1k))
        mean_relief_500m = float(np.mean(relief_500m))
        entropy_val = calculate_entropy(cleaned_dem)
        auto_len = calculate_autocorrelation_length(cleaned_dem, res_dem)
        ups_mean = float(np.mean(ups_map))
        
        # TSI Calculation
        tsi_score, sub_scores = calculate_tsi(std_dev, tri_mean, entropy_val, auto_len, ups_mean)
        
        # Generate Visual Report
        report_png_path = os.path.join(out_dir, "terrain_analysis_report.png")
        autocorr_profile = compute_autocorrelation_profile(cleaned_dem)
        print("Generating Terrain Analysis Visual Report...")
        generate_terrain_dashboard(
            cleaned_dem, slope_map, tri_map, relief_1k, curvature_map, ups_map,
            slope_dist, curvature_dist, autocorr_profile, tsi_score, sub_scores,
            report_png_path, lat, lon, size_km
        )
        print(f"Terrain Analysis Visual Report saved to: {os.path.abspath(report_png_path)}")
        
        print("\n=========================================")
        print("          ELEVATION PROFILE ANALYSIS     ")
        print("=========================================")
        print(f"Min Elevation      : {min_elev:.2f} meters")
        print(f"Max Elevation      : {max_elev:.2f} meters")
        print(f"Elevation Range    : {max_elev - min_elev:.2f} meters")
        print(f"Mean Elevation     : {mean_elev:.2f} meters")
        print(f"Median (50th)      : {p50:.2f} meters")
        print(f"Std Deviation      : {std_dev:.2f} meters")
        print("-----------------------------------------")
        print("Percentile Distribution (meters):")
        print(f"  5th %ile : {p05:.2f} m")
        print(f" 25th %ile : {p25:.2f} m")
        print(f" 75th %ile : {p75:.2f} m")
        print(f" 95th %ile : {p95:.2f} m")
        print("-----------------------------------------")
        print("Advanced Terrain Metrics:")
        print(f"  Mean TRI (Ruggedness)      : {tri_mean:.2f} meters")
        print(f"  Slope (Mean/Std/95th)      : {slope_dist['mean']:.2f}° / {slope_dist['std']:.2f}° / {slope_dist['p95']:.2f}°")
        print(f"  Profile Autocorr Length    : {auto_len:.2f} meters")
        print(f"  2D Shannon Entropy         : {entropy_val:.2f} bits")
        print(f"  Mean Relief (500m / 1km)   : {mean_relief_500m:.2f} m / {mean_relief_1k:.2f} m")
        print(f"  Curvature (Mean/Std)       : {curvature_dist['mean']:.6f} / {curvature_dist['std']:.6f}")
        print(f"  Convex / Concave Terrain   : {convex_pct:.1f}% / {concave_pct:.1f}%")
        print(f"  Unique Profile Score (UPS) : {ups_mean:.4f}")
        print("-----------------------------------------")
        print("TERCOM NAV SUITABILITY:")
        print(f"  TSI Score                  : {tsi_score:.1f} / 100.0")
        
        # Terrain Profile Recommendation
        if std_dev < 1.5:
            recommendation = "Extremely Flat (poor for TERCOM)"
        elif peakiness > 1.2:
            if base_range < 5.0:
                recommendation = "Single Isolated Peak / Outlier (poor for TERCOM)"
            else:
                recommendation = "Prominent Peak / Localized Outlier (moderate suitability)"
        else:
            recommendation = "Varying Altitude Region (Ideal/High Suitability for TERCOM!)"
            
        print(f"  Terrain Classification     : {recommendation}")
        print("=========================================\n")
        
        # Clean up temporary tile cache and intermediate files
        if not keep_temp:
            print("Cleaning up temporary tile cache and intermediate files...")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(vrt_dem):
                os.remove(vrt_dem)
                
        print("Processing completed successfully!")
        print(f"Cropped elevation map saved to: {os.path.abspath(dem_warped)}")
        return
    
    # 2. Gather Imagery Tile Tasks
    img_tasks = []
    for x in range(xmin, xmax+1):
        for y in range(ymin, ymax+1):
            if provider == 'google':
                url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"
                ext = "jpg"
            elif provider == 'mapbox':
                url = f"https://api.mapbox.com/v4/mapbox.satellite/{zoom}/{x}/{y}.jpg?access_token={mapbox_token}"
                ext = "jpg"
            else:  # osm
                url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                ext = "png"
            path = os.path.join(temp_dir, "imagery", f"{zoom}_{x}_{y}.{ext}")
            img_tasks.append((url, path, x, y))
            
    # Gather DEM Tile Tasks
    dem_tasks = []
    for x in range(d_xmin, d_xmax+1):
        for y in range(d_ymin, d_ymax+1):
            url = f"https://elevation-tiles-prod.s3.amazonaws.com/geotiff/{dem_zoom}/{x}/{y}.tif"
            path = os.path.join(temp_dir, "elevation", f"{dem_zoom}_{x}_{y}.tif")
            dem_tasks.append((url, path, x, y))
            
    # Download in parallel
    print("Downloading Imagery Tiles...")
    img_downloaded, img_errors = download_tiles_parallel(img_tasks, max_workers=10, desc="Imagery")
    
    print("Downloading Elevation Tiles...")
    dem_downloaded, dem_errors = download_tiles_parallel(dem_tasks, max_workers=10, desc="Elevation")
    
    if not img_downloaded:
        print("Error: Failed to download any imagery tiles.")
        if img_errors:
            print("Download errors encountered:")
            for err, count in img_errors.items():
                print(f"  - {err} ({count} times)")
        return
    if not dem_downloaded:
        print("Error: Failed to download any elevation tiles.")
        if dem_errors:
            print("Download errors encountered:")
            for err, count in dem_errors.items():
                print(f"  - {err} ({count} times)")
        return
        
    # 3. Create VRT Files
    vrt_img = os.path.join(out_dir, "imagery.vrt")
    vrt_dem = os.path.join(out_dir, "elevation.vrt")
    
    print("Generating Virtual Dataset (VRT) structures...")
    generate_vrt(vrt_img, 256, 256, xmin, xmax, ymin, ymax, zoom, img_downloaded, is_dem=False)
    generate_vrt(vrt_dem, 512, 512, d_xmin, d_xmax, d_ymin, d_ymax, dem_zoom, dem_downloaded, is_dem=True)
    
    # Calculate crop resolution based on ortho imagery pixel resolution
    tile_size_m_img = (2.0 * M) / (2**zoom)
    res_img = tile_size_m_img / 256.0
    
    # Warped intermediate files
    img_warped = os.path.join(out_dir, "img_cropped.tif")
    dem_warped = os.path.join(out_dir, "dem_cropped.tif")
    final_tif = os.path.join(out_dir, "final_map.tif")
    
    print(f"Stitching & cropping to boundaries (Resampling to {res_img:.4f} m/pixel)...")
    warp_raster(vrt_img, img_warped, bbox, res_img)
    warp_raster(vrt_dem, dem_warped, bbox, res_img)
    
    # 4. Merge RGB + DEM into 4-Band GeoTIFF
    print("Merging RGB imagery and Elevation into final 4-band GeoTIFF...")
    with rasterio.open(img_warped) as src_img:
        img_data = src_img.read()
        profile = src_img.profile
        
    with rasterio.open(dem_warped) as src_dem:
        dem_data = src_dem.read(1)
        
    profile.update(
        count=4,
        dtype='float32',
        compress='lzw'
    )
    
    with rasterio.open(final_tif, 'w', **profile) as dst:
        dst.write(img_data[0].astype('float32'), 1)
        dst.write(img_data[1].astype('float32'), 2)
        dst.write(img_data[2].astype('float32'), 3)
        dst.write(dem_data.astype('float32'), 4)
        
    print(f"Multi-band GeoTIFF created at: {final_tif}")
    
    # 5. Build 3D Mesh & Textures & Metadata
    build_3d_mesh(final_tif, bbox, mesh_res, z_scale, out_dir, lat, lon, size_km, zoom)
    
    # Rename img_warped (img_cropped.tif) to map_texture.tif to keep georeferenced texture
    texture_tif_dest = os.path.join(out_dir, "map_texture.tif")
    if os.path.exists(img_warped):
        shutil.move(img_warped, texture_tif_dest)
        print(f"Georeferenced GeoTIFF texture saved to: {texture_tif_dest}")
    
    # Clean up intermediate files
    if not keep_temp:
        print("Cleaning up temporary tile cache and intermediate cropped files...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        for f in [vrt_img, vrt_dem, dem_warped]:
            if os.path.exists(f):
                os.remove(f)
                
    print("\nProcessing completed successfully!")
    print("Outputs located in:", os.path.abspath(out_dir))
    print(f"- [GeoTIFF Map] final_map.tif (Bands 1-3: RGB, Band 4: Elevation)")
    print(f"- [GeoTIFF Texture] map_texture.tif & map_texture.tfw (with spatial headers)")
    print(f"- [PNG Texture] map_texture.png & map_texture.pgw (with world files)")
    print(f"- [OBJ Model] map.obj & map.mtl (with embedded metadata comments)")
    print(f"- [Metadata JSON] map_metadata.json (for spatial resolution of coordinates)")
    if a3d:
        print(f"- [FBX Model] map.fbx (with embedded metadata properties)")

def interactive_wizard():
    print("=========================================")
    print("      Welcome to GIS Data Blender Wizard ")
    print("=========================================")
    try:
        lat_str = input("Latitude (e.g. 35.3606 for Mt. Fuji): ").strip()
        lat = float(lat_str)
        lon_str = input("Longitude (e.g. 138.7274): ").strip()
        lon = float(lon_str)
        
        if not (-85.0511 <= lat <= 85.0511):
            print("Error: Latitude must be between -85.0511 and 85.0511 (Web Mercator limit).")
            return
        if not (-180.0 <= lon <= 180.0):
            print("Error: Longitude must be between -180 and 180.")
            return

        size_str = input("Map Area Dimension in km (default 10.0): ").strip()
        size_km = float(size_str) if size_str else 10.0
        
        zoom_str = input("Map Imagery Zoom Level (default 15, range 0-20): ").strip()
        zoom = int(zoom_str) if zoom_str else 15
        
        print("\nSelect Imagery Provider:")
        print("1. Google Maps Satellite (No API key needed, high detail) [Default]")
        print("2. Mapbox Satellite (Requires Mapbox Access Token)")
        print("3. OpenStreetMap (No API key needed, street vector map)")
        prov_choice = input("Provider choice (1-3): ").strip()
        
        if prov_choice == '2':
            provider = 'mapbox'
            mapbox_token = input("Enter Mapbox Access Token: ").strip()
            if not mapbox_token:
                print("Error: Mapbox token is required for Mapbox provider.")
                return
        elif prov_choice == '3':
            provider = 'osm'
            mapbox_token = None
        else:
            provider = 'google'
            mapbox_token = None

        mesh_str = input("3D Mesh Grid Resolution (e.g., 200 for 200x200 grid, default 200): ").strip()
        mesh_res = int(mesh_str) if mesh_str else 200
        
        z_str = input("Vertical Scale/Exaggeration (default 1.0): ").strip()
        z_scale = float(z_str) if z_str else 1.0
        
        out_dir = input("Output folder path (default './output'): ").strip()
        if not out_dir:
            out_dir = './output'
            
        keep_temp_str = input("Keep temporary download cache tiles? (y/n, default n): ").strip()
        keep_temp = keep_temp_str.lower() == 'y'
        
        height_str = input("Measure height only (skip imagery/mesh generation)? (y/n, default n): ").strip()
        height_only = height_str.lower() == 'y'
        
        run_pipeline(lat, lon, size_km, zoom, provider, mapbox_token, mesh_res, z_scale, out_dir, keep_temp, height_only=height_only)
        
    except ValueError as e:
        print(f"Error parsing input values: {e}")
    except KeyboardInterrupt:
        print("\nWizard cancelled.")

# def main():
#     parser = argparse.ArgumentParser(description="Download ortho imagery & DEM elevation and generate 3D models.")
#     parser.add_argument("--lat", type=float, help="Latitude of the center coordinate")
#     parser.add_argument("--lon", type=float, help="Longitude of the center coordinate")
#     parser.add_argument("--size", type=float, default=10.0, help="Area size in kilometers (e.g. 10.0 for 10x10km)")
#     parser.add_argument("--zoom", type=int, default=15, help="Zoom level of map tiles (0-20, default 15)")
#     parser.add_argument("--provider", choices=["google", "mapbox", "osm"], default="google", help="Imagery source provider")
#     parser.add_argument("--mapbox-token", type=str, help="Mapbox API access token")
#     parser.add_argument("--mesh-res", type=int, default=200, help="Mesh grid vertex resolution (default 200)")
#     parser.add_argument("--z-scale", type=float, default=1.0, help="Vertical scale factor for mesh elevation (default 1.0)")
#     parser.add_argument("--out-dir", type=str, default="./output", help="Directory to save final files")
#     parser.add_argument("--keep-temp", action="store_true", help="Keep cached tile images and intermediate warp files")
#     parser.add_argument("--height", action="store_true", help="Only download DEM elevation data and analyze the height profile of the region")
    
#     args = parser.parse_args()
    
#     # If no latitude and longitude arguments are provided, launch the interactive wizard
#     if args.lat is None or args.lon is None:
#         interactive_wizard()
#     else:
#         # Validate mapbox token if chosen
#         if args.provider == "mapbox" and not args.mapbox_token:
#             print("Error: --mapbox-token is required when --provider is set to 'mapbox'")
#             sys.exit(1)
#         run_pipeline(
#             args.lat, args.lon, args.size, args.zoom,
#             args.provider, args.mapbox_token, args.mesh_res,
#             args.z_scale, args.out_dir, args.keep_temp,
#             height_only=args.height
#         )

def main():
    parser = argparse.ArgumentParser(
        description="Download ortho imagery & DEM elevation and generate 3D models."
    )

    parser.add_argument("--lat", type=float,
                        help="Latitude of the center coordinate")
    parser.add_argument("--lon", type=float,
                        help="Longitude of the center coordinate")
    parser.add_argument("--size", type=float, default=10.0,
                        help="Area size in kilometers")
    parser.add_argument("--zoom", type=int, default=15,
                        help="Zoom level of map tiles (0-20)")
    parser.add_argument("--provider",
                        choices=["google", "mapbox", "osm"],
                        default="google",
                        help="Imagery source provider")
    parser.add_argument("--mapbox-token", type=str,
                        help="Mapbox API access token")
    parser.add_argument("--mesh-res", type=int, default=200,
                        help="Mesh grid vertex resolution")
    parser.add_argument("--z-scale", type=float, default=1.0,
                        help="Vertical scale factor for mesh elevation")
    parser.add_argument("--out-dir", type=str, default="./output",
                        help="Directory to save final files")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep cached tile images and intermediate files")
    parser.add_argument("--height", action="store_true",
                        help="Only download DEM elevation data")

    args = parser.parse_args()

    # If no latitude and longitude are provided,
    # launch the interactive wizard.
    if args.lat is None or args.lon is None:
        interactive_wizard()

    else:
        # Validate Mapbox token if needed
        if args.provider == "mapbox" and not args.mapbox_token:
            print(
                "Error: --mapbox-token is required "
                "when --provider is set to 'mapbox'"
            )
            sys.exit(1)

        run_pipeline(
            args.lat,
            args.lon,
            args.size,
            args.zoom,
            args.provider,
            args.mapbox_token,
            args.mesh_res,
            args.z_scale,
            args.out_dir,
            args.keep_temp,
            height_only=args.height
        )


if __name__ == "__main__":
    main()





if __name__ == "__main__":
    main()
