#!/usr/bin/env python3
"""
Mapillary BBox Downloader for Multi-Sequence Section Mapping
=============================================================
Universal pipeline — works for any city/location.

Features:
  - Downloads images within a given bounding box
  - Skips images missing camera_parameters (avoids broken perspective projections)
  - Supports configurable image resolution (thumb_1024_url, thumb_2048_url, thumb_original_url)
  - Caches OSM road network using actual image spread (not download bbox)

Note: Mapillary Detection API pre-filtering has been removed — its semantic
segmentation labels are too coarse and unreliable for sidewalk detection.
DINOv3 segmentation in Phase 2 handles this far more accurately.
"""

import os
import json
import requests
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

ACCESS_TOKEN = os.environ.get("MAPILLARY_CLIENT_TOKEN", "YOUR_API_KEY_HERE")

# Add your study areas here — each is a "west,south,east,north" bounding box string
SECTIONS = {
    "boston_backbay": "-71.088,42.348,-71.075,42.355",
    "boston_south_end": "-71.0777,42.3383,-71.0697,42.3443",
}

BASE_DIR = Path("./data")
MAX_WORKERS = 8

# Fields to request from Mapillary — includes ALL needed metadata for the pipeline
IMAGE_FIELDS = [
    'id', 'width', 'height', 'sequence', 'captured_at',
    'camera_parameters', 'camera_type', 'is_pano',
    'computed_geometry', 'computed_rotation', 'computed_compass_angle',
    'geometry', 'compass_angle',
    'thumb_original_url', 'thumb_2048_url', 'thumb_1024_url'
]

# =============================================================================
# RESOLUTION MAP
# =============================================================================
RESOLUTION_MAP = {
    "original": "thumb_original_url",
    "2048":     "thumb_2048_url",
    "1024":     "thumb_1024_url",
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def api_get(url, params=None, retries=3):
    headers = {'Authorization': f'OAuth {ACCESS_TOKEN}'}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Failed: {e}")
                return None
            time.sleep(1)
    return None

def download_file(url, output_path, retries=3):
    headers = {'Authorization': f'OAuth {ACCESS_TOKEN}'}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
        except Exception:
            if attempt == retries - 1:
                return False
            time.sleep(1)
    return False

def get_images_in_bbox(bbox_str):
    """Fetch all images from the Mapillary API within a bounding box.
    For large bboxes, splits into smaller tiles to avoid API 500 errors.
    Handles pagination and deduplicates by image ID.
    """
    west, south, east, north = map(float, bbox_str.split(','))

    # Tile size in degrees (~300m at mid-latitudes)
    tile_size = 0.003

    # Generate tiles
    tiles = []
    lon = west
    while lon < east:
        lat = south
        while lat < north:
            t_west = lon
            t_south = lat
            t_east = min(lon + tile_size, east)
            t_north = min(lat + tile_size, north)
            tiles.append(f"{t_west},{t_south},{t_east},{t_north}")
            lat += tile_size
        lon += tile_size

    print(f"    Querying {len(tiles)} tiles...")

    seen_ids = set()
    all_images = []

    for tile_idx, tile_bbox in enumerate(tiles):
        url = "https://graph.mapillary.com/images"
        params = {
            'bbox': tile_bbox,
            'fields': ','.join(IMAGE_FIELDS),
            'limit': 2000
        }
        page = 1
        while True:
            data = api_get(url, params)
            if not data:
                break
            images = data.get('data', [])
            for img in images:
                if img['id'] not in seen_ids:
                    seen_ids.add(img['id'])
                    all_images.append(img)
            # Check for next page
            next_url = data.get('paging', {}).get('next')
            if next_url and len(images) == 2000:
                url = next_url
                params = None
                page += 1
            else:
                break

        if (tile_idx + 1) % 5 == 0 or tile_idx == len(tiles) - 1:
            print(f"    Tile {tile_idx+1}/{len(tiles)}: {len(all_images)} unique images so far")

    return all_images

def validate_image_metadata(img):
    """Check if an image has all the metadata required for the pipeline.
    Returns (is_valid, reason_if_invalid).
    """
    # Must have computed geometry (GPS position)
    geom = img.get("computed_geometry", {}).get("coordinates", [None, None])
    if geom[0] is None or geom[1] is None:
        return False, "no_gps"

    # Must have computed rotation
    if not img.get("computed_rotation"):
        return False, "no_rotation"

    # Perspective images MUST have camera_parameters for focal length
    is_pano = bool(img.get("is_pano", False)) or img.get("camera_type") == "spherical"
    if not is_pano:
        cam_params = img.get("camera_parameters")
        if cam_params is None or len(cam_params) == 0:
            return False, "no_camera_params"

    return True, "ok"

def compute_osm_bbox_from_images(images, margin=0.002):
    """Compute a bounding box from the actual GPS spread of downloaded images.
    This ensures the OSM road network covers all the streets where images exist,
    not just the tiny download bbox.
    """
    lats, lons = [], []
    for img in images:
        geom = img.get("computed_geometry", {}).get("coordinates", [None, None])
        if geom[0] is not None:
            lons.append(geom[0])
            lats.append(geom[1])
    if not lats:
        return None
    return (min(lons) - margin, min(lats) - margin,
            max(lons) + margin, max(lats) + margin)


def main():
    parser = argparse.ArgumentParser(description="Download Mapillary images for sidewalk mapping")
    parser.add_argument("--sections", nargs='+', default=list(SECTIONS.keys()),
                        help="Section names to process (default: all)")
    parser.add_argument("--resolution", choices=["original", "2048", "1024"], default="2048",
                        help="Image download resolution (default: 2048). "
                             "Lower = faster segmentation & depth, but less detail.")
    parser.add_argument("--base-dir", type=str, default=str(BASE_DIR),
                        help="Base directory for output")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    image_quality = RESOLUTION_MAP[args.resolution]

    for section_name in args.sections:
        bbox = SECTIONS.get(section_name)
        if not bbox:
            print(f"  Unknown section: {section_name}")
            continue

        print(f"\n{'='*60}\nProcessing Section: {section_name} [{bbox}]\n"
              f"Resolution: {args.resolution}\n{'='*60}")

        section_dir = base_dir / section_name
        section_dir.mkdir(exist_ok=True)
        img_dir = section_dir / "images"
        meta_dir = section_dir / "metadata"
        img_dir.mkdir(exist_ok=True)
        meta_dir.mkdir(exist_ok=True)

        # 1. Search bbox
        print("  [1/4] Searching Mapillary API...")
        all_images = get_images_in_bbox(bbox)
        print(f"  Found {len(all_images)} raw images in bbox")

        if not all_images:
            continue

        # 2. Validate metadata — skip images that will fail in the pipeline
        print("  [2/4] Validating image metadata...")
        valid_images = []
        skip_reasons = {}
        for img in all_images:
            is_valid, reason = validate_image_metadata(img)
            if is_valid:
                valid_images.append(img)
            else:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        print(f"  Metadata validation: {len(valid_images)} valid, "
              f"{len(all_images) - len(valid_images)} skipped")
        if skip_reasons:
            for reason, count in skip_reasons.items():
                print(f"    Skipped {count} images: {reason}")

        if not valid_images:
            continue

        images = valid_images

        # 3. Pre-download OSM network (using image spread, not download bbox!)
        import pickle
        osm_cache = section_dir / "_osm_roads.pkl"
        print("  [3/4] Caching OSM road network...")
        osm_bbox = compute_osm_bbox_from_images(images)
        if osm_bbox:
            try:
                import osmnx as ox
                # osmnx bbox=(left, bottom, right, top) = (west, south, east, north)
                _w, _s, _e, _n = osm_bbox
                G = ox.graph_from_bbox(bbox=(_w, _s, _e, _n),
                                       network_type="drive", retain_all=False, simplify=True)
                with open(osm_cache, 'wb') as f:
                    pickle.dump(G, f)
                print(f"    Saved OSM cache: {len(G.nodes())} nodes, {len(G.edges())} edges")
            except Exception as e:
                print(f"    Warning: Could not download OSM network: {e}")
        else:
            print("    Warning: Could not compute OSM bbox from images")

        # 4. Download images + save metadata
        print(f"  [4/4] Downloading images ({args.resolution} resolution)...")

        # Save individual metadata files
        for img in images:
            with open(meta_dir / f"{img['id']}_metadata.json", 'w') as f:
                json.dump(img, f, indent=2)

        # Save combined metadata
        with open(section_dir / f"{section_name}_all_metadata.json", 'w') as f:
            json.dump(images, f, indent=2)

        # Download images
        download_tasks = []
        for img in images:
            url = img.get(image_quality) or img.get('thumb_original_url')
            if url:
                out_path = img_dir / f"{img['id']}.jpg"
                if not out_path.exists():
                    download_tasks.append((url, out_path))

        if download_tasks:
            success = 0
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(download_file, url, path): path
                           for url, path in download_tasks}
                for i, future in enumerate(as_completed(futures)):
                    if future.result():
                        success += 1
                    if (i + 1) % 100 == 0:
                        print(f"    Downloaded {i+1}/{len(download_tasks)}...")
            print(f"  Downloaded {success} new images.")
        else:
            print("  All images already downloaded.")

        # Print summary
        n_pano = sum(1 for i in images if bool(i.get('is_pano')) or i.get('camera_type') == 'spherical')
        print(f"\n  Summary for {section_name}:")
        print(f"    Total images:     {len(images)}")
        print(f"    Panoramic:        {n_pano}")
        print(f"    Perspective:      {len(images) - n_pano}")
        print(f"    Resolution:       {args.resolution}")

if __name__ == "__main__":
    main()
