#!/usr/bin/env python3
"""
Step 0: Setup a new test area by downloading Mapillary image metadata
and caching the OSM road network graph for offline use.

Handles large bounding boxes by tiling them into smaller sub-boxes
to avoid Mapillary API 500 errors.
"""

import os, sys, json, time, requests
from pathlib import Path

import osmnx as ox

import os
ACCESS_TOKEN = os.environ.get("MAPILLARY_CLIENT_TOKEN", "YOUR_API_KEY_HERE")

# ── Area Configuration ────────────────────────────────────────────────────────
# Change BBOX and AREA_NAME here to run for any new city / district.
# BBOX format: (min_lon, min_lat, max_lon, max_lat)
#
# For a BIGGER area, just expand the bbox — e.g. full Amsterdam centre:
#   BBOX = (4.870, 52.345, 4.930, 52.385)  # ~5 km × 5 km, ~200K images
#
# Current: wider Zuidas (~4× bigger — covers full business district + surrounding streets)
# Previous small bbox was: (4.865, 52.334, 4.875, 52.3415)  → 13,824 images
# This expanded bbox adds RAI, Vrije Universiteit, WTC surroundings
BBOX      = (4.855, 52.328, 4.890, 52.348)
AREA_NAME = "amsterdam_zuidas_wide"

# Tile size in degrees — small enough that Mapillary won't reject the request.
# ~0.002° ≈ 136m lon × 222m lat at this latitude. Works well for dense urban areas.
TILE_STEP = 0.002

IMAGE_FIELDS = [
    'id', 'width', 'height', 'sequence', 'captured_at',
    'camera_type', 'is_pano', 'computed_geometry', 'computed_compass_angle',
    'geometry', 'compass_angle', 'thumb_original_url'
]


def fetch_images_in_bbox(bbox_str: str) -> list:
    """Fetch images from Graph API for a single small bbox. Handles pagination."""
    url = "https://graph.mapillary.com/images"
    params = {
        'bbox': bbox_str,
        'fields': ','.join(IMAGE_FIELDS),
        'limit': 2000
    }
    headers = {'Authorization': f'OAuth {ACCESS_TOKEN}'}
    images = []

    while url:
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    images.extend(data.get('data', []))
                    # Follow pagination cursor
                    url = data.get('paging', {}).get('next')
                    params = {}  # cursor URL already has params
                    break
                elif r.status_code == 500:
                    print(f"  [warn] 500 for bbox {bbox_str} — skipping this tile")
                    return images
                else:
                    print(f"  [warn] HTTP {r.status_code} for bbox {bbox_str}")
                    time.sleep(2)
            except Exception as e:
                if attempt == 2:
                    print(f"  [err] Failed bbox {bbox_str}: {e}")
                time.sleep(2)
        else:
            break  # all retries exhausted

    return images


def tile_bbox(min_lon, min_lat, max_lon, max_lat, step):
    """Yield (min_lon, min_lat, max_lon, max_lat) sub-tiles."""
    lon = min_lon
    while lon < max_lon:
        lat = min_lat
        while lat < max_lat:
            yield (
                round(lon, 6),
                round(lat, 6),
                round(min(lon + step, max_lon), 6),
                round(min(lat + step, max_lat), 6),
            )
            lat += step
        lon += step


def main():
    target_dir = Path(AREA_NAME)
    meta_dir = target_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"0_SETUP: Initializing {AREA_NAME}")
    print(f"Bounding box: {BBOX}")
    print("=" * 60)

    # ---- 1. Download Mapillary metadata via tiled sub-boxes ----
    existing_meta = list(meta_dir.glob("*.json"))
    if len(existing_meta) > 0:
        print(f"\n1. SKIPPING — {len(existing_meta)} metadata files already exist in {meta_dir}/")
    else:
        print("\n1. Downloading Mapillary image metadata (tiled)...")
        tiles = list(tile_bbox(*BBOX, TILE_STEP))
        print(f"   Splitting into {len(tiles)} sub-tiles of ~{TILE_STEP}° each")

        all_images = {}
        for i, (t_minlon, t_minlat, t_maxlon, t_maxlat) in enumerate(tiles):
            bbox_str = f"{t_minlon},{t_minlat},{t_maxlon},{t_maxlat}"
            imgs = fetch_images_in_bbox(bbox_str)
            for img in imgs:
                all_images[img['id']] = img
            print(f"   Tile {i+1}/{len(tiles)}: +{len(imgs)} images  (total unique: {len(all_images)})")
            time.sleep(0.3)

        print(f"\n   Total unique images found: {len(all_images)}")

        if not all_images:
            print("   ERROR: No images found! Check the bounding box coordinates.")
            sys.exit(1)

        for img_id, img in all_images.items():
            with open(meta_dir / f"{img_id}_metadata.json", 'w') as f:
                json.dump(img, f, indent=2)

        with open(target_dir / f"{AREA_NAME}_all_metadata.json", 'w') as f:
            json.dump(list(all_images.values()), f, indent=2)

        print(f"   Metadata saved to {meta_dir}/")

    # ---- 2. Download OSM road network for the same bbox ----
    graphml_path = target_dir / f"{AREA_NAME}.graphml"
    pkl_path = target_dir / f"_{AREA_NAME}_osm.pkl"
    if graphml_path.exists() or pkl_path.exists():
        print(f"\n2. SKIPPING — OSM graph already cached ({graphml_path.name})")
    else:
        print("\n2. Downloading OSM network graph...")
        import glob, pickle
        meta_files = glob.glob(str(meta_dir / "*_metadata.json"))
        lats, lons = [], []
        for mf in meta_files:
            try:
                with open(mf) as f:
                    meta = json.load(f)
                geom = meta.get("computed_geometry") or meta.get("geometry")
                if geom and geom.get("type") == "Point":
                    lons.append(geom["coordinates"][0])
                    lats.append(geom["coordinates"][1])
            except Exception:
                pass

        if not lats:
            print("   ERROR: No valid coordinates found in metadata")
            sys.exit(1)

        margin = 0.001
        west  = min(lons) - margin
        south = min(lats) - margin
        east  = max(lons) + margin
        north = max(lats) + margin
        print(f"   Computed bbox from {len(lats)} images: W={west:.4f} S={south:.4f} E={east:.4f} N={north:.4f}")

        G = ox.graph_from_bbox(bbox=(west, south, east, north),
                               network_type="all", retain_all=False, simplify=True)

        with open(pkl_path, "wb") as f:
            pickle.dump(G, f)
        ox.save_graphml(G, filepath=str(graphml_path))

        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        print(f"   OSM graph saved:  {graphml_path}")
        print(f"   Road segments:    {len(edges)}")
        print(f"   Highway types:    {edges['highway'].explode().value_counts().to_dict()}")

    print(f"\n{'='*60}")
    print("SETUP COMPLETE — ready for Phase 1, 2, 3")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
