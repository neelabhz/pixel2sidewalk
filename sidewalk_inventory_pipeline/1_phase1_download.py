#!/usr/bin/env python3
"""
Phase 1: Download Mapillary Sidewalk Masks & Metadata
Run on the login node (requires internet).

Downloads segmentation masks and camera metadata for each image
in the study area. Uses SfM-refined compass angles and geometry
where available, and flags panoramic images for special handling
in Phase 2.
"""

import os
import sys
import json
import base64
import requests
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

ACCESS_TOKEN = os.environ.get("MAPILLARY_CLIENT_TOKEN")

SIDEWALK_CLASSES = [
    "construction--flat--sidewalk",
]


def decode_detection_geometry(geometry_b64: str, img_width: int, img_height: int) -> list:
    try:
        import mapbox_vector_tile
    except ImportError:
        print("Error: mapbox-vector-tile not installed. Run: pip install mapbox-vector-tile")
        sys.exit(1)

    decoded = base64.decodebytes(geometry_b64.encode("utf-8"))
    tile_data = mapbox_vector_tile.decode(decoded)

    polygons = []
    for layer_name, layer in tile_data.items():
        extent = layer.get("extent", 4096)
        for feature in layer.get("features", []):
            geom = feature.get("geometry", {})
            if geom.get("type") == "Polygon":
                for ring in geom.get("coordinates", []):
                    pixel_coords = []
                    for x, y in ring:
                        px = (x / extent) * img_width
                        # Vector tiles: Y=0 at bottom. Images: Y=0 at top. Flip required.
                        py = (1 - y / extent) * img_height
                        pixel_coords.append((px, py))
                    polygons.append(pixel_coords)
    return polygons


def get_image_details(image_id: str) -> dict:
    url = f"https://graph.mapillary.com/{image_id}"
    params = {
        "access_token": ACCESS_TOKEN,
        "fields": (
            "make,model,width,height,camera_type,is_pano,"
            "compass_angle,computed_compass_angle,"
            "geometry,computed_geometry,"
            "captured_at"
        ),
    }
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                d = r.json()
                raw_angle      = d.get("compass_angle")
                computed_angle = d.get("computed_compass_angle")
                best_angle     = computed_angle if computed_angle is not None else raw_angle

                raw_geom      = d.get("geometry")
                computed_geom = d.get("computed_geometry")
                best_geom     = computed_geom if computed_geom is not None else raw_geom

                return {
                    "make":              d.get("make", "Unknown"),
                    "model":             d.get("model", "Unknown"),
                    "width":             d.get("width", 4000),
                    "height":            d.get("height", 3000),
                    "camera_type":       d.get("camera_type", "perspective"),
                    "is_pano":           bool(d.get("is_pano", False)),
                    "compass_angle":          raw_angle,
                    "computed_compass_angle": computed_angle,
                    "best_compass_angle":     best_angle,
                    "geometry":          raw_geom,
                    "computed_geometry": computed_geom,
                    "best_geometry":     best_geom,
                    "captured_at":       d.get("captured_at"),
                }
        except Exception as e:
            if attempt == 2:
                print(f"  [warn] get_image_details failed for {image_id}: {e}")
            time.sleep(1)

    return {
        "make": "Unknown", "model": "Unknown", "width": 4000, "height": 3000,
        "camera_type": "perspective", "is_pano": False,
        "compass_angle": None, "computed_compass_angle": None, "best_compass_angle": None,
        "geometry": None, "computed_geometry": None, "best_geometry": None,
        "captured_at": None,
    }


def get_image_detections(image_id: str) -> list:
    detections = []
    url = f"https://graph.mapillary.com/{image_id}/detections"
    params = {"access_token": ACCESS_TOKEN, "fields": "value,geometry"}

    while url:
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    detections.extend(data.get("data", []))
                    url = data.get("paging", {}).get("next")
                    params = {}
                    break
            except Exception as e:
                if attempt == 2:
                    print(f"  [warn] detections page failed for {image_id}: {e}")
                    url = None
                time.sleep(1)
        else:
            url = None

    return detections


def process_image(meta_file: Path, masks_dir: Path) -> tuple:
    try:
        with open(meta_file, "r") as f:
            meta = json.load(f)
    except Exception as e:
        return False, f"Could not read {meta_file}: {e}"

    img_id = meta.get("id")
    if not img_id:
        return False, f"No 'id' field in {meta_file}"

    mask_path = masks_dir / f"{img_id}_mask.png"
    already_have_mask    = mask_path.exists()
    already_have_details = "best_compass_angle" in meta

    if already_have_mask and already_have_details:
        return True, ""

    details = get_image_details(img_id)

    # Write all fetched fields into metadata
    for key, val in details.items():
        meta[key] = val

    if not already_have_mask:
        width, height = details["width"], details["height"]
        detections = get_image_detections(img_id)

        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)

        for det in detections:
            val = det.get("value", "")
            if any(cls in val for cls in SIDEWALK_CLASSES):
                geo = det.get("geometry", "")
                if geo:
                    polys = decode_detection_geometry(geo, width, height)
                    for poly in polys:
                        if len(poly) >= 3:
                            draw.polygon(poly, fill=255)

        mask_arr = np.array(mask_img)
        meta["sidewalk_pixel_count_raw"] = int(np.count_nonzero(mask_arr))
        mask_img.save(mask_path)

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    args = parser.parse_args()

    meta_dir  = args.target_dir / "metadata"
    masks_dir = args.target_dir / "masks"

    if not meta_dir.exists():
        print(f"Error: {meta_dir} not found.")
        sys.exit(1)

    masks_dir.mkdir(exist_ok=True)
    meta_files = list(meta_dir.glob("*.json"))

    print("=" * 60)
    print(f"PHASE 1: DOWNLOADING MAPILLARY MASKS & METADATA - {args.target_dir.name}")
    print(f"Images to process: {len(meta_files)}")
    print("=" * 60)

    successes, errors = 0, []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_image, mf, masks_dir): mf for mf in meta_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            try:
                ok, err = future.result()
                if ok:
                    successes += 1
                else:
                    errors.append(err)
            except Exception as e:
                errors.append(f"Thread exception: {e}")

    print(f"\nCompleted: {successes}/{len(meta_files)}")
    if errors:
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
