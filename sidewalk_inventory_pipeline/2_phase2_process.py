#!/usr/bin/env python3
"""
Phase 2: Process Downloaded Masks to Point Map (cardinal-aware, pano-aware)
Run entirely offline on the compute cluster.

Fixes applied:
  - Uses best_compass_angle (computed SfM value where available, raw EXIF fallback)
  - Uses best_geometry (computed SfM position where available)
  - Branches on is_pano: perspective cameras get ±90° cardinal bearings;
    panoramic images get 8-sector analysis covering all compass directions
"""

import sys
import json
import math
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

PANO_SECTORS = [
    ("N",  337.5, 22.5),
    ("NE",  22.5, 67.5),
    ("E",   67.5, 112.5),
    ("SE", 112.5, 157.5),
    ("S",  157.5, 202.5),
    ("SW", 202.5, 247.5),
    ("W",  247.5, 292.5),
    ("NW", 292.5, 337.5),
]


def normalize_bearing(deg: float) -> float:
    return deg % 360

def angular_difference(a: float, b: float) -> float:
    diff = (b - a + 360) % 360
    if diff > 180:
        diff -= 360
    return diff

def cardinal_sides(compass_angle: float) -> tuple:
    return normalize_bearing(compass_angle - 90.0), normalize_bearing(compass_angle + 90.0)


def pano_sector_col_range(sector_bearing_start: float, sector_bearing_end: float,
                           compass_angle: float, img_width: int) -> tuple:
    def bearing_to_col(bearing: float) -> float:
        relative = normalize_bearing(bearing - compass_angle)
        centered_relative = normalize_bearing(relative + 180.0)
        return (centered_relative / 360.0) * img_width

    c_start = bearing_to_col(sector_bearing_start)
    c_end   = bearing_to_col(sector_bearing_end)
    return int(c_start), int(c_end)


def analyse_pano_mask(mask: np.ndarray, compass_angle: float) -> dict:
    height, width = mask.shape
    total_pixels  = width * height
    lower = mask[int(height * 0.4):, :]
    lower_h, lower_w = lower.shape
    lower_total = lower_h * lower_w

    results = {}
    for name, b_start, b_end in PANO_SECTORS:
        col_start, col_end = pano_sector_col_range(b_start, b_end, compass_angle, lower_w)

        if col_start < col_end:
            sector_mask = lower[:, col_start:col_end]
        else:
            left_part  = lower[:, col_start:]
            right_part = lower[:, :col_end]
            sector_mask = np.concatenate([left_part, right_part], axis=1)

        pixels = int(np.count_nonzero(sector_mask))
        pct = round(pixels / lower_total * 100, 3) if lower_total > 0 else 0.0
        results[name] = pct

    return results


def analyse_perspective_mask(mask: np.ndarray) -> tuple:
    height, width = mask.shape
    total_pixels  = width * height

    lower    = mask[int(height * 0.5):, :]
    lower_h, lower_w = lower.shape

    left_pixels  = int(np.count_nonzero(lower[:, :lower_w // 2]))
    right_pixels = int(np.count_nonzero(lower[:, lower_w // 2:]))

    left_pct  = round(left_pixels  / total_pixels * 100, 3)
    right_pct = round(right_pixels / total_pixels * 100, 3)

    return left_pct, right_pct, left_pixels, right_pixels


def process_offline(target_dir: Path):
    meta_dir  = target_dir / "metadata"
    masks_dir = target_dir / "masks"

    if not masks_dir.exists():
        print(f"Error: {masks_dir} not found. Run Phase 1 first.")
        sys.exit(1)

    meta_files = list(meta_dir.glob("*.json"))
    print("=" * 60)
    print("PHASE 2: OFFLINE MASK PROCESSING (cardinal-aware, pano-aware)")
    print(f"Images to process: {len(meta_files)}")
    print("=" * 60)

    features = []
    n_pano = 0
    n_no_compass = 0
    MIN_PCT = 0.5  # minimum % of total pixels that must be sidewalk to count as detected

    for meta_file in meta_files:
        with open(meta_file, "r") as f:
            meta = json.load(f)

        img_id = meta.get("id")
        if not img_id:
            continue

        mask_path = masks_dir / f"{img_id}_mask.png"
        if not mask_path.exists():
            continue

        best_geom = meta.get("best_geometry") or meta.get("computed_geometry") or meta.get("geometry")
        lon, lat = 0.0, 0.0
        if best_geom and best_geom.get("type") == "Point":
            lon, lat = best_geom["coordinates"]
        if lon == 0.0 and lat == 0.0:
            continue

        compass_angle = meta.get("best_compass_angle") or meta.get("computed_compass_angle") or meta.get("compass_angle")
        is_pano   = bool(meta.get("is_pano", False))
        cam_type  = meta.get("camera_type", "perspective")

        if compass_angle is None:
            n_no_compass += 1

        try:
            mask = np.array(Image.open(mask_path).convert("L"))
        except Exception as e:
            print(f"  Failed to read mask {mask_path.name}: {e}")
            continue

        props = {
            "image_id":                img_id,
            "captured_at":             meta.get("captured_at", ""),
            "compass_angle":           meta.get("compass_angle"),
            "computed_compass_angle":  meta.get("computed_compass_angle"),
            "best_compass_angle":      compass_angle,
            "is_pano":                 is_pano,
            "camera_type":             cam_type,
            "vehicle_make":            meta.get("vehicle_make", "Unknown"),
            "vehicle_model":           meta.get("vehicle_model", "Unknown"),
            "image_url":               meta.get("thumb_original_url", meta.get("thumb_2048_url", "")),
            "image_total_pixels":      int(mask.shape[0] * mask.shape[1]),
        }

        if is_pano and compass_angle is not None:
            n_pano += 1
            sector_results = analyse_pano_mask(mask, compass_angle)
            props["pano_sector_coverage"] = sector_results
            l_bearing, r_bearing = cardinal_sides(compass_angle)
            props["cardinal_left_bearing"]  = l_bearing
            props["cardinal_right_bearing"] = r_bearing
            props["sidewalk_left"]         = "No"
            props["sidewalk_right"]        = "No"
            left_covs = []
            right_covs = []
            
            SECTOR_CENTRES = {
                "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
                "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0
            }
            
            for sec_name, sec_pct in sector_results.items():
                centre_bearing = SECTOR_CENTRES[sec_name]
                diff = angular_difference(compass_angle, centre_bearing)
                if diff < 0: # left of vehicle
                    left_covs.append(sec_pct)
                    if sec_pct > MIN_PCT:
                        props["sidewalk_left"] = "Yes"
                else: # right of vehicle
                    right_covs.append(sec_pct)
                    if sec_pct > MIN_PCT:
                        props["sidewalk_right"] = "Yes"
                        
            props["left_coverage_pct"]  = round(max(left_covs), 3) if left_covs else 0.0
            props["right_coverage_pct"] = round(max(right_covs), 3) if right_covs else 0.0

        elif not is_pano and compass_angle is not None:
            l_pct, r_pct, l_px, r_px = analyse_perspective_mask(mask)
            l_bearing, r_bearing = cardinal_sides(compass_angle)

            props["cardinal_left_bearing"]  = l_bearing
            props["cardinal_right_bearing"] = r_bearing
            props["sidewalk_left"]          = "Yes" if l_pct > MIN_PCT else "No"
            props["sidewalk_right"]         = "Yes" if r_pct > MIN_PCT else "No"
            props["left_coverage_pct"]      = l_pct
            props["right_coverage_pct"]     = r_pct
            props["left_sidewalk_pixels"]   = l_px
            props["right_sidewalk_pixels"]  = r_px
            props["pano_sector_coverage"]   = None

        else:
            props["cardinal_left_bearing"]  = None
            props["cardinal_right_bearing"] = None
            props["sidewalk_left"]          = "unknown"
            props["sidewalk_right"]         = "unknown"
            props["left_coverage_pct"]      = None
            props["right_coverage_pct"]     = None
            props["pano_sector_coverage"]   = None

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        })

    if not features:
        print("No valid points processed.")
        return

    out_file = target_dir / f"{target_dir.name}_phase2_points.geojson"
    with open(out_file, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

    print(f"\nPoints written:          {len(features)}")
    print(f"  Panoramic images:        {n_pano}")
    print(f"  No compass angle:        {n_no_compass}")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    args = parser.parse_args()
    process_offline(args.target_dir)
