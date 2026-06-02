#!/usr/bin/env python3
"""
Phase 3: Aggregate Point Detections onto OSM Road Centerlines (Topological)
Maps point detections to OSM street centerlines without geometric offsetting.
Supports 5m micro-chunking for high-resolution mapping.

Core aggregation logic restored from the verified old script, adapted to
single-line topological output (no parallel_offset).
"""

import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point as SPoint, LineString
from shapely.ops import substring
from shapely.strtree import STRtree
import osmnx as ox
from pyproj import Transformer

BUFFER_M          = 15.0
HEADING_TOL_DEG   = 40.0
MIN_IMAGES        = 3
CONFIDENCE_THRESH = 0.5
LOCAL_CRS         = 28992


def normalize_bearing(deg: float) -> float:
    return deg % 360

def bearing_between(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle bearing between two WGS84 points."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def angular_difference(a: float, b: float) -> float:
    """Signed angular difference: positive = clockwise from a."""
    diff = (b - a + 180) % 360 - 180
    return diff

def is_heading_parallel_to_road(heading: float, road_bearing: float, tol: float = 40.0) -> bool:
    """Check if image heading is parallel to road bearing (either direction)."""
    if heading is None or road_bearing is None:
        return False
    diff1 = abs(angular_difference(heading, road_bearing))
    diff2 = abs(angular_difference(heading, normalize_bearing(road_bearing + 180)))
    return (diff1 <= tol) or (diff2 <= tol)

def local_road_bearing_at_point(linestring, px: float, py: float) -> tuple:
    """Find projected distance and local bearing on the LineString."""
    pt = SPoint(px, py)
    dist_along = linestring.project(pt)
    dist_c = linestring.distance(pt)
    
    # Get local bearing by looking +/- 1 meter
    d1 = max(0, dist_along - 1.0)
    d2 = min(linestring.length, dist_along + 1.0)
    
    if d2 <= d1:
        # Too short, just use endpoints
        p1 = linestring.coords[0]
        p2 = linestring.coords[-1]
    else:
        p1 = linestring.interpolate(d1).coords[0]
        p2 = linestring.interpolate(d2).coords[0]
        
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    local_bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    return local_bearing, dist_along, dist_c

def get_physical_side(px: float, py: float, linestring, dist_along: float) -> str:
    """Determine if point is physically on the left or right side of the LineString."""
    p_line = linestring.interpolate(dist_along)
    d_fwd = min(linestring.length, dist_along + 0.1)
    
    if d_fwd <= dist_along:
        d_back = max(0.0, dist_along - 0.1)
        p_line = linestring.interpolate(d_back)
        p_fwd = linestring.interpolate(dist_along)
    else:
        p_fwd = linestring.interpolate(d_fwd)
        
    # Cross product
    # v1 = line forward vector
    v1x = p_fwd.x - p_line.x
    v1y = p_fwd.y - p_line.y
    
    # v2 = vector to point
    v2x = px - p_line.x
    v2y = py - p_line.y
    
    cross = v1x * v2y - v1y * v2x
    return "left" if cross > 0 else "right"

def assign_image_to_road_side(sector_bearing: float, road_bearing: float, phys_side: str, dist_from_center: float) -> str:
    """Assign an image detection (sector) to the logical left or right side of the road."""
    diff = angular_difference(road_bearing, sector_bearing)
    
    # If the camera is roughly looking forward along the road
    if abs(diff) < 90:
        return "left" if diff < 0 else "right"
    else:
        # Camera is looking backward relative to road vector
        return "right" if diff < 0 else "left"

def is_valid_osm_val(val):
    if val is None: return False
    if isinstance(val, float) and math.isnan(val): return False
    if val == "": return False
    return True

def build_side_props(sw_list, all_list, side_label: str) -> dict:
    n_total = len(all_list)
    n_sw = len(sw_list)
    
    if n_total == 0:
        quality = "no_data"
        present = "no_data"
        mean_cov = 0.0
        conf = 0.0
    else:
        if n_total < MIN_IMAGES:
            quality = "low_sample"
        elif n_total < MIN_IMAGES * 3:
            quality = "medium"
        else:
            quality = "high"
            
        mean_cov = round(float(np.mean(sw_list)), 2) if sw_list else 0.0
        
        # Calculate coverage-weighted confidence (0.0 to 1.0)
        # We give more weight to higher coverage detections
        total_weight = 0.0
        sw_weight = 0.0
        
        for cov in all_list:
            w = 1.0 + (cov / 10.0) # boost weight for higher coverage
            total_weight += w
        
        for cov in sw_list:
            w = 1.0 + (cov / 10.0)
            sw_weight += w
            
        conf = sw_weight / total_weight if total_weight > 0 else 0.0
        present = (conf >= CONFIDENCE_THRESH)

    return {
        f"sidewalk_{side_label}_present":   present,
        f"sidewalk_{side_label}_confidence": round(conf, 3),
        f"sidewalk_{side_label}_n_images":  n_total,
        f"sidewalk_{side_label}_n_detected": n_sw,
        f"sidewalk_{side_label}_mean_coverage":  mean_cov,
        f"sidewalk_{side_label}_data_quality":   quality,
    }

def load_points(geojson_path: Path) -> list:
    with open(geojson_path) as f:
        return json.load(f).get("features", [])

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global LOCAL_CRS
    
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir",  type=Path)
    parser.add_argument("--place",     type=str,  default=None)
    parser.add_argument("--osm-file",  type=Path, default=None)
    parser.add_argument("--buffer-m",  type=float, default=BUFFER_M)
    parser.add_argument("--heading-tol", type=float, default=HEADING_TOL_DEG)
    parser.add_argument("--chunk-size-m", type=float, default=0.0,
                        help="If > 0, splits roads into chunks (e.g. 5.0) before aggregating.")
    parser.add_argument("--local-crs", type=int, default=LOCAL_CRS,
                        help="EPSG code for local projection (e.g., 28992 for NL, 2249 for MA)")
    parser.add_argument("--output-prefix", type=str, default="api_output",
                        help="Prefix for output GeoJSON file")
    args = parser.parse_args()
    
    LOCAL_CRS = args.local_crs

    point_files = list(args.target_dir.glob("*_phase2_points.geojson"))
    if not point_files:
        print("Error: no *_phase2_points.geojson found.")
        return

    all_points = []
    for pf in point_files:
        all_points.extend(load_points(pf))
    print(f"Loaded {len(all_points)} image points from {args.target_dir.name}.")

    usable = [p for p in all_points if p["properties"].get("best_compass_angle") is not None]

    if args.osm_file and args.osm_file.exists():
        if args.osm_file.suffix == '.pkl':
            import pickle
            with open(args.osm_file, 'rb') as f:
                G = pickle.load(f)
            print(f"Loaded OSM graph from pickle: {args.osm_file.name}")
        else:
            G = ox.load_graphml(args.osm_file)
            print(f"Loaded OSM graph from graphml: {args.osm_file.name}")
    elif args.place:
        print(f"Downloading OSM graph for: {args.place}")
        G = ox.graph_from_place(args.place, network_type="all")
    else:
        print("Error: provide --place or --osm-file")
        return

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges = edges.to_crs(epsg=LOCAL_CRS)

    # Filter out non-vehicular paths to reduce parallel line clutter
    exclude_types = ["footway", "cycleway", "path", "pedestrian", "steps", "track", "corridor"]

    def is_excluded(hw):
        if hw is None: return False
        if isinstance(hw, list):
            return any(h in exclude_types for h in hw)
        return str(hw).lower() in exclude_types

    mask = edges["highway"].apply(lambda hw: not is_excluded(hw))
    edges = edges[mask].copy()
    print(f"Filtered graph down to {len(edges)} main road segments (removed cycleways/footpaths).")

    # Optional micro-chunking
    if args.chunk_size_m > 0:
        print(f"Splitting {len(edges)} roads into {args.chunk_size_m}m chunks...")
        chunked_edges = []
        for idx, row in edges.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            length = geom.length
            if length <= args.chunk_size_m:
                chunked_edges.append(row)
            else:
                for d in np.arange(0, length, args.chunk_size_m):
                    sub_geom = substring(geom, d, min(d + args.chunk_size_m, length))
                    new_row = row.copy()
                    new_row.geometry = sub_geom
                    chunked_edges.append(new_row)
        edges = gpd.GeoDataFrame(chunked_edges, crs=edges.crs)
        print(f"Resulted in {len(edges)} sub-segments.")

    tf_to_proj   = Transformer.from_crs("epsg:4326", f"epsg:{LOCAL_CRS}", always_xy=True)
    tf_to_wgs84  = Transformer.from_crs(f"epsg:{LOCAL_CRS}", "epsg:4326", always_xy=True)

    # Build point records (including pano_sector_coverage!)
    point_records = []
    points_geom = []
    for pt in usable:
        lon, lat = pt["geometry"]["coordinates"]
        x, y = tf_to_proj.transform(lon, lat)
        p = pt["properties"]
        point_records.append({
            "x": x, "y": y, "lon": lon, "lat": lat,
            "best_compass_angle": p["best_compass_angle"],
            "is_pano": p.get("is_pano", False),
            "pano_sector_coverage": p.get("pano_sector_coverage"),
            "cardinal_left_bearing": p.get("cardinal_left_bearing"),
            "cardinal_right_bearing": p.get("cardinal_right_bearing"),
            "sidewalk_left": p.get("sidewalk_left") == "Yes",
            "sidewalk_right": p.get("sidewalk_right") == "Yes",
            "left_coverage_pct": p.get("left_coverage_pct", 0.0),
            "right_coverage_pct": p.get("right_coverage_pct", 0.0),
        })
        points_geom.append(SPoint(x, y))

    print(f"Building spatial index for {len(points_geom)} points...")
    tree = STRtree(points_geom)

    print(f"Aggregating onto {len(edges)} road segments...")
    output_features = []
    
    for idx, row in edges.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
            
        # Get road endpoints to compute overall bearing (for output properties)
        coords_proj = list(geom.coords)
        if len(coords_proj) < 2:
            continue
            
        p_start_wgs84 = tf_to_wgs84.transform(coords_proj[0][0], coords_proj[0][1])
        p_end_wgs84   = tf_to_wgs84.transform(coords_proj[-1][0], coords_proj[-1][1])
        overall_bearing = bearing_between(p_start_wgs84[0], p_start_wgs84[1], p_end_wgs84[0], p_end_wgs84[1])

        coords_wgs84 = [tf_to_wgs84.transform(c[0], c[1]) for c in coords_proj]

        buffered = geom.buffer(args.buffer_m)
        possible_matches = tree.query(buffered)
        inside = [point_records[i] for i in possible_matches if buffered.contains(points_geom[i])]

        # NO CONTINUE ON EMPTY INSIDE

        # FILTER: Use LOCAL bearing for parallel check to correctly handle curved roads
        parallel_images = []
        for pr in inside:
            local_bearing, dist_along, dist_from_centerline = local_road_bearing_at_point(geom, pr["x"], pr["y"])
            dist_to_int = min(dist_along, geom.length - dist_along)
            
            # Bypass filter if we are within 5m of an intersection node
            is_parallel = is_heading_parallel_to_road(pr["best_compass_angle"], local_bearing, args.heading_tol)
            if is_parallel or dist_to_int < 5.0:
                pr_copy = pr.copy()
                pr_copy["_local_bearing"] = local_bearing  # Save to avoid recomputing
                pr_copy["_dist_along"] = dist_along
                pr_copy["_dist_from_centerline"] = dist_from_centerline
                pr_copy["_physical_side"] = get_physical_side(pr["x"], pr["y"], geom, dist_along)
                parallel_images.append(pr_copy)

        # NO CONTINUE ON EMPTY PARALLEL IMAGES

        road_left_sw,   road_left_all  = [], []
        road_right_sw,  road_right_all = [], []

        for pr in parallel_images:
            local_bearing = pr["_local_bearing"]
            phys_side = pr["_physical_side"]
            dist_c = pr["_dist_from_centerline"]

            # PANO branch: use 8-sector coverage data
            if pr["is_pano"] and pr.get("pano_sector_coverage"):
                sectors = pr["pano_sector_coverage"]
                SECTOR_CENTRES = {
                    "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
                    "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0,
                }
                for sec_name, sec_pct in sectors.items():
                    centre_bearing = SECTOR_CENTRES.get(sec_name, 0.0)
                    side = assign_image_to_road_side(centre_bearing, local_bearing, phys_side, dist_c)
                    has_sw = sec_pct > 0.5

                    if side == "left":
                        road_left_all.append(sec_pct)
                        if has_sw:
                            road_left_sw.append(sec_pct)
                    else:
                        road_right_all.append(sec_pct)
                        if has_sw:
                            road_right_sw.append(sec_pct)

            # PERSPECTIVE branch: use cardinal left/right bearings
            else:
                if pr["cardinal_left_bearing"] is not None:
                    side_l = assign_image_to_road_side(pr["cardinal_left_bearing"], local_bearing, phys_side, dist_c)
                    cov_l  = pr["left_coverage_pct"]
                    if side_l == "left":
                        road_left_all.append(cov_l)
                        if pr["sidewalk_left"]:
                            road_left_sw.append(cov_l)
                    else:
                        road_right_all.append(cov_l)
                        if pr["sidewalk_left"]:
                            road_right_sw.append(cov_l)

                if pr["cardinal_right_bearing"] is not None:
                    side_r = assign_image_to_road_side(pr["cardinal_right_bearing"], local_bearing, phys_side, dist_c)
                    cov_r  = pr["right_coverage_pct"]
                    if side_r == "left":
                        road_left_all.append(cov_r)
                        if pr["sidewalk_right"]:
                            road_left_sw.append(cov_r)
                    else:
                        road_right_all.append(cov_r)
                        if pr["sidewalk_right"]:
                            road_right_sw.append(cov_r)

        # Build output properties
        highway = row.get("highway", "unknown")
        if isinstance(highway, list):
            highway = highway[0]

        name_val = row.get("name", "")
        if not is_valid_osm_val(name_val):
            name_val = ""
        elif isinstance(name_val, list):
            name_val = name_val[0]

        shared_props = {
            "osm_name":            str(name_val),
            "osm_highway":         str(highway),
            "road_bearing":        round(overall_bearing, 1),
            "length_m":            round(geom.length, 1),
            "n_images_in_buffer":  len(inside),
            "n_images_parallel":   len(parallel_images),
        }

        left_props  = build_side_props(road_left_sw,  road_left_all,  "left")
        right_props = build_side_props(road_right_sw, road_right_all, "right")

        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords_wgs84},
            "properties": {**shared_props, **left_props, **right_props}
        }
        output_features.append(feature)

    suffix = f"_{int(args.chunk_size_m)}m" if args.chunk_size_m > 0 else ""
    out_file = args.target_dir / f"{args.output_prefix}_roads{suffix}.geojson"
    with open(out_file, "w") as f:
        json.dump({"type": "FeatureCollection", "features": output_features}, f, indent=2)

    print(f"\nPhase 3 complete! Centerlines written: {len(output_features)} to {out_file.name}")
    print(f"Saved to: {out_file}")

if __name__ == "__main__":
    main()
