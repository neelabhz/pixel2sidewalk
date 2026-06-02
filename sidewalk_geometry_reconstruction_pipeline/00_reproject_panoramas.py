#!/usr/bin/env python3
"""
Panorama Reprojector (High-Speed I/O Optimized)
====================
Converts 360° equirectangular panoramas into 4 perspective (pinhole) images:
Front (0°), Right (90°), Back (180°), Left (270°).
"""

import os
import json
import glob
import math
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import cv2

def extract_tilted_perspective(equi_img, out_w, out_h, fov_deg, yaw_deg, pitch_deg):
    eq_h, eq_w = equi_img.shape[:2]
    
    f = (out_w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    cx, cy = out_w / 2.0, out_h / 2.0
    
    x, y = np.meshgrid(np.arange(out_w), np.arange(out_h))
    
    X_cam = x - cx
    Y_cam = cy - y 
    Z_cam = np.full_like(X_cam, f)
    
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), math.sin(pitch)],
        [0, -math.sin(pitch), math.cos(pitch)]
    ])
    Ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])
    R = Ry @ Rx
    
    rays = np.stack([X_cam, Y_cam, Z_cam], axis=-1) 
    rays_rot = np.einsum('ij,hwj->hwi', R, rays)
    
    r = np.linalg.norm(rays_rot, axis=-1)
    lat = np.arcsin(rays_rot[..., 1] / r)
    lon = np.arctan2(rays_rot[..., 0], rays_rot[..., 2])
    
    map_x = (lon / np.pi + 1.0) * (eq_w / 2.0)
    map_y = (-lat / (np.pi / 2.0) + 1.0) * (eq_h / 2.0)
    
    persp_img = cv2.remap(equi_img, map_x.astype(np.float32), map_y.astype(np.float32), 
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return persp_img, f

def process_panorama(img_path, meta_path, out_img_dir, out_meta_dir, out_w=1024, out_h=1024, fov=90):
    img_id = os.path.basename(img_path).split('.')[0]
    
    with open(meta_path) as f:
        meta = json.load(f)
        
    if meta.get("camera_type") not in ["equirectangular", "spherical"]:
        return 0 
        
    img = cv2.imread(img_path)
    if img is None: return 0
        
    base_compass = meta.get("computed_compass_angle", meta.get("compass_angle", 0))
    views = [("front", 0, -20), ("right", 90, -20), ("back", 180, -20), ("left", 270, -20)]
    
    for view_name, yaw_deg, pitch_deg in views:
        view_img, f_px = extract_tilted_perspective(img, out_w, out_h, fov, yaw_deg, pitch_deg)
        out_name = f"{img_id}_pano_{view_name}"
        cv2.imwrite(os.path.join(out_img_dir, f"{out_name}.jpg"), view_img)
        
        focal_ratio = f_px / max(out_w, out_h)
        view_meta = meta.copy()
        
        view_meta["camera_type"] = "perspective"
        view_meta["width"] = out_w
        view_meta["height"] = out_h
        view_meta["camera_parameters"] = [focal_ratio, 0.0, 0.0]
        
        view_compass = (base_compass + yaw_deg) % 360
        if "computed_compass_angle" in view_meta: view_meta["computed_compass_angle"] = view_compass
        if "compass_angle" in view_meta: view_meta["compass_angle"] = view_compass
            
        view_meta["local_yaw"] = yaw_deg
        view_meta["local_pitch"] = pitch_deg
        view_meta["id"] = out_name
        view_meta["original_pano_id"] = img_id
        
        with open(os.path.join(out_meta_dir, f"{out_name}_metadata.json"), 'w') as f:
            json.dump(view_meta, f, indent=2)
            
    os.rename(img_path, img_path + ".equi")
    os.rename(meta_path, meta_path + ".equi")
    return 4

def check_metadata(mf, img_dir):
    """Worker function to quickly read JSON files in parallel."""
    try:
        with open(mf) as f:
            meta = json.load(f)
        if meta.get("camera_type") in ["equirectangular", "spherical"]:
            img_id = meta["id"]
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                return (img_path, mf)
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    section_dir = args.section
    img_dir = os.path.join(section_dir, "images")
    meta_dir = os.path.join(section_dir, "metadata")
    
    meta_files = glob.glob(os.path.join(meta_dir, "*_metadata.json"))
    pano_tasks = []
    
    print(f"Scanning {len(meta_files)} metadata files (using 32 threads to bypass scratch drive latency)...")
    
    # Use ThreadPoolExecutor to blast through the tiny JSON files simultaneously
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(check_metadata, mf, img_dir): mf for mf in meta_files}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                pano_tasks.append(result)
            if (i + 1) % 2000 == 0:
                print(f"  Scanned {i+1}/{len(meta_files)} files...")
                
    print(f"Found {len(pano_tasks)} panoramas out of {len(meta_files)} total images")
    if not pano_tasks: return
        
    print(f"Extracting 4 perspective views per panorama (Total: {len(pano_tasks)*4} new images)...")
    
    total_new = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_panorama, img_p, meta_p, img_dir, meta_dir): img_p 
            for img_p, meta_p in pano_tasks
        }
        for i, future in enumerate(as_completed(futures)):
            try: total_new += future.result()
            except Exception as e: pass
            if (i + 1) % 50 == 0:
                print(f"  Chopped {i+1}/{len(pano_tasks)} panoramas...")
                
    print(f"Done! Generated {total_new} perspective views.")

if __name__ == "__main__":
    main()