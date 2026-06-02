#!/usr/bin/env python3
"""
Phase 3: DA3 Pose-Conditioned Metric Depth — Sequence-Window Approach
======================================================================
Uses DA3NESTED-GIANT-LARGE-1.1 with pose-conditioned inference.

DESIGN
------
Images are grouped by their Mapillary `sequence` field and sorted by
`captured_at` timestamp. A sliding window of WINDOW_SIZE frames is moved
along each sequence. For each target frame that has a segmentation mask,
DA3 is called with a window of WINDOW_SIZE frames centred on the target.

This gives DA3 real spatial parallax (frames 2–5 m apart along the street)
which enables cross-view consistency and metric scale estimation via
align_to_input_ext_scale=True.

HANDLES BOTH IMAGE TYPES
------------------------
Perspective images:
  Used directly as single frames. Focal length from camera_parameters[0].

Panoramic images (equirectangular/spherical):
  Each panorama contributes its 4 perspective crops (front/right/back/left)
  as individual frames. All 4 crops share the same sequence and captured_at.

  For each crop that needs depth, the window is built as:
    [front_crop_of_prev_pano, TARGET_CROP, front_crop_of_next_pano, ...]
  This gives 3 different physical positions (real parallax) plus the target
  crop's specific view direction.

OUTPUT
------
For every frame with a mask:
  dav3_depth/{image_id}.npy        — metric depth map (H, W), float32, metres
  dav3_depth/{image_id}_conf.npy   — confidence map (H, W), float32 (if available)

Also saves:
  enu_reference.json  — ENU origin used for extrinsics (must match Phase 4)
"""

import os
import sys
import json
import glob
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import defaultdict
import argparse
import math

from depth_anything_3.api import DepthAnything3

# ─── Constants ────────────────────────────────────────────────────────────────

PANO_VIEWS   = ["front", "right", "back", "left"]
PANO_YAWS    = {"front": 0, "right": 90, "back": 180, "left": 270}
PANO_PITCH   = -20.0   # degrees downward tilt applied in Phase 0

# ─── Coordinate helpers ───────────────────────────────────────────────────────

def rodrigues_to_matrix(rvec):
    rvec = np.array(rvec, dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    return R

def latlon_to_enu(lat, lon, ref_lat, ref_lon):
    """GPS → ENU metres relative to (ref_lat, ref_lon)."""
    R_EARTH = 6378137.0
    north = math.radians(lat - ref_lat) * R_EARTH
    east  = math.radians(lon - ref_lon) * R_EARTH * math.cos(math.radians(ref_lat))
    return np.array([east, north, 0.0], dtype=np.float64)

def local_rotation_matrix(yaw_deg, pitch_deg):
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    Ry = np.array([[math.cos(yaw),  0, math.sin(yaw)],
                   [0,              1, 0             ],
                   [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float64)
    Rx = np.array([[1, 0,               0             ],
                   [0, math.cos(pitch), math.sin(pitch)],
                   [0, -math.sin(pitch), math.cos(pitch)]], dtype=np.float64)
    return Ry @ Rx

def build_w2c(R_sfm, t_enu, yaw_deg=0.0, pitch_deg=0.0):
    """
    4×4 world-to-camera matrix.
    R_sfm  : 3×3 from Mapillary computed_rotation (world → pano camera frame)
    t_enu  : camera position in ENU metres
    yaw/pitch: local crop tilt (0/0 for native perspective images)
    """
    R_local = local_rotation_matrix(yaw_deg, pitch_deg)
    R_total = R_local @ R_sfm
    t_cam   = -R_total @ t_enu
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R_total
    E[:3,  3] = t_cam
    return E

def build_K(meta, is_pano_crop):
    """
    3×3 intrinsic matrix.
    Pano crops : FOV=90°, always 1024×1024 → f=512
    Perspective: focal_ratio from camera_parameters[0]
    """
    W = meta.get("width",  1024)
    H = meta.get("height", 1024)
    if is_pano_crop:
        f = (W / 2.0) / math.tan(math.radians(45.0))   # FOV=90° → tan(45°)=1
    else:
        cam_params = meta.get("camera_parameters") or [0.5, 0.0, 0.0]
        f = cam_params[0] * max(W, H)
    return np.array([[f, 0, W/2.0],
                     [0, f, H/2.0],
                     [0, 0, 1.0  ]], dtype=np.float64)

# ─── Frame object ─────────────────────────────────────────────────────────────

class Frame:
    """One image that can be passed to DA3."""
    __slots__ = ['image_id', 'img_path', 'meta', 'is_pano_crop',
                 'sequence', 'captured_at', 'has_mask', 'pano_id']

    def __init__(self, image_id, img_path, meta, is_pano_crop,
                 sequence, captured_at, has_mask, pano_id=None):
        self.image_id     = image_id
        self.img_path     = img_path
        self.meta         = meta
        self.is_pano_crop = is_pano_crop
        self.sequence     = sequence
        self.captured_at  = captured_at
        self.has_mask     = has_mask
        self.pano_id      = pano_id   # set for pano crops; None for perspective

    def get_pose(self, ref_lat, ref_lon):
        """Return (E_w2c 4×4, K 3×3) or raise ValueError."""
        meta = self.meta
        coords = (meta.get("computed_geometry", {}).get("coordinates") or
                  meta.get("geometry", {}).get("coordinates"))
        if not coords:
            raise ValueError(f"No GPS for {self.image_id}")
        rot_vec = meta.get("computed_rotation")
        if rot_vec is None:
            raise ValueError(f"No computed_rotation for {self.image_id}")

        t_enu = latlon_to_enu(coords[1], coords[0], ref_lat, ref_lon)
        R_sfm = rodrigues_to_matrix(rot_vec)
        yaw   = float(meta.get("local_yaw",   0.0))
        pitch = float(meta.get("local_pitch", 0.0))

        E = build_w2c(R_sfm, t_enu, yaw, pitch)
        K = build_K(meta, self.is_pano_crop)
        return E, K

# ─── Build frame list ─────────────────────────────────────────────────────────

def build_frame_list(section_dir, img_dir, meta_dir, seg_dir):
    """
    Returns a list of Frame objects sorted by (sequence, captured_at).

    Panoramas → 4 crop Frames each (front/right/back/left)
    Perspective → 1 Frame each
    """
    all_meta_path = section_dir / f"{section_dir.name}_all_metadata.json"

    pano_metas  = {}   # pano_id  → raw metadata dict
    persp_metas = {}   # image_id → raw metadata dict

    if all_meta_path.exists():
        with open(all_meta_path) as f:
            data = json.load(f)
        for d in data:
            is_pano = bool(d.get("is_pano")) or \
                      d.get("camera_type") in ("spherical", "equirectangular")
            if is_pano:
                pano_metas[str(d["id"])] = d
            else:
                persp_metas[str(d["id"])] = d
    else:
        # Fallback: scan individual metadata files
        for mf in sorted(meta_dir.glob("*_metadata.json")):
            try:
                with open(mf) as f:
                    d = json.load(f)
                is_pano = bool(d.get("is_pano")) or \
                          d.get("camera_type") in ("spherical", "equirectangular")
                if is_pano:
                    pano_metas[str(d["id"])] = d
                else:
                    persp_metas[str(d["id"])] = d
            except Exception:
                pass

    frames = []

    # ── Panorama crops ────────────────────────────────────────────────────────
    for pano_id, pmeta in pano_metas.items():
        seq = pmeta.get("sequence", "unknown")
        ts  = pmeta.get("captured_at", 0)

        for view in PANO_VIEWS:
            crop_id  = f"{pano_id}_pano_{view}"
            img_path = img_dir / f"{crop_id}.jpg"
            if not img_path.exists():
                continue

            # Prefer the individual crop metadata file (has local_yaw/pitch)
            crop_meta_path = meta_dir / f"{crop_id}_metadata.json"
            if crop_meta_path.exists():
                with open(crop_meta_path) as f:
                    crop_meta = json.load(f)
            else:
                # Reconstruct from panorama metadata
                crop_meta = dict(pmeta)
                crop_meta["id"]                = crop_id
                crop_meta["local_yaw"]         = PANO_YAWS[view]
                crop_meta["local_pitch"]       = PANO_PITCH
                crop_meta["width"]             = 1024
                crop_meta["height"]            = 1024
                crop_meta["camera_parameters"] = [0.5, 0.0, 0.0]
                crop_meta["camera_type"]       = "perspective"

            has_mask = (seg_dir / f"{crop_id}_mask.png").exists()
            frames.append(Frame(
                image_id    = crop_id,
                img_path    = img_path,
                meta        = crop_meta,
                is_pano_crop = True,
                sequence    = seq,
                captured_at = ts,
                has_mask    = has_mask,
                pano_id     = pano_id,
            ))

    # ── Native perspective images ─────────────────────────────────────────────
    for img_id, pmeta in persp_metas.items():
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        meta_path = meta_dir / f"{img_id}_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = pmeta

        has_mask = (seg_dir / f"{img_id}_mask.png").exists()
        seq = meta.get("sequence", "unknown")
        ts  = meta.get("captured_at", 0)

        frames.append(Frame(
            image_id    = img_id,
            img_path    = img_path,
            meta        = meta,
            is_pano_crop = False,
            sequence    = seq,
            captured_at = ts,
            has_mask    = has_mask,
        ))

    # Sort: sequence first, then time
    frames.sort(key=lambda f: (f.sequence, f.captured_at, f.image_id))
    return frames

# ─── Build context window for a target frame ──────────────────────────────────

def build_window(target_frame, seq_frames, window_size=4):
    """
    Build a window of `window_size` frames for DA3 inference.

    For PERSPECTIVE targets:
      Use consecutive frames from the same sequence (real parallax).

    For PANORAMA CROP targets:
      Use the target crop + front crops of neighbouring panoramas.
      This gives:
        - Real parallax from different pano positions
        - The target crop's specific view direction
      Window = [front_of_prev_pano, ..., TARGET_CROP, ..., front_of_next_pano, ...]

    Returns (window_frames, centre_index_in_window).
    """
    n = len(seq_frames)
    target_idx = next((i for i, f in enumerate(seq_frames)
                       if f.image_id == target_frame.image_id), None)
    if target_idx is None:
        return [target_frame], 0

    if not target_frame.is_pano_crop:
        # ── Perspective: simple sliding window ───────────────────────────────
        half  = window_size // 2
        start = max(0, target_idx - half)
        end   = min(n, start + window_size)
        start = max(0, end - window_size)
        window = seq_frames[start:end]
        centre_idx = target_idx - start
        return window, centre_idx

    else:
        # ── Panorama crop: use front crops of neighbours + target crop ────────
        # Collect unique pano positions in this sequence (by pano_id)
        pano_order = []
        seen = set()
        for f in seq_frames:
            pid = f.pano_id or f.image_id
            if pid not in seen:
                seen.add(pid)
                pano_order.append(pid)

        target_pano = target_frame.pano_id
        try:
            pano_idx = pano_order.index(target_pano)
        except ValueError:
            return [target_frame], 0

        # Pick front crops of neighbouring panoramas as context
        half = window_size // 2
        pano_start = max(0, pano_idx - half)
        pano_end   = min(len(pano_order), pano_start + window_size)
        pano_start = max(0, pano_end - window_size)
        neighbour_pano_ids = pano_order[pano_start:pano_end]

        # Build window: front crop of each neighbour, but TARGET crop for target pano
        window = []
        centre_idx = 0
        # Map pano_id → front frame
        front_map = {f.pano_id: f for f in seq_frames
                     if f.is_pano_crop and f.image_id.endswith("_pano_front")}

        for i, pid in enumerate(neighbour_pano_ids):
            if pid == target_pano:
                window.append(target_frame)
                centre_idx = i
            else:
                # Use front crop of this neighbour as context
                ctx = front_map.get(pid)
                if ctx is not None:
                    window.append(ctx)
                # If front crop not available, skip this neighbour
                # (window may be smaller than window_size — that's fine)

        if not window:
            return [target_frame], 0

        return window, centre_idx

# ─── Main processing ──────────────────────────────────────────────────────────

def process_section_depth(section_dir, window_size=4):
    section_dir = Path(section_dir)
    img_dir  = section_dir / "images"
    meta_dir = section_dir / "metadata"
    seg_dir  = section_dir / "segmentation"
    out_dir  = section_dir / "dav3_depth"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DA3NESTED-GIANT-LARGE-1.1 on {device}...", flush=True)
    model_path = "../models/DA3NESTED-GIANT-LARGE-1.1"
    model = DepthAnything3.from_pretrained(model_path).to(device)
    print("Model loaded!", flush=True)

    # ── Build frame list ──────────────────────────────────────────────────────
    print("Building frame list...", flush=True)
    frames = build_frame_list(section_dir, img_dir, meta_dir, seg_dir)

    n_pano  = sum(1 for f in frames if f.is_pano_crop)
    n_persp = sum(1 for f in frames if not f.is_pano_crop)
    n_mask  = sum(1 for f in frames if f.has_mask)
    print(f"  Panorama crops : {n_pano}", flush=True)
    print(f"  Perspective    : {n_persp}", flush=True)
    print(f"  Total frames   : {len(frames)}", flush=True)
    print(f"  With masks     : {n_mask}  (these will get depth maps)", flush=True)

    if not frames:
        print("ERROR: No frames found.", flush=True)
        sys.exit(1)

    # ── ENU reference origin ──────────────────────────────────────────────────
    ref_lat, ref_lon = None, None
    for f in frames:
        coords = (f.meta.get("computed_geometry", {}).get("coordinates") or
                  f.meta.get("geometry", {}).get("coordinates"))
        if coords:
            ref_lat, ref_lon = coords[1], coords[0]
            break

    if ref_lat is None:
        print("ERROR: No GPS reference found.", flush=True)
        sys.exit(1)

    ref_file = section_dir / "enu_reference.json"
    with open(ref_file, "w") as f:
        json.dump({"ref_lat": ref_lat, "ref_lon": ref_lon}, f, indent=2)
    print(f"ENU reference: lat={ref_lat:.6f}, lon={ref_lon:.6f}  → saved to {ref_file}",
          flush=True)

    # ── Group frames by sequence ──────────────────────────────────────────────
    seq_groups = defaultdict(list)
    for f in frames:
        seq_groups[f.sequence].append(f)

    # ── Process every frame that has a mask ───────────────────────────────────
    processed       = 0
    skipped_done    = 0
    skipped_no_mask = 0
    skipped_error   = 0
    total_with_mask = n_mask

    for frame_idx, target in enumerate(frames):

        if not target.has_mask:
            skipped_no_mask += 1
            continue

        out_path = out_dir / f"{target.image_id}.npy"
        if out_path.exists():
            skipped_done += 1
            continue

        seq_frames = seq_groups[target.sequence]

        # Build window: target + context neighbours
        window, centre_idx = build_window(target, seq_frames, window_size)

        # Build poses for all frames in window
        image_paths = []
        extrinsics  = []
        intrinsics  = []
        valid = True

        for frame in window:
            try:
                E, K = frame.get_pose(ref_lat, ref_lon)
            except ValueError as e:
                print(f"  [Skip] {e}", flush=True)
                valid = False
                break
            image_paths.append(str(frame.img_path))
            extrinsics.append(E)
            intrinsics.append(K)

        if not valid or len(image_paths) == 0:
            skipped_error += 1
            continue

        extrinsics_np = np.stack(extrinsics).astype(np.float32)  # (N, 4, 4)
        intrinsics_np = np.stack(intrinsics).astype(np.float32)  # (N, 3, 3)

        try:
            with torch.no_grad():
                prediction = model.inference(
                    image      = image_paths,
                    extrinsics = extrinsics_np,
                    intrinsics = intrinsics_np,
                    # align_to_input_ext_scale=True (default):
                    # rescales depth to metric scale of our ENU extrinsics (metres)
                )

            depths = prediction.depth   # (N, H, W)
            np.save(out_path, depths[centre_idx])

            # Save confidence map if available
            confs = getattr(prediction, 'conf', None)
            if confs is not None:
                np.save(out_dir / f"{target.image_id}_conf.npy",
                        confs[centre_idx])

            processed += 1
            if processed % 50 == 0 or processed <= 3:
                pct = 100.0 * processed / max(total_with_mask, 1)
                N   = len(image_paths)
                print(f"  [{processed}/{total_with_mask} {pct:.1f}%] "
                      f"{target.image_id}  "
                      f"({'pano' if target.is_pano_crop else 'persp'}, "
                      f"window N={N}, centre={centre_idx})",
                      flush=True)

        except Exception as e:
            print(f"  [Error] {target.image_id}: {e}", flush=True)
            skipped_error += 1

    print(f"\nDone!", flush=True)
    print(f"  Depth maps saved : {processed}", flush=True)
    print(f"  Already done     : {skipped_done}", flush=True)
    print(f"  No mask (skipped): {skipped_no_mask}", flush=True)
    print(f"  Errors           : {skipped_error}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence-based DA3 pose-conditioned depth inference"
    )
    parser.add_argument("section_dir", type=Path,
                        help="Section directory (e.g. ../boston_backbay)")
    parser.add_argument("--window-size", type=int, default=4,
                        help="Frames per DA3 inference call (default: 4). "
                             "Higher = more context but more VRAM.")
    args = parser.parse_args()
    process_section_depth(args.section_dir, window_size=args.window_size)
