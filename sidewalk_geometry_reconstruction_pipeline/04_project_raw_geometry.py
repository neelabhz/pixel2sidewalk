#!/usr/bin/env python3
"""
RAW Sidewalk Backprojection — No Filtering, No Constraints, No Post-Processing
===============================================================================
Same DA3 matrix math as 04_project_geometry.py, but with ALL filtering removed:
  - No road exclusion zones
  - No intersection exclusion zones
  - No density threshold
  - No confidence threshold
  - No depth clipping (beyond basic validity)
  - No post-processing (no polygon dedup, no tiny/thin removal)
  - No smoothing on polygon edges

Produces raw backprojection results to evaluate the unfiltered pipeline output.

Output:
  {section}_raw_polygons.geojson
  {section}_raw_centrelines.geojson
  {section}_raw_dots.geojson
  {section}_raw_map.html
"""
import os, sys, json, glob, pickle, argparse, warnings, math
import numpy as np, cv2
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union
warnings.filterwarnings("ignore")

# ── Coordinate helpers ────────────────────────────────────────────────────────

def meters_to_latlon(lat_ref, lon_ref, east_m, north_m):
    return lat_ref + north_m / 111320.0, lon_ref + east_m / (111320.0 * np.cos(np.radians(lat_ref)))

def latlon_to_meters(lat_ref, lon_ref, lat, lon):
    return (lon - lon_ref) * 111320.0 * np.cos(np.radians(lat_ref)), (lat - lat_ref) * 111320.0

# ── OSM helpers (only for road sampling — NO exclusion zones) ────────────────

def load_osm_graph(section_dir, bbox_wsen):
    cache_path = os.path.join(section_dir, "_osm_roads.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    try:
        import osmnx as ox
    except ImportError:
        return None
    west, south, east, north = bbox_wsen
    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive",
                           retain_all=False, simplify=True)
    with open(cache_path, "wb") as f:
        pickle.dump(G, f)
    return G

VIEW_ANGLES = {
    "front": (0.0, -20.0),
    "right": (90.0, -20.0),
    "back": (180.0, -20.0),
    "left": (270.0, -20.0)
}

class MetadataCache:
    def __init__(self, section_dir):
        self.section_dir = section_dir
        self.section_name = os.path.basename(os.path.normpath(section_dir))
        self.all_meta_path = os.path.join(section_dir, f"{self.section_name}_all_metadata.json")
        self.pano_dict = {}
        
        if os.path.exists(self.all_meta_path):
            try:
                print(f"  [Cache] Loading all metadata from {self.all_meta_path}...")
                with open(self.all_meta_path) as f:
                    data = json.load(f)
                for d in data:
                    if "id" in d:
                        self.pano_dict[str(d["id"])] = d
                print(f"  [Cache] Loaded {len(self.pano_dict)} panoramas into memory.")
            except Exception as e:
                print(f"  [Cache] Failed to load {self.all_meta_path}: {e}")

    def get_metadata(self, image_id):
        parts = image_id.split("_pano_")
        if len(parts) == 2:
            pano_id, view = parts[0], parts[1]
            if pano_id in self.pano_dict:
                pano_meta = self.pano_dict[pano_id]
                yaw, pitch = VIEW_ANGLES.get(view, (0.0, -20.0))
                return {
                    "computed_geometry": pano_meta.get("computed_geometry", {}),
                    "geometry": pano_meta.get("geometry", {}),
                    "computed_rotation": pano_meta.get("computed_rotation"),
                    "camera_parameters": pano_meta.get("camera_parameters") or [0.5, 0.0, 0.0],
                    "local_yaw": yaw,
                    "local_pitch": pitch,
                    "original_pano_id": pano_id
                }
        
        file_path = os.path.join(self.section_dir, "metadata", f"{image_id}_metadata.json")
        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

def bbox_from_metadata(section_dir):
    section_name = os.path.basename(os.path.normpath(section_dir))
    all_meta_path = os.path.join(section_dir, f"{section_name}_all_metadata.json")
    lats, lons = [], []
    
    if os.path.exists(all_meta_path):
        try:
            with open(all_meta_path) as f:
                data = json.load(f)
            for d in data:
                geom = d.get("computed_geometry", {}).get("coordinates") or d.get("geometry", {}).get("coordinates", [None, None])
                if geom[0] is not None:
                    lons.append(geom[0])
                    lats.append(geom[1])
        except Exception as e:
            print(f"  [Warning] Failed to load {all_meta_path}: {e}")
            
    if not lats:
        meta_files = glob.glob(os.path.join(section_dir, "metadata", "*_metadata.json"))
        if len(meta_files) > 1000:
            step = len(meta_files) // 1000
            meta_files = meta_files[::step]
        for mf in meta_files:
            try:
                with open(mf) as f:
                    meta = json.load(f)
                geom = meta.get("computed_geometry", {}).get("coordinates") or meta.get("geometry", {}).get("coordinates", [None, None])
                if geom[0] is not None:
                    lons.append(geom[0])
                    lats.append(geom[1])
            except Exception:
                pass
                
    if not lats: return None
    margin = 0.001
    return (min(lons) - margin, min(lats) - margin, max(lons) + margin, max(lats) + margin)

# ── RAW Point cloud generation (NO filtering beyond mask + basic depth > 0) ──

def get_base_points_raw(image_id, section_dir, depth_dir, ref_lat, ref_lon, meta_cache=None, subsample=5):
    """
    Project masked sidewalk pixels into 3D ENU metres using DA3 matrix math.
    RAW VERSION: No confidence filter, no depth range filter, no anomaly check.
    """
    # 1. Load Mask
    mask_path = os.path.join(section_dir, "segmentation", f"{image_id}_mask.png")
    if not os.path.exists(mask_path): return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or (mask > 127).sum() < 50:
        return None
    h_img, w_img = mask.shape

    # 2. Load Depth Map
    depth_path = os.path.join(section_dir, depth_dir, f"{image_id}.npy")
    if not os.path.exists(depth_path): return None
    depth = np.load(depth_path)
    
    if depth.shape != (h_img, w_img):
        depth = cv2.resize(depth, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    # 3. Load Metadata
    meta = None
    if meta_cache is not None:
        meta = meta_cache.get_metadata(image_id)
    if meta is None:
        mp = os.path.join(section_dir, "metadata", f"{image_id}_metadata.json")
        if not os.path.exists(mp): return None
        try:
            with open(mp) as f:
                meta = json.load(f)
        except Exception:
            return None
        
    geom = meta.get("computed_geometry", {}).get("coordinates", [None, None])
    cam_lon, cam_lat = geom[0], geom[1]
    if cam_lat is None: return None
    
    # A) BUILD INTRINSIC MATRIX (K)
    cam_params = meta.get("camera_parameters") or [0.5]
    focal_ratio = cam_params[0]
    fx = focal_ratio * max(w_img, h_img)
    fy = fx
    cx, cy = w_img / 2.0, h_img / 2.0
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # B) BUILD EXTRINSIC MATRIX (w2c → c2w)
    cam_east, cam_north = latlon_to_meters(ref_lat, ref_lon, cam_lat, cam_lon)
    C_world = np.array([cam_east, cam_north, 0.0], dtype=np.float64)

    axis_angle = np.array(meta.get("computed_rotation"), dtype=np.float64)
    R_world_to_pano, _ = cv2.Rodrigues(axis_angle)

    pitch = math.radians(meta.get("local_pitch", 0.0))
    yaw = math.radians(meta.get("local_yaw", 0.0))
    
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
    R_local = Ry @ Rx

    R_w2c = R_local @ R_world_to_pano
    t_w2c = -R_w2c @ C_world

    ext_44 = np.eye(4, dtype=np.float64)
    ext_44[:3, :3] = R_w2c
    ext_44[:3, 3] = t_w2c
    c2w = np.linalg.inv(ext_44)

    # C) DA3 OFFICIAL PROJECTION MATH
    if h_img > 1024:
        subsample = int(subsample * (h_img / 1024.0))

    rows = np.arange(0, h_img, subsample)
    cols = np.arange(0, w_img, subsample)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    rr, cc = rr.ravel(), cc.ravel()
    
    # Filter by mask only
    sw = mask[rr, cc] > 127
    rr, cc = rr[sw], cc[sw]
    if len(rr) < 5: return None

    d = depth[rr, cc].astype(np.float64)

    valid_depth = d > 0
    rr, cc, d = rr[valid_depth], cc[valid_depth], d[valid_depth]
    if len(rr) < 5: return None

    # 1. Homogeneous pixel coordinates
    pix = np.stack([cc, rr, np.ones_like(cc)], axis=1)
    
    # 2. Shoot rays in camera space
    rays = (K_inv @ pix.T)
    
    # 3. Scale by metric depth
    Xc = rays * d[None, :]
    
    # 4. Homogeneous
    Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
    
    # 5. Transform to world
    Xw = (c2w @ Xc_h)[:3, :].T

    east_full  = Xw[:, 0]
    north_full = Xw[:, 1]

    weights = np.ones(len(east_full))

    n = len(east_full)
    if n < 5: return None
    
    cam_east_arr = np.full(n, cam_east)
    cam_north_arr = np.full(n, cam_north)
    
    return np.stack([east_full, north_full, cam_east_arr, cam_north_arr, weights], axis=1)


# ── Road sampling ─────────────────────────────────────────────────────────────

def sample_road_every_n_metres(line_enu, step=5.0):
    coords = np.array(line_enu)
    n_nodes = len(coords)
    smoothed_tangents = []
    for i in range(n_nodes):
        prev  = coords[max(0, i - 1)]
        next_ = coords[min(n_nodes - 1, i + 1)]
        d = next_ - prev
        length = np.linalg.norm(d)
        smoothed_tangents.append(d / max(length, 1e-9))
    smoothed_tangents = np.array(smoothed_tangents)

    samples = []
    accumulated = 0.0
    seg_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    if seg_lengths.sum() < step: return []

    for i in range(n_nodes - 1):
        seg_len = seg_lengths[i]
        t = 0.0
        while t < seg_len:
            if accumulated >= step or len(samples) == 0:
                pos  = coords[i] + (t / max(seg_len, 1e-9)) * (coords[i+1] - coords[i])
                frac = t / max(seg_len, 1e-9)
                seg_dir = (1 - frac) * smoothed_tangents[i] + frac * smoothed_tangents[i+1]
                seg_dir = seg_dir / max(np.linalg.norm(seg_dir), 1e-9)
                perp_left = np.array([-seg_dir[1], seg_dir[0]])
                samples.append((pos, seg_dir.copy(), perp_left.copy()))
                accumulated = 0.0
            step_size = min(step - (accumulated % step), seg_len - t)
            accumulated += step_size
            t += step_size
    return samples

# ── RAW cross-section analysis (no density threshold, wider range) ───────────

def analyse_cross_section_raw(cross_pt, perp_dir, all_pts, max_cam_dist=30.0, along_half=5.0):
    """Same structure as original but with no density filter and wider lateral range."""
    east, north, cam_e, cam_n, w = (all_pts[:, i] for i in range(5))

    cam_dists = np.sqrt((cam_e - cross_pt[0])**2 + (cam_n - cross_pt[1])**2)
    local_mask = cam_dists <= max_cam_dist
    if local_mask.sum() < 3: return None

    east_l, north_l = east[local_mask], north[local_mask]
    w_l = w[local_mask]

    road_dir = np.array([perp_dir[1], -perp_dir[0]])
    rel_east  = east_l  - cross_pt[0]
    rel_north = north_l - cross_pt[1]
    along = rel_east * road_dir[0] + rel_north * road_dir[1]
    along_mask = np.abs(along) <= along_half
    if along_mask.sum() < 3: return None

    east_f  = east_l[along_mask]
    north_f = north_l[along_mask]
    w_f     = w_l[along_mask]

    rel_east2  = east_f  - cross_pt[0]
    rel_north2 = north_f - cross_pt[1]
    signed_lat = rel_east2 * perp_dir[0] + rel_north2 * perp_dir[1]

    result = {}
    for side, sign in [('left', 1), ('right', -1)]:
        zone = (signed_lat * sign >= 0.5) & (signed_lat * sign <= 40.0)
        if zone.sum() < 3:
            result[side] = None
            continue
        dists   = signed_lat[zone] * sign
        density = w_f[zone].sum()

        p10 = np.percentile(dists, 10)
        p95 = np.percentile(dists, 95)

        inner  = float(p10)
        outer  = float(max(p95, inner + 0.5))  # ensure at least 0.5m width
        centre = (inner + outer) / 2.0
        width  = outer - inner

        geo_side = get_geographic_side(perp_dir, sign)

        result[side] = {
            "inner": inner, "outer": outer,
            "centre": centre, "width": width,
            "n": int(zone.sum()), "density": float(density),
            "geo_side": geo_side,
        }
    return result

def get_geographic_side(perp_left, offset_sign):
    perp = perp_left * offset_sign
    if perp[1] >= 0:
        return "north-side" if abs(perp[1]) >= abs(perp[0]) else "east-side"
    else:
        return "south-side" if abs(perp[1]) >= abs(perp[0]) else "west-side"


# ── Process one section — RAW ────────────────────────────────────────────────

def process_section_raw(sec_dir, ref_lat, ref_lon, G, depth_dir, step_m=5.0, max_cam_dist=30.0):
    meta_cache = MetadataCache(sec_dir)
    
    # Enumerate from depth directory (much smaller than segmentation dir)
    # This avoids a slow glob on 16k+ mask files on network filesystems
    depth_path = os.path.join(sec_dir, depth_dir)
    depth_files = [f for f in os.listdir(depth_path) if f.endswith(".npy") and not f.endswith("_conf.npy")]
    image_ids = [os.path.splitext(f)[0] for f in depth_files]
    print(f"  [Scan] Found {len(image_ids)} depth maps in {depth_dir}/")

    all_chunks = []
    projected = 0
    for i, img_id in enumerate(image_ids):
        chunk = get_base_points_raw(img_id, sec_dir, depth_dir, ref_lat, ref_lon, meta_cache=meta_cache, subsample=5)
        if chunk is not None:
            all_chunks.append(chunk)
            projected += 1
        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}/{len(image_ids)} checked, {projected} projected")

    if not all_chunks: return [], [], []
    all_pts  = np.vstack(all_chunks)
    pts_east = all_pts[:, 0]
    pts_north= all_pts[:, 1]
    print(f"  [Points] {len(all_pts):,} RAW points stored.")

    poly_outputs, line_outputs, dot_outputs  = [], [], []

    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            lc = list(data['geometry'].coords)
            line_enu = [latlon_to_meters(ref_lat, ref_lon, lat, lon) for lon, lat in lc]
        else:
            lon1, lat1 = G.nodes[u]['x'], G.nodes[u]['y']
            lon2, lat2 = G.nodes[v]['x'], G.nodes[v]['y']
            line_enu = [latlon_to_meters(ref_lat, ref_lon, lat, lon)
                        for lat, lon in [(lat1, lon1), (lat2, lon2)]]

        line_arr = np.array(line_enu)

        bbox_pad = max_cam_dist + 15.0
        minE, maxE = line_arr[:, 0].min() - bbox_pad, line_arr[:, 0].max() + bbox_pad
        minN, maxN = line_arr[:, 1].min() - bbox_pad, line_arr[:, 1].max() + bbox_pad
        bbox_mask = ((pts_east  >= minE) & (pts_east  <= maxE) &
                     (pts_north >= minN) & (pts_north <= maxN))
        if bbox_mask.sum() < 5: continue
        local_pts = all_pts[bbox_mask]

        samples = sample_road_every_n_metres(line_enu, step=step_m)
        if not samples: continue

        slices = {"left": [None] * len(samples), "right": [None] * len(samples)}

        for i, (cross_pt, road_dir, perp_left) in enumerate(samples):
            res = analyse_cross_section_raw(cross_pt, perp_left, local_pts,
                                            max_cam_dist=max_cam_dist, along_half=step_m / 2)
            if res is None: continue

            for sign in [1, -1]:
                side_key = 'left' if sign == 1 else 'right'
                info = res.get(side_key)
                if info is None: continue

                perp      = perp_left * sign
                inner_pt  = cross_pt + info["inner"]  * perp
                outer_pt  = cross_pt + info["outer"]  * perp
                centre_pt = cross_pt + info["centre"] * perp

                geo_side = info.get("geo_side", "unknown")

                slices[side_key][i] = {
                    "inner": inner_pt, "outer": outer_pt, "centre": centre_pt,
                    "width": info["width"], "density": info["density"], "geo_side": geo_side,
                }

                dot_lat, dot_lon = meters_to_latlon(ref_lat, ref_lon, centre_pt[0], centre_pt[1])
                dot_outputs.append({
                    "lat": dot_lat, "lon": dot_lon, "side": geo_side,
                    "width_m": round(info["width"], 2), "density": round(info["density"], 2),
                })

        for side_key in ['left', 'right']:
            # Simple gap fill: just remove Nones, no interpolation
            sl = [s for s in slices[side_key] if s is not None]
            if len(sl) < 2: continue

            poly_geo_side = sl[len(sl) // 2].get("geo_side", side_key)
            inner_pts  = np.array([s["inner"]  for s in sl])
            outer_pts  = np.array([s["outer"]  for s in sl])
            centre_pts = np.array([s["centre"] for s in sl])

            ring = np.vstack([inner_pts, outer_pts[::-1], inner_pts[0:1]])
            if ring.shape[0] >= 4:
                try:
                    poly = Polygon(ring)
                    if poly.is_valid and not poly.is_empty:
                        poly_outputs.append((poly, poly_geo_side))
                except Exception: pass

            try:
                if len(centre_pts) >= 2:
                    line_outputs.append((LineString(centre_pts), poly_geo_side))
            except Exception: pass

    print(f"  [RAW] {len(poly_outputs)} polygons, {len(line_outputs)} centrelines, {len(dot_outputs)} dots")
    return poly_outputs, line_outputs, dot_outputs

# ── GeoJSON savers ────────────────────────────────────────────────────────────

def save_geojson(polygons, centrelines, dot_outputs, ref_lat, ref_lon,
                 poly_path, line_path, dot_path):
    def to_lonlat_poly(poly):
        return [[meters_to_latlon(ref_lat, ref_lon, e, n)[1], meters_to_latlon(ref_lat, ref_lon, e, n)[0]]
                for e, n in poly.exterior.coords]

    def to_lonlat_line(line):
        return [[meters_to_latlon(ref_lat, ref_lon, e, n)[1], meters_to_latlon(ref_lat, ref_lon, e, n)[0]]
                for e, n in line.coords]

    poly_feats = [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [to_lonlat_poly(p)]}, "properties": {"side": side}}
        for p, side in polygons if p.geom_type == 'Polygon'
    ]
    line_feats = [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": to_lonlat_line(l)}, "properties": {"side": side}}
        for l, side in centrelines if l.geom_type == 'LineString'
    ]
    dot_feats = [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
         "properties": {"side": d["side"], "width_m": d["width_m"], "density": d["density"]}}
        for d in dot_outputs
    ]
    for feats, path in [(poly_feats, poly_path), (line_feats, line_path), (dot_feats, dot_path)]:
        with open(path, "w") as f: json.dump({"type": "FeatureCollection", "features": feats}, f)
        print(f"  Saved: {path}")

# ── Interactive map ───────────────────────────────────────────────────────────

def create_map(polygons, centrelines, dot_outputs, ref_lat, ref_lon,
               center_lat, center_lon, output_html):
    import folium
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False
    ).add_to(m)

    layers = {
        "ne_poly": folium.FeatureGroup(name="Polygons — North/East",     show=True),
        "sw_poly": folium.FeatureGroup(name="Polygons — South/West",     show=True),
        "ne_cl":   folium.FeatureGroup(name="Centrelines — North/East",  show=True),
        "sw_cl":   folium.FeatureGroup(name="Centrelines — South/West",  show=True),
        "ne_dot":  folium.FeatureGroup(name="Width Dots — North/East",   show=True),
        "sw_dot":  folium.FeatureGroup(name="Width Dots — South/West",   show=True),
    }

    def group_color(side, cl=False):
        if side in ("north-side", "east-side"): return "ne", "#1abc9c" if cl else "#27ae60"
        return "sw", "#e67e22" if cl else "#e74c3c"

    def to_latlons_poly(poly): return [meters_to_latlon(ref_lat, ref_lon, e, n) for e, n in poly.exterior.coords]
    def to_latlons_line(line): return [meters_to_latlon(ref_lat, ref_lon, e, n) for e, n in line.coords]

    for poly, side in polygons:
        g, c = group_color(side)
        if poly.geom_type == 'Polygon':
            folium.Polygon(locations=to_latlons_poly(poly), color=c, fill_color=c, fill_opacity=0.35, weight=1).add_to(layers[f"{g}_poly"])

    for line, side in centrelines:
        g, c = group_color(side, cl=True)
        if line.geom_type == 'LineString':
            folium.PolyLine(locations=to_latlons_line(line), color=c, weight=2.5, opacity=0.9).add_to(layers[f"{g}_cl"])

    for d in dot_outputs:
        g, c = group_color(d["side"])
        folium.Circle(
            location=[d["lat"], d["lon"]], radius=d["width_m"] / 2.0, color=c, fill=True, fill_color=c, fill_opacity=0.7,
            tooltip=f"{d['side']} | width: {d['width_m']:.1f} m | score: {d['density']:.1f}"
        ).add_to(layers[f"{g}_dot"])

    for grp in layers.values(): grp.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"  Map saved: {output_html}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAW backprojection — no filtering/post-processing")
    parser.add_argument("--sections",    nargs='+', required=True)
    parser.add_argument("--depth-dir",   type=str, default="dav3_depth")
    parser.add_argument("--step",        type=float, default=5.0)
    parser.add_argument("--cam-dist",    type=float, default=30.0)
    args = parser.parse_args()

    ref_lat, ref_lon = None, None

    for sec_dir in args.sections:
        ref_file = os.path.join(sec_dir, "enu_reference.json")
        if os.path.exists(ref_file):
            try:
                with open(ref_file) as f:
                    d = json.load(f)
                ref_lat = d["ref_lat"]
                ref_lon = d["ref_lon"]
                print(f"  [ENU] Loaded reference origin from {ref_file}: "
                      f"lat={ref_lat:.6f}, lon={ref_lon:.6f}")
                break
            except Exception as e:
                print(f"  [ENU] Warning: could not read {ref_file}: {e}")

    if ref_lat is None:
        print("  [ENU] WARNING: enu_reference.json not found — falling back to bbox centroid.")
        for sec_dir in args.sections:
            if os.path.isdir(sec_dir):
                bbox = bbox_from_metadata(sec_dir)
                if bbox:
                    ref_lon, ref_lat = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
                    break

    if ref_lat is None:
        print("ERROR: No valid metadata found in any section."); sys.exit(1)
    
    all_polys, all_lines, all_dots = [], [], []

    for sec_dir in args.sections:
        if not os.path.isdir(sec_dir): continue
        bbox = bbox_from_metadata(sec_dir)
        if not bbox: continue
        print(f"\n=== {os.path.basename(sec_dir)} (RAW — no filtering) ===")
        G = load_osm_graph(sec_dir, bbox)
        polys, lines, dots = process_section_raw(
            sec_dir, ref_lat, ref_lon, G, args.depth_dir,
            step_m=args.step, max_cam_dist=args.cam_dist
        )
        all_polys.extend(polys)
        all_lines.extend(lines)
        all_dots.extend(dots)

    section_names = [os.path.basename(s) for s in args.sections if os.path.isdir(s)]
    prefix = section_names[0] if len(section_names) == 1 else "combined"
    
    # Save with _raw suffix
    out_dir = os.path.dirname(os.path.abspath(args.sections[0]))
    
    save_geojson(all_polys, all_lines, all_dots, ref_lat, ref_lon,
                 os.path.join(out_dir, prefix, f"{prefix}_raw_polygons.geojson"),
                 os.path.join(out_dir, prefix, f"{prefix}_raw_centrelines.geojson"),
                 os.path.join(out_dir, prefix, f"{prefix}_raw_dots.geojson"))
    create_map(all_polys, all_lines, all_dots, ref_lat, ref_lon, ref_lat, ref_lon,
               os.path.join(out_dir, prefix, f"{prefix}_raw_map.html"))
    print(f"\nDONE (RAW). Output prefix: {prefix}_raw")

if __name__ == "__main__":
    main()
