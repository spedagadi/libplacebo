#!/usr/bin/env python
"""
Extract per-frame DV + HDR10+ metadata from a Dolby Vision file for ML dataset construction.

Handles DV profiles and source formats:
  Profile 5 (pure DV, single layer):      RPU embedded in main stream v:0.
  Profile 7 (dual layer BL+EL):
    - MKV / ISO interleaved:  BL in v:0, EL (RPU) also accessible via v:0 or v:1.
    - BDMV folder:            BL = largest m2ts, EL = second-largest m2ts (separate file).
  Profile 8 (HDR10-compatible, single):   RPU in v:0; HDR10+ side-data carries features.

Source formats accepted:
  - Single file: MKV, MP4, m2ts  (pass file path)
  - BDMV folder: pass disc root or BDMV parent (auto-discovers BL + EL streams)
  - Mounted ISO: pass mount-point folder (treated as BDMV folder)

Per-frame features (inputs):
  - maxscl, average_maxrgb, fraction_bright_pixels
  - distrib_val_0..8: PQ luma percentiles
  - zone_mean/max_rR_cC: 3x3 SAT spatial grid

Per-frame targets (tone curve):
  Profile 8 - HDR10+: knee_x, knee_y, 9 bezier anchors
  Profile 5/7 - DV RPU Y-component: up to 8 pivot segments, each quadratic polynomial

Usage:
  python dv_metadata_extract.py INPUT.mkv -o dataset.csv --sample-fps 1
  python dv_metadata_extract.py "G:/Dune.Part.Two.2024.COMPLETE.UHD" -o dune.csv
  python dv_metadata_extract.py INPUT.mkv -o dataset.csv --sample-fps 0.5 --resume
"""

import subprocess
import json
import csv
import sys
import time
import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np

FFPROBE = "ffprobe"
FFMPEG  = "ffmpeg"
CHUNK_SECS = 60

# Downscale resolution for pixel stat extraction (Profile 5)
PIXEL_W = 256
PIXEL_H = 144

DISTRIB_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# Spatial grid for SAT features
SAT_GRID_ROWS = 3
SAT_GRID_COLS = 3

HDR10P_KEY = "HDR Dynamic Metadata SMPTE2094-40 (HDR10+)"
DV_KEY     = "Dolby Vision Metadata"
MD_KEY     = "Mastering display metadata"
CLL_KEY    = "Content light level metadata"

MAX_POLY_SEGS = 8

COLUMNS = [
    "frame_idx", "pts_time", "pict_type", "key_frame",
    "dv_profile",
    # Luminance input features (computed from pixels for P5, from HDR10+ for P8)
    "maxscl", "average_maxrgb", "fraction_bright_pixels",
    "targeted_display_max_nits",
    # Luminance CDF — 9 percentile buckets
    "distrib_pct_0", "distrib_pct_1", "distrib_pct_2", "distrib_pct_3", "distrib_pct_4",
    "distrib_pct_5", "distrib_pct_6", "distrib_pct_7", "distrib_pct_8",
    "distrib_val_0", "distrib_val_1", "distrib_val_2", "distrib_val_3", "distrib_val_4",
    "distrib_val_5", "distrib_val_6", "distrib_val_7", "distrib_val_8",
    # DV RPU source range
    "source_min_pq", "source_max_pq", "scene_refresh",
    # Profile 8 tone curve targets (HDR10+ bezier)
    "knee_x", "knee_y",
    "bezier_0", "bezier_1", "bezier_2", "bezier_3", "bezier_4",
    "bezier_5", "bezier_6", "bezier_7", "bezier_8",
    # Profile 5 tone curve targets (DV RPU Y-component piecewise polynomial)
    "poly_num_segs",
    "poly_pivots",     # space-separated PQ pivot values, e.g. "0 21 84 211 423 679 935 1006 1022"
    "seg0_order", "seg0_c0", "seg0_c1", "seg0_c2",
    "seg1_order", "seg1_c0", "seg1_c1", "seg1_c2",
    "seg2_order", "seg2_c0", "seg2_c1", "seg2_c2",
    "seg3_order", "seg3_c0", "seg3_c1", "seg3_c2",
    "seg4_order", "seg4_c0", "seg4_c1", "seg4_c2",
    "seg5_order", "seg5_c0", "seg5_c1", "seg5_c2",
    "seg6_order", "seg6_c0", "seg6_c1", "seg6_c2",
    "seg7_order", "seg7_c0", "seg7_c1", "seg7_c2",
    # Static mastering metadata (Profile 8 only)
    "mastering_max_lum", "mastering_min_lum", "maxcll", "maxfall",
] + [
    # Spatial zone features (3x3 grid SAT) — mean and max PQ per zone
    # Zones: row0..2 (top→bottom), col0..2 (left→right)
    f"zone_mean_r{r}_c{c}" for r in range(SAT_GRID_ROWS) for c in range(SAT_GRID_COLS)
] + [
    f"zone_max_r{r}_c{c}"  for r in range(SAT_GRID_ROWS) for c in range(SAT_GRID_COLS)
] + [
    # Shadow texture features (dual SAT — variance in dark zones)
    "shadow_texture_var",    # variance of luma in zones with mean < 0.1 PQ
    "dark_zone_count",       # number of 3x3 zones with mean < 0.1 PQ
    "highlight_zone_count",  # number of 3x3 zones with mean > 0.5 PQ
    # Highlight locality features
    "highlight_x",           # normalised x of peak zone centre (0=left, 1=right)
    "highlight_y",           # normalised y of peak zone centre (0=top,  1=bottom)
    "highlight_edge_flag",   # 1 if peak zone is on frame edge, 0 if interior
    # Stratification labels (not model features — used for balanced sampling)
    "lum_tier",              # dark/mid/bright  (from maxscl)
    "spatial_cls",           # concentrated/distributed/flat (from highlight_concentration)
    "shadow_tex_cls",        # textured/flat (from shadow_texture_var)
    "cell_id",               # e.g. "dark_concentrated_textured"
]


# ---------------------------------------------------------------------------
# Source discovery — handles MKV/MP4/m2ts, BDMV folders, and mounted ISOs
# ---------------------------------------------------------------------------

def _scan_rpu_nals(path: str, stream_sel: str = "v:0", check_secs: int = 5) -> int:
    """Return count of UNSPEC62 (RPU) NALs found in the first check_secs of stream_sel."""
    cmd = [FFMPEG, "-v", "error", "-i", path,
           "-map", f"0:{stream_sel}", "-t", str(check_secs),
           "-c:v", "copy", "-f", "hevc", "pipe:1"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = proc.stdout.read()
        proc.wait()
        return sum(1 for i in range(len(data) - 5)
                   if data[i:i+4] == b'\x00\x00\x00\x01' and ((data[i+4] >> 1) & 0x3F) == 62)
    except Exception:
        return 0


def _find_bdmv_streams(folder: Path):
    """
    Given a BDMV disc root (or BDMV parent), locate:
      - bl_path: largest .m2ts in BDMV/STREAM/ (base layer — used for pixel decode)
      - el_path: second-largest .m2ts if size ratio < 50× AND it carries RPU NALs

    Returns (bl_path_str, el_path_str_or_None).
    Raises FileNotFoundError if no BDMV/STREAM found.
    """
    # Accept: disc_root/, disc_root/BDMV/, or disc_root/Title/ (any depth ≤ 3)
    stream_dir = None
    for candidate in [folder / "BDMV" / "STREAM",
                      folder / "STREAM",
                      folder]:
        if candidate.is_dir() and any(candidate.glob("*.m2ts")):
            stream_dir = candidate
            break

    if stream_dir is None:
        raise FileNotFoundError(f"No BDMV/STREAM directory found under {folder}")

    files = sorted(
        [(p.stat().st_size, p) for p in stream_dir.glob("*.m2ts")],
        reverse=True
    )
    bl_size, bl_path = files[0]

    el_path = None
    for el_size, candidate in files[1:]:
        if bl_size / el_size > 50:
            break                        # too small to be EL
        if _scan_rpu_nals(str(candidate), "v:0", check_secs=5) > 0:
            el_path = candidate
            break

    return str(bl_path), (str(el_path) if el_path else None)


def discover_sources(input_path: str):
    """
    Auto-detect the correct streams for a DV title.

    Returns:
        pixel_path  - file for pixel decode (BL for P7 BDMV, same as input otherwise)
        rpu_path    - file for RPU/metadata probe
        rpu_stream  - stream selector for rpu_path, e.g. "v:0" or "v:1"
        dv_profile  - int or None (detected from rpu_path)

    Handles:
        - Single file (MKV/MP4/m2ts): checks v:0 and v:1 for RPU NALs
        - BDMV folder or mounted ISO: finds BL + EL via _find_bdmv_streams()
    """
    path = Path(input_path)

    # ---- BDMV folder or mounted ISO mount point ----
    if path.is_dir():
        print(f"  Source type: BDMV/disc folder — scanning for BL + EL streams")
        bl_path, el_path = _find_bdmv_streams(path)
        print(f"  BL (pixels): {Path(bl_path).name}")
        if el_path:
            print(f"  EL (RPU):    {Path(el_path).name}  [Profile 7 separate EL]")
            return bl_path, el_path, "v:0", 7
        # No separate EL — check if BL has interleaved EL in v:1
        if _scan_rpu_nals(bl_path, "v:1", check_secs=5) > 0:
            print(f"  EL (RPU):    {Path(bl_path).name} v:1  [Profile 7 interleaved]")
            return bl_path, bl_path, "v:1", 7
        # Single layer — P5 or P8
        print(f"  RPU in BL v:0  [Profile 5/8 single layer]")
        profile = _probe_dv_profile_file(bl_path, "v:0") or 5
        return bl_path, bl_path, "v:0", profile

    # ---- Single file (MKV, MP4, m2ts) ----
    input_str = str(path)

    # Check v:1 first — Profile 7 MKV with BL in v:0 and EL accessible via v:1
    if _scan_rpu_nals(input_str, "v:1", check_secs=5) > 0:
        print(f"  Source type: single file, RPU in v:1  [Profile 7 interleaved MKV/m2ts]")
        profile = _probe_dv_profile_file(input_str, "v:0")
        return input_str, input_str, "v:1", profile

    # v:0 only — Profile 5, 8, or Profile 7 MKV with EL muxed into v:0
    print(f"  Source type: single file, RPU in v:0")
    profile = _probe_dv_profile_file(input_str, "v:0")
    return input_str, input_str, "v:0", profile


def _probe_dv_profile_file(path: str, stream_sel: str = "v:0") -> int:
    """Read DV profile from DOVI configuration record side data. Returns int or None."""
    sel = stream_sel.split(":")[-1]   # "v:0" → "v:0", or just "v:0"
    cmd = [FFPROBE, "-v", "error",
           "-select_streams", sel,
           "-probesize", "50000000",
           "-show_entries", "stream_side_data=dv_profile",
           "-of", "default=noprint_wrappers=1", path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=15)
        for line in out.splitlines():
            if line.startswith("dv_profile="):
                val = line.split("=", 1)[1].strip()
                return int(val) if val.isdigit() else None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup_pairs(pairs):
    """JSON object_pairs_hook: collect duplicate keys into lists."""
    d = {}
    for k, v in pairs:
        if k in d:
            existing = d[k]
            if not isinstance(existing, list):
                d[k] = [existing]
            d[k].append(v)
        else:
            d[k] = v
    return d


def frac(s, pick="max"):
    """Parse a fraction string '3051/100000' or number to float.
    If s is a list (duplicate JSON keys), pick 'max'/'min'/'last'.
    """
    if s is None:
        return None
    if isinstance(s, list):
        vals = [frac(x) for x in s]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return max(vals) if pick == "max" else (min(vals) if pick == "min" else vals[-1])
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return float(Fraction(str(s)))
    except Exception:
        return None


def as_list(v):
    if isinstance(v, list):
        return v
    return [v] if v is not None else []


def parse_poly_coef(raw):
    """Parse space-separated or list poly_coef to list of floats."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [float(x) for x in raw.split()]
    return [float(x) for x in as_list(raw)]


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg wrappers
# ---------------------------------------------------------------------------

def get_probe_info(pixel_path: str, rpu_path: str = None, rpu_stream: str = "v:0"):
    """
    Return (duration_secs, dv_profile).
    duration comes from pixel_path; dv_profile from rpu_path (falls back to pixel_path).
    """
    # Duration from the pixel/BL file
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_entries", "format=duration",
           pixel_path]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    duration = float(json.loads(out)["format"]["duration"])

    # DV profile: prefer already-discovered value; otherwise probe rpu_path
    dv_profile = None
    probe_target = rpu_path or pixel_path
    stream_sel = rpu_stream.split(":")[-1]   # "v:0" → "v:0"
    cmd2 = [FFPROBE, "-v", "error", "-print_format", "json",
            "-show_entries", "stream_side_data_list",
            "-select_streams", stream_sel,
            probe_target]
    try:
        out2 = subprocess.check_output(cmd2, stderr=subprocess.DEVNULL)
        d2 = json.loads(out2, object_pairs_hook=_dedup_pairs)
        for s in d2.get("streams", [{}])[0].get("side_data_list", []):
            if isinstance(s, dict) and s.get("side_data_type") == "DOVI configuration record":
                dv_profile = s.get("dv_profile")
                break
    except Exception:
        pass

    return duration, dv_profile


def probe_chunk(rpu_path: str, rpu_stream: str, start_sec: float, duration_sec: float) -> list:
    """
    Run ffprobe on a time window of rpu_path:rpu_stream.
    Returns list of frame metadata dicts with DV side data.
    """
    stream_sel = rpu_stream.split(":")[-1]   # "v:0" or "v:1"
    cmd = [
        FFPROBE, "-v", "error",
        "-print_format", "json",
        "-show_frames",
        "-read_intervals", f"{start_sec}%+{duration_sec}",
        "-select_streams", stream_sel,
        rpu_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"  ffprobe error at {start_sec:.0f}s: {e}", file=sys.stderr)
        return []
    return json.loads(out, object_pairs_hook=_dedup_pairs).get("frames", [])


def compute_sat_features(y_2d: np.ndarray, grid_rows: int = SAT_GRID_ROWS,
                          grid_cols: int = SAT_GRID_COLS) -> dict:
    """
    Compute zonal mean and max PQ using a summed area table (integral image).
    y_2d: (H, W) float32 PQ luma frame.
    Returns dict of zone_mean_rR_cC and zone_max_rR_cC for a grid_rows x grid_cols grid.
    """
    H, W = y_2d.shape

    # Summed area table for mean: O(1) zone sum queries
    sat = np.zeros((H + 1, W + 1), dtype=np.float64)
    sat[1:, 1:] = np.cumsum(np.cumsum(y_2d.astype(np.float64), axis=0), axis=1)

    feats = {}
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Zone pixel boundaries
            r0 = int(round(r     * H / grid_rows))
            r1 = int(round((r+1) * H / grid_rows))
            c0 = int(round(c     * W / grid_cols))
            c1 = int(round((c+1) * W / grid_cols))

            n = max((r1 - r0) * (c1 - c0), 1)
            zone_sum = sat[r1, c1] - sat[r0, c1] - sat[r1, c0] + sat[r0, c0]
            zone_mean = float(zone_sum / n)
            zone_max  = float(y_2d[r0:r1, c0:c1].max())

            feats[f"zone_mean_r{r}_c{c}"] = zone_mean
            feats[f"zone_max_r{r}_c{c}"]  = zone_max

    return feats


def compute_texture_and_locality(y_2d: np.ndarray, zone_feats: dict,
                                  grid_rows: int = SAT_GRID_ROWS,
                                  grid_cols: int = SAT_GRID_COLS) -> dict:
    """
    Compute shadow texture, highlight locality, and stratification labels.

    Shadow texture: variance of luma in dark zones using dual SAT (sum + sum-of-squares).
    Highlight locality: which zone holds peak brightness and where it sits spatially.
    Stratification: classify into 18 cells for balanced dataset sampling.
    """
    H, W = y_2d.shape
    feats = {}

    # --- Dual SAT for variance in O(1) per zone ---
    sat_sum = np.zeros((H + 1, W + 1), dtype=np.float64)
    sat_sq  = np.zeros((H + 1, W + 1), dtype=np.float64)
    y64 = y_2d.astype(np.float64)
    sat_sum[1:, 1:] = np.cumsum(np.cumsum(y64,      axis=0), axis=1)
    sat_sq [1:, 1:] = np.cumsum(np.cumsum(y64 ** 2, axis=0), axis=1)

    def zone_var(r0, r1, c0, c1):
        n = max((r1 - r0) * (c1 - c0), 1)
        s  = sat_sum[r1,c1] - sat_sum[r0,c1] - sat_sum[r1,c0] + sat_sum[r0,c0]
        s2 = sat_sq [r1,c1] - sat_sq [r0,c1] - sat_sq [r1,c0] + sat_sq [r0,c0]
        return max(0.0, float(s2 / n - (s / n) ** 2))

    # --- Shadow texture & highlight locality ---
    dark_vars  = []
    dark_count = 0
    hi_count   = 0
    peak_zone_r, peak_zone_c = 0, 0
    peak_zone_val = -1.0

    for r in range(grid_rows):
        for c in range(grid_cols):
            r0 = int(round(r     * H / grid_rows))
            r1 = int(round((r+1) * H / grid_rows))
            c0 = int(round(c     * W / grid_cols))
            c1 = int(round((c+1) * W / grid_cols))
            zm = zone_feats[f"zone_mean_r{r}_c{c}"]
            zx = zone_feats[f"zone_max_r{r}_c{c}"]

            if zm < 0.1:
                dark_count += 1
                dark_vars.append(zone_var(r0, r1, c0, c1))
            if zm > 0.5:
                hi_count += 1
            if zx > peak_zone_val:
                peak_zone_val = zx
                peak_zone_r, peak_zone_c = r, c

    feats["shadow_texture_var"]   = float(np.mean(dark_vars)) if dark_vars else 0.0
    feats["dark_zone_count"]      = dark_count
    feats["highlight_zone_count"] = hi_count

    # Highlight locality — normalised centre of peak zone
    feats["highlight_x"]         = (peak_zone_c + 0.5) / grid_cols  # 0=left, 1=right
    feats["highlight_y"]         = (peak_zone_r + 0.5) / grid_rows  # 0=top,  1=bottom
    feats["highlight_edge_flag"] = int(
        peak_zone_r == 0 or peak_zone_r == grid_rows - 1 or
        peak_zone_c == 0 or peak_zone_c == grid_cols - 1
    )

    # --- Stratification labels ---
    maxscl = zone_feats.get("maxscl", float(y_2d.max()))  # fallback
    # Use zone features if available, otherwise compute inline
    peak_zone_max = max(zone_feats.get(f"zone_max_r{r}_c{c}", 0.0)
                        for r in range(grid_rows) for c in range(grid_cols))
    hi_conc = peak_zone_max / max(maxscl, 1e-6)

    lum_tier = "bright" if maxscl >= 0.6 else ("mid" if maxscl >= 0.3 else "dark")
    spatial_cls = ("concentrated" if hi_conc >= 0.85
                   else ("distributed" if hi_conc >= 0.5 else "flat"))
    shadow_tex_cls = "textured" if feats["shadow_texture_var"] > 0.002 else "flat"

    feats["lum_tier"]      = lum_tier
    feats["spatial_cls"]   = spatial_cls
    feats["shadow_tex_cls"] = shadow_tex_cls
    feats["cell_id"]       = f"{lum_tier}_{spatial_cls}_{shadow_tex_cls}"

    return feats


def decode_chunk_pixel_stats(input_path: str, start_sec: float, duration_sec: float) -> dict:
    """
    Decode the chunk at low resolution via ffmpeg at 1fps, extract 10-bit PQ Y channel,
    compute luminance stats per frame.

    Returns dict: round(abs_pts_sec) -> stats_dict
    The caller looks up a frame's pixel stats by rounding its pts_time to the nearest second.
    """
    w, h = PIXEL_W, PIXEL_H
    y_bytes = w * h * 2                           # 10-bit LE, 2 bytes/pixel
    uv_bytes = (w // 2) * (h // 2) * 2 * 2       # U + V planes
    frame_bytes = y_bytes + uv_bytes

    cmd = [
        FFMPEG, "-v", "error",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", input_path,
        "-vf", f"scale={w}:{h}:flags=bilinear,fps=1",
        "-pix_fmt", "yuv420p10le",
        "-f", "rawvideo",
        "pipe:1",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    result = {}
    frame_idx = 0

    while True:
        data = proc.stdout.read(frame_bytes)
        if len(data) < frame_bytes:
            break

        # Y plane: 10-bit little-endian uint16, values 0-1023 (full range PQ)
        y_flat = np.frombuffer(data[:y_bytes], dtype="<u2").astype(np.float32) / 1023.0
        y_2d   = y_flat.reshape(h, w)
        pcts   = np.percentile(y_flat, DISTRIB_PERCENTILES)

        abs_sec = round(start_sec + frame_idx)
        base_stats = {
            "maxscl":                  float(y_flat.max()),
            "average_maxrgb":          float(y_flat.mean()),
            "fraction_bright_pixels":  float((y_flat > 0.5).mean()),
            "targeted_display_max_nits": None,
            **{f"distrib_pct_{i}": float(DISTRIB_PERCENTILES[i]) for i in range(9)},
            **{f"distrib_val_{i}": float(pcts[i]) for i in range(9)},
        }
        sat_feats = compute_sat_features(y_2d)
        # Pass base maxscl into texture/locality so stratification uses it
        sat_feats["maxscl"] = base_stats["maxscl"]
        tex_feats = compute_texture_and_locality(y_2d, sat_feats)
        result[abs_sec] = {**base_stats, **sat_feats, **tex_feats}
        frame_idx += 1

    proc.wait()
    return result


def lookup_pixel_stats(pixel_stats: dict, pts_time: float, pts_offset: float = 0.0) -> dict:
    """Find closest pixel stats entry to the given pts_time (tolerance ±2 s).
    pts_offset: subtract this from pts_time before lookup (corrects BDMV timestamp base)."""
    if not pixel_stats:
        return {}
    key = round(pts_time - pts_offset)
    for delta in range(3):
        for k in (key + delta, key - delta):
            if k in pixel_stats:
                return pixel_stats[k]
    return {}


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------

def extract_row(frame_idx: int, f: dict, dv_profile: int, pixel_stats: dict,
                pts_offset: float = 0.0) -> dict:
    """Convert a ffprobe frame dict (+ optional pixel stats) to a CSV row dict."""
    sdata = {s["side_data_type"]: s for s in f.get("side_data_list", [])}
    h   = sdata.get(HDR10P_KEY, {})
    dv  = sdata.get(DV_KEY, {})
    md  = sdata.get(MD_KEY, {})
    cll = sdata.get(CLL_KEY, {})

    bezier      = as_list(h.get("bezier_curve_anchors"))
    distrib_pct = as_list(h.get("distribution_maxrgb_percentage"))
    distrib_val = as_list(h.get("distribution_maxrgb_percentile"))

    comps    = dv.get("components", [])
    y_pieces = []
    y_pivots = ""
    if comps:
        y_comp   = comps[0]
        y_pivots = y_comp.get("pivots", "")
        y_pieces = as_list(y_comp.get("pieces", []))

    pts_time = float(f.get("pts_time") or 0)

    # Choose feature source: HDR10+ metadata (P8) or pixel-derived stats (P5)
    if h:
        feat = {
            "maxscl":                  frac(h.get("maxscl"), pick="max"),
            "average_maxrgb":          frac(h.get("average_maxrgb")),
            "fraction_bright_pixels":  frac(h.get("fraction_bright_pixels")),
            "targeted_display_max_nits": frac(h.get("targeted_system_display_maximum_luminance")),
            **{f"distrib_pct_{i}": frac(distrib_pct[i]) if i < len(distrib_pct) else None
               for i in range(9)},
            **{f"distrib_val_{i}": frac(distrib_val[i]) if i < len(distrib_val) else None
               for i in range(9)},
        }
    else:
        feat = lookup_pixel_stats(pixel_stats, pts_time, pts_offset)

    row = {
        "frame_idx":  frame_idx,
        "pts_time":   pts_time,
        "pict_type":  f.get("pict_type"),
        "key_frame":  f.get("key_frame"),
        "dv_profile": dv_profile,
        "source_min_pq": dv.get("source_min_pq"),
        "source_max_pq": dv.get("source_max_pq"),
        "scene_refresh": dv.get("scene_refresh_flag"),
        "knee_x":     frac(h.get("knee_point_x")),
        "knee_y":     frac(h.get("knee_point_y")),
        "poly_num_segs": len(y_pieces),
        "poly_pivots":   y_pivots,
        "mastering_max_lum": frac(md.get("max_luminance")),
        "mastering_min_lum": frac(md.get("min_luminance")),
        "maxcll":  cll.get("max_content"),
        "maxfall": cll.get("max_average"),
        **feat,
    }

    for i in range(9):
        row.setdefault(f"distrib_pct_{i}", None)
        row.setdefault(f"distrib_val_{i}", None)
        row[f"bezier_{i}"] = frac(bezier[i]) if i < len(bezier) else None

    for i in range(MAX_POLY_SEGS):
        if i < len(y_pieces):
            coef = parse_poly_coef(y_pieces[i].get("poly_coef"))
            row[f"seg{i}_order"] = y_pieces[i].get("poly_order")
            row[f"seg{i}_c0"]    = coef[0] if len(coef) > 0 else None
            row[f"seg{i}_c1"]    = coef[1] if len(coef) > 1 else None
            row[f"seg{i}_c2"]    = coef[2] if len(coef) > 2 else None
        else:
            row[f"seg{i}_order"] = None
            row[f"seg{i}_c0"]    = None
            row[f"seg{i}_c1"]    = None
            row[f"seg{i}_c2"]    = None

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract DV+HDR10+ per-frame metadata to CSV.")
    ap.add_argument("input", help="Input MKV/MP4 file")
    ap.add_argument("-o", "--output", default="dv_dataset.csv")
    ap.add_argument("--sample-fps", type=float, default=1.0,
                    help="Target sample rate fps (default 1.0). 0 = every frame.")
    ap.add_argument("--chunk-secs", type=int, default=CHUNK_SECS)
    ap.add_argument("--resume", action="store_true",
                    help="Append to existing CSV, resuming after last row's pts_time.")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end",   type=float, default=None)
    ap.add_argument("--no-pixels", action="store_true",
                    help="Stage 1 manifest mode: skip pixel decode, extract RPU metadata only.")
    args = ap.parse_args()

    print(f"Probing: {args.input}")

    # Auto-discover pixel source, RPU source, and stream selector
    pixel_path, rpu_path, rpu_stream, disc_profile = discover_sources(args.input)

    total_dur, probe_profile = get_probe_info(pixel_path, rpu_path, rpu_stream)
    dv_profile = disc_profile or probe_profile
    end_sec = args.end if args.end is not None else total_dur

    needs_pixel_decode = (dv_profile != 8) and not args.no_pixels
    feature_src = ("HDR10+ side-data" if dv_profile == 8
                   else ("manifest only — no pixel decode" if args.no_pixels
                         else f"pixel decode (Profile {dv_profile})"))
    rpu_note = Path(rpu_path).name if rpu_path != pixel_path else "(same as pixel)"

    print(f"  Duration: {total_dur:.1f}s ({total_dur/3600:.2f}h)  DV profile: {dv_profile}")
    print(f"  Feature source: {feature_src}")
    print(f"  RPU stream: {rpu_note} [{rpu_stream}]")
    print(f"  Processing: {args.start:.0f}s - {end_sec:.0f}s")

    resume_start = args.start
    if args.resume and Path(args.output).exists():
        with open(args.output, newline="") as fh:
            existing = list(csv.DictReader(fh))
        if existing:
            last_pts = float(existing[-1]["pts_time"] or 0)
            resume_start = max(args.start, last_pts - args.chunk_secs)
            print(f"  Resuming from {resume_start:.1f}s (last pts={last_pts:.1f}s)")

    write_mode   = "a" if args.resume and Path(args.output).exists() else "w"
    write_header = (write_mode == "w")
    min_interval = 1.0 / args.sample_fps if args.sample_fps > 0 else 0.0

    frame_idx      = 0
    frames_written = 0
    chunk_start    = resume_start
    prev_pts       = -999.0
    total_chunks   = int((end_sec - chunk_start) / args.chunk_secs) + 1
    chunk_num      = 0

    with open(args.output, write_mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        while chunk_start < end_sec:
            chunk_end = min(chunk_start + args.chunk_secs, end_sec)
            dur       = chunk_end - chunk_start
            chunk_num += 1
            t0 = time.time()

            # RPU/metadata pass — probes rpu_path:rpu_stream for DV polynomial + scene flags
            frames = probe_chunk(rpu_path, rpu_stream, chunk_start, dur)

            # Pixel pass — decodes pixel_path (BL / main file) for histogram+SAT features
            pixel_stats = {}
            pts_offset  = 0.0
            if needs_pixel_decode:
                pixel_stats = decode_chunk_pixel_stats(pixel_path, chunk_start, dur)
                # BDMV m2ts files may have non-zero pts base — calibrate offset from first frame
                if frames and pixel_stats:
                    first_pts = float(frames[0].get("pts_time", chunk_start) or chunk_start)
                    pts_offset = first_pts - chunk_start

            chunk_written = 0
            for f in frames:
                pts = float(f.get("pts_time", 0) or 0)
                sd  = {s["side_data_type"]: s for s in f.get("side_data_list", [])}
                is_scene = bool(sd.get(DV_KEY, {}).get("scene_refresh_flag"))

                if args.sample_fps > 0 and not is_scene and (pts - prev_pts) < min_interval:
                    frame_idx += 1
                    continue

                row = extract_row(frame_idx, f, dv_profile, pixel_stats,
                                  pts_offset=pts_offset)
                writer.writerow(row)
                prev_pts = pts
                chunk_written += 1
                frames_written += 1
                frame_idx += 1

            # Flush after every chunk so concurrent readers see consistent data
            csvfile.flush()
            os.fsync(csvfile.fileno())

            elapsed = time.time() - t0
            pct = 100.0 * (chunk_start - args.start) / max(1, end_sec - args.start)
            px_note = f" px_frames={len(pixel_stats)}" if needs_pixel_decode else ""
            print(f"  [{chunk_num}/{total_chunks}] {chunk_start:.0f}s-{chunk_end:.0f}s"
                  f" | +{chunk_written} rows | total={frames_written}{px_note} | {elapsed:.1f}s [{pct:.0f}%]")

            chunk_start = chunk_end

    print(f"\nDone. {frames_written} rows -> {args.output}")


if __name__ == "__main__":
    main()
