"""
dv_rpu_curves.py
================
Extract per-frame DV RPU curves via libdovi (C FFI), using raw HEVC UNSPEC62 NAL units
piped from ffmpeg. Produces a reference curve CSV that can be cross-checked against
the ffprobe-derived curves in dv_dataset_full.csv.

For each frame outputs:
  frame_idx, pts_approx, coef_log2_denom, bl_bit_depth,
  num_pivots, pivots (space-sep),
  seg{0..7}_order, seg{0..7}_c0, seg{0..7}_c1, seg{0..7}_c2,  (raw int64 from libdovi)
  seg{0..7}_c0f, seg{0..7}_c1f, seg{0..7}_c2f,                 (scaled floats: /2^denom)
  l1_min_pq, l1_max_pq, l1_avg_pq,                              (Level1 DM block)
  scene_refresh, source_min_pq, source_max_pq

Usage:
  python dv_rpu_curves.py INPUT.mp4 -o rpu_curves.csv [--start 0] [--duration 60]
"""

import subprocess
import ctypes
import csv
import sys
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# libdovi FFI
# ---------------------------------------------------------------------------
LIBDOVI_PATH = "C:/msys64/ucrt64/bin/libdovi.dll"

class DoviData(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint8)), ("len", ctypes.c_size_t)]

class DoviU16Data(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint16)), ("len", ctypes.c_size_t)]

class DoviU64Data(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint64)), ("len", ctypes.c_size_t)]

class DoviI64Data(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_int64)), ("len", ctypes.c_size_t)]

class DoviI64Data2D(ctypes.Structure):
    _fields_ = [("list", ctypes.POINTER(ctypes.POINTER(DoviI64Data))), ("len", ctypes.c_size_t)]

class DoviU64Data2D(ctypes.Structure):
    _fields_ = [("list", ctypes.POINTER(ctypes.POINTER(DoviU64Data))), ("len", ctypes.c_size_t)]

class DoviPolynomialCurve(ctypes.Structure):
    _fields_ = [
        ("poly_order_minus1", DoviU64Data),
        ("linear_interp_flag", DoviData),
        ("poly_coef_int", DoviI64Data2D),
        ("poly_coef", DoviU64Data2D),
    ]

class DoviReshapingCurve(ctypes.Structure):
    _fields_ = [
        ("num_pivots_minus2", ctypes.c_uint64),
        ("pivots", DoviU16Data),
        ("mapping_idc", ctypes.c_uint8),
        ("polynomial", ctypes.POINTER(DoviPolynomialCurve)),
        ("mmr", ctypes.c_void_p),
    ]

class DoviRpuDataMapping(ctypes.Structure):
    _fields_ = [
        ("vdr_rpu_id", ctypes.c_uint64),
        ("mapping_color_space", ctypes.c_uint64),
        ("mapping_chroma_format_idc", ctypes.c_uint64),
        ("num_x_partitions_minus1", ctypes.c_uint64),
        ("num_y_partitions_minus1", ctypes.c_uint64),
        ("curves", DoviReshapingCurve * 3),
        ("nlq_method_idc", ctypes.c_int32),
        ("nlq_num_pivots_minus2", ctypes.c_int32),
        ("nlq_pred_pivot_value", DoviU16Data),
        ("nlq", ctypes.c_void_p),
    ]

class DoviRpuDataHeader(ctypes.Structure):
    _fields_ = [
        ("guessed_profile", ctypes.c_uint8),
        ("el_type", ctypes.c_char_p),
        ("rpu_nal_prefix", ctypes.c_uint8),
        ("rpu_type", ctypes.c_uint8),
        ("rpu_format", ctypes.c_uint16),
        ("vdr_rpu_profile", ctypes.c_uint8),
        ("vdr_rpu_level", ctypes.c_uint8),
        ("vdr_seq_info_present_flag", ctypes.c_bool),
        ("chroma_resampling_explicit_filter_flag", ctypes.c_bool),
        ("coefficient_data_type", ctypes.c_uint8),
        ("coefficient_log2_denom", ctypes.c_uint64),
        ("vdr_rpu_normalized_idc", ctypes.c_uint8),
        ("bl_video_full_range_flag", ctypes.c_bool),
        ("bl_bit_depth_minus8", ctypes.c_uint64),
        ("el_bit_depth_minus8", ctypes.c_uint64),
        ("vdr_bit_depth_minus8", ctypes.c_uint64),
        ("spatial_resampling_filter_flag", ctypes.c_bool),
        ("reserved_zero_3bits", ctypes.c_uint8),
        ("el_spatial_resampling_filter_flag", ctypes.c_bool),
        ("disable_residual_flag", ctypes.c_bool),
        ("vdr_dm_metadata_present_flag", ctypes.c_bool),
        ("use_prev_vdr_rpu_flag", ctypes.c_bool),
        ("prev_vdr_rpu_id", ctypes.c_uint64),
    ]

class DoviExtMetadataBlockLevel1(ctypes.Structure):
    _fields_ = [
        ("min_pq", ctypes.c_uint16),
        ("max_pq", ctypes.c_uint16),
        ("avg_pq", ctypes.c_uint16),
    ]

class DoviDmData(ctypes.Structure):
    # Simplified: we only need num_ext_blocks and level1 pointer
    # The full struct has many more fields but we only access level1
    _fields_ = [
        ("num_ext_blocks", ctypes.c_uint64),
        ("level1", ctypes.POINTER(DoviExtMetadataBlockLevel1)),
        # remaining fields omitted - we don't access them via pointer arithmetic
    ]

class DoviVdrDmData(ctypes.Structure):
    _fields_ = [
        ("compressed", ctypes.c_bool),
        ("affected_dm_metadata_id", ctypes.c_uint64),
        ("current_dm_metadata_id", ctypes.c_uint64),
        ("scene_refresh_flag", ctypes.c_uint64),
        ("ycc_to_rgb_coef0", ctypes.c_int16),
        ("ycc_to_rgb_coef1", ctypes.c_int16),
        ("ycc_to_rgb_coef2", ctypes.c_int16),
        ("ycc_to_rgb_coef3", ctypes.c_int16),
        ("ycc_to_rgb_coef4", ctypes.c_int16),
        ("ycc_to_rgb_coef5", ctypes.c_int16),
        ("ycc_to_rgb_coef6", ctypes.c_int16),
        ("ycc_to_rgb_coef7", ctypes.c_int16),
        ("ycc_to_rgb_coef8", ctypes.c_int16),
        ("ycc_to_rgb_offset0", ctypes.c_uint32),
        ("ycc_to_rgb_offset1", ctypes.c_uint32),
        ("ycc_to_rgb_offset2", ctypes.c_uint32),
        ("rgb_to_lms_coef0", ctypes.c_int16),
        ("rgb_to_lms_coef1", ctypes.c_int16),
        ("rgb_to_lms_coef2", ctypes.c_int16),
        ("rgb_to_lms_coef3", ctypes.c_int16),
        ("rgb_to_lms_coef4", ctypes.c_int16),
        ("rgb_to_lms_coef5", ctypes.c_int16),
        ("rgb_to_lms_coef6", ctypes.c_int16),
        ("rgb_to_lms_coef7", ctypes.c_int16),
        ("rgb_to_lms_coef8", ctypes.c_int16),
        ("signal_eotf", ctypes.c_uint16),
        ("signal_eotf_param0", ctypes.c_uint16),
        ("signal_eotf_param1", ctypes.c_uint16),
        ("signal_eotf_param2", ctypes.c_uint32),
        ("signal_bit_depth", ctypes.c_uint8),
        ("signal_color_space", ctypes.c_uint8),
        ("signal_chroma_format", ctypes.c_uint8),
        ("signal_full_range_flag", ctypes.c_uint8),
        ("source_min_pq", ctypes.c_uint16),
        ("source_max_pq", ctypes.c_uint16),
        ("source_diagonal", ctypes.c_uint16),
        ("dm_data", DoviDmData),
    ]


def load_libdovi():
    lib = ctypes.CDLL(LIBDOVI_PATH)
    lib.dovi_parse_unspec62_nalu.restype  = ctypes.c_void_p
    lib.dovi_parse_unspec62_nalu.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.dovi_rpu_free.restype  = None
    lib.dovi_rpu_free.argtypes = [ctypes.c_void_p]
    lib.dovi_rpu_get_error.restype  = ctypes.c_char_p
    lib.dovi_rpu_get_error.argtypes = [ctypes.c_void_p]
    lib.dovi_rpu_get_header.restype  = ctypes.POINTER(DoviRpuDataHeader)
    lib.dovi_rpu_get_header.argtypes = [ctypes.c_void_p]
    lib.dovi_rpu_free_header.restype  = None
    lib.dovi_rpu_free_header.argtypes = [ctypes.POINTER(DoviRpuDataHeader)]
    lib.dovi_rpu_get_data_mapping.restype  = ctypes.POINTER(DoviRpuDataMapping)
    lib.dovi_rpu_get_data_mapping.argtypes = [ctypes.c_void_p]
    lib.dovi_rpu_free_data_mapping.restype  = None
    lib.dovi_rpu_free_data_mapping.argtypes = [ctypes.POINTER(DoviRpuDataMapping)]
    lib.dovi_rpu_get_vdr_dm_data.restype  = ctypes.POINTER(DoviVdrDmData)
    lib.dovi_rpu_get_vdr_dm_data.argtypes = [ctypes.c_void_p]
    lib.dovi_rpu_free_vdr_dm_data.restype  = None
    lib.dovi_rpu_free_vdr_dm_data.argtypes = [ctypes.POINTER(DoviVdrDmData)]
    return lib


# ---------------------------------------------------------------------------
# NAL extraction
# ---------------------------------------------------------------------------
def iter_rpu_nals(input_path: str, start_sec: float, duration_sec: float):
    """
    Yield raw bytes of each UNSPEC62 RPU NAL unit (without the 4-byte start code,
    including the 2-byte NAL header) from the given time window.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", str(start_sec), "-t", str(duration_sec),
        "-i", input_path,
        "-map", "0:v:0", "-c:v", "copy",
        "-bsf:v", "hevc_mp4toannexb",
        "-f", "hevc", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buf = b""
    CHUNK = 1 << 20  # 1 MB

    def emit_nal(nal_bytes):
        if len(nal_bytes) < 2:
            return None
        nal_type = (nal_bytes[0] >> 1) & 0x3F
        if nal_type == 62:  # UNSPEC62 = RPU
            return nal_bytes
        return None

    starts = []
    while True:
        chunk = proc.stdout.read(CHUNK)
        if not chunk:
            break
        buf += chunk
        # find all start codes in current buffer, emit complete NALs
        i = 0
        while i < len(buf) - 4:
            if buf[i:i+4] == b'\x00\x00\x00\x01':
                starts.append(i)
                if len(starts) >= 2:
                    nal = buf[starts[-2]+4 : starts[-1]]
                    r = emit_nal(nal)
                    if r is not None:
                        yield r
            i += 1
        # keep tail (last start code onward) in buffer
        if starts:
            buf = buf[starts[-1]:]
            starts = [0]

    proc.wait()
    # emit final NAL
    if starts:
        nal = buf[starts[-1]+4:]
        r = emit_nal(nal)
        if r is not None:
            yield r


# ---------------------------------------------------------------------------
# Parse one RPU NAL via libdovi
# ---------------------------------------------------------------------------
MAX_SEGS = 8

COLUMNS = (
    ["frame_idx", "coef_log2_denom", "bl_bit_depth",
     "num_pivots", "pivots",
     "scene_refresh", "source_min_pq", "source_max_pq",
     "l1_min_pq", "l1_max_pq", "l1_avg_pq"] +
    [f"seg{i}_{f}" for i in range(MAX_SEGS) for f in ("order","c0","c1","c2")] +
    [f"seg{i}_{f}f" for i in range(MAX_SEGS) for f in ("c0","c1","c2")]
)


def parse_rpu(lib, nal_bytes: bytes):
    buf = (ctypes.c_uint8 * len(nal_bytes))(*nal_bytes)
    rpu = lib.dovi_parse_unspec62_nalu(buf, len(nal_bytes))
    if not rpu:
        return None

    err = lib.dovi_rpu_get_error(rpu)
    if err:
        lib.dovi_rpu_free(rpu)
        return None

    row = {}

    # --- header ---
    hdr_ptr = lib.dovi_rpu_get_header(rpu)
    if hdr_ptr:
        hdr = hdr_ptr.contents
        row["coef_log2_denom"] = hdr.coefficient_log2_denom
        row["bl_bit_depth"]    = hdr.bl_bit_depth_minus8 + 8
        lib.dovi_rpu_free_header(hdr_ptr)

    coef_scale = 1.0 / (2 ** row.get("coef_log2_denom", 23))
    bl_max     = (2 ** row.get("bl_bit_depth", 10)) - 1  # 1023

    # --- mapping (Y component = curves[0]) ---
    mapping_ptr = lib.dovi_rpu_get_data_mapping(rpu)
    if mapping_ptr:
        mapping = mapping_ptr.contents
        curve   = mapping.curves[0]
        num_pivots = curve.num_pivots_minus2 + 2
        row["num_pivots"] = num_pivots
        pivots = [curve.pivots.data[i] for i in range(min(num_pivots, curve.pivots.len))]
        row["pivots"] = " ".join(str(p) for p in pivots)

        poly_ptr = curve.polynomial
        poly = poly_ptr.contents if poly_ptr else None
        if poly:
            n_segs = num_pivots - 1
            for i in range(MAX_SEGS):
                if i < n_segs and i < poly.poly_order_minus1.len:
                    order = int(poly.poly_order_minus1.data[i]) + 1

                    # poly_coef_int: signed integer part
                    # poly_coef:     fractional part (unsigned)
                    # combined = poly_coef_int * 2^denom + poly_coef (not used here —
                    # ffprobe already combines them as a single int64 in poly_coef)
                    # libdovi separates int and frac — we reconstruct the combined value
                    c_int = [0, 0, 0]
                    c_frac = [0, 0, 0]
                    if i < poly.poly_coef_int.len and poly.poly_coef_int.list[i]:
                        coef_int_row = poly.poly_coef_int.list[i].contents
                        for k in range(min(3, coef_int_row.len)):
                            c_int[k] = coef_int_row.data[k]
                    if i < poly.poly_coef.len and poly.poly_coef.list[i]:
                        coef_frac_row = poly.poly_coef.list[i].contents
                        for k in range(min(3, coef_frac_row.len)):
                            c_frac[k] = coef_frac_row.data[k]

                    # Reconstruct: combined = int_part * 2^denom + frac_part
                    denom = row.get("coef_log2_denom", 23)
                    combined = [c_int[k] * (2**denom) + c_frac[k] for k in range(3)]

                    row[f"seg{i}_order"] = order
                    row[f"seg{i}_c0"]    = combined[0]
                    row[f"seg{i}_c1"]    = combined[1]
                    row[f"seg{i}_c2"]    = combined[2]
                    row[f"seg{i}_c0f"]   = combined[0] * coef_scale
                    row[f"seg{i}_c1f"]   = combined[1] * coef_scale
                    row[f"seg{i}_c2f"]   = combined[2] * coef_scale
                else:
                    for f in ("order","c0","c1","c2","c0f","c1f","c2f"):
                        row[f"seg{i}_{f}"] = None

        lib.dovi_rpu_free_data_mapping(mapping_ptr)

    # --- VDR DM data (scene_refresh, source PQ, Level1) ---
    dm_ptr = lib.dovi_rpu_get_vdr_dm_data(rpu)
    if dm_ptr:
        dm = dm_ptr.contents
        row["scene_refresh"]  = int(dm.scene_refresh_flag)
        row["source_min_pq"]  = dm.source_min_pq
        row["source_max_pq"]  = dm.source_max_pq
        l1 = dm.dm_data.level1
        if l1:
            row["l1_min_pq"] = l1.contents.min_pq
            row["l1_max_pq"] = l1.contents.max_pq
            row["l1_avg_pq"] = l1.contents.avg_pq
        lib.dovi_rpu_free_vdr_dm_data(dm_ptr)

    lib.dovi_rpu_free(rpu)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="rpu_curves.csv")
    ap.add_argument("--start",    type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()

    duration = args.duration or 999999.0
    print(f"Loading libdovi from {LIBDOVI_PATH}")
    lib = load_libdovi()
    print(f"Extracting RPU NALs: {args.input}  start={args.start}s  dur={duration}s")

    t0 = time.time()
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        count = 0
        for nal_bytes in iter_rpu_nals(args.input, args.start, duration):
            row = parse_rpu(lib, nal_bytes)
            if row is None:
                continue
            row["frame_idx"] = count
            writer.writerow(row)
            count += 1
            if count % 500 == 0:
                print(f"  {count} frames... ({time.time()-t0:.1f}s)")

    print(f"\nDone. {count} frames -> {args.output}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
