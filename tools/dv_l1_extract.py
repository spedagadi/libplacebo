"""
Fast L1 DM block extractor via libdovi.
Outputs one row per frame: frame_pts_approx, l1_min_pq, l1_max_pq, l1_avg_pq, scene_refresh.
The frame_pts_approx is computed as start_sec + frame_count/fps so it can be
joined to dv_dataset_full.csv on nearest pts_time.
"""
import subprocess, ctypes, csv, sys, argparse, time

LIBDOVI = "C:/msys64/ucrt64/bin/libdovi.dll"
FPS     = 24000 / 1001   # ~23.976

class DoviExtMetadataBlockLevel1(ctypes.Structure):
    _fields_ = [("min_pq", ctypes.c_uint16),
                ("max_pq", ctypes.c_uint16),
                ("avg_pq", ctypes.c_uint16)]

class DoviDmData(ctypes.Structure):
    _fields_ = [("num_ext_blocks", ctypes.c_uint64),
                ("level1", ctypes.POINTER(DoviExtMetadataBlockLevel1)),
                # remaining fields unused — we only read level1 via pointer
                ]

class DoviVdrDmData(ctypes.Structure):
    _fields_ = [
        ("compressed",             ctypes.c_bool),
        ("affected_dm_metadata_id",ctypes.c_uint64),
        ("current_dm_metadata_id", ctypes.c_uint64),
        ("scene_refresh_flag",     ctypes.c_uint64),
        ("ycc_to_rgb_coef0",  ctypes.c_int16),("ycc_to_rgb_coef1",  ctypes.c_int16),
        ("ycc_to_rgb_coef2",  ctypes.c_int16),("ycc_to_rgb_coef3",  ctypes.c_int16),
        ("ycc_to_rgb_coef4",  ctypes.c_int16),("ycc_to_rgb_coef5",  ctypes.c_int16),
        ("ycc_to_rgb_coef6",  ctypes.c_int16),("ycc_to_rgb_coef7",  ctypes.c_int16),
        ("ycc_to_rgb_coef8",  ctypes.c_int16),
        ("ycc_to_rgb_offset0",ctypes.c_uint32),("ycc_to_rgb_offset1",ctypes.c_uint32),
        ("ycc_to_rgb_offset2",ctypes.c_uint32),
        ("rgb_to_lms_coef0",  ctypes.c_int16),("rgb_to_lms_coef1",  ctypes.c_int16),
        ("rgb_to_lms_coef2",  ctypes.c_int16),("rgb_to_lms_coef3",  ctypes.c_int16),
        ("rgb_to_lms_coef4",  ctypes.c_int16),("rgb_to_lms_coef5",  ctypes.c_int16),
        ("rgb_to_lms_coef6",  ctypes.c_int16),("rgb_to_lms_coef7",  ctypes.c_int16),
        ("rgb_to_lms_coef8",  ctypes.c_int16),
        ("signal_eotf",       ctypes.c_uint16),("signal_eotf_param0",ctypes.c_uint16),
        ("signal_eotf_param1",ctypes.c_uint16),("signal_eotf_param2",ctypes.c_uint32),
        ("signal_bit_depth",  ctypes.c_uint8), ("signal_color_space",ctypes.c_uint8),
        ("signal_chroma_format",ctypes.c_uint8),("signal_full_range_flag",ctypes.c_uint8),
        ("source_min_pq",     ctypes.c_uint16),("source_max_pq",     ctypes.c_uint16),
        ("source_diagonal",   ctypes.c_uint16),
        ("dm_data",           DoviDmData),
    ]


def load_lib():
    lib = ctypes.CDLL(LIBDOVI)
    lib.dovi_parse_unspec62_nalu.restype  = ctypes.c_void_p
    lib.dovi_parse_unspec62_nalu.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.dovi_rpu_free.restype  = None
    lib.dovi_rpu_free.argtypes = [ctypes.c_void_p]
    lib.dovi_rpu_get_error.restype  = ctypes.c_char_p
    lib.dovi_rpu_get_error.argtypes = [ctypes.c_void_p]
    lib.dovi_rpu_get_vdr_dm_data.restype  = ctypes.POINTER(DoviVdrDmData)
    lib.dovi_rpu_get_vdr_dm_data.argtypes = [ctypes.c_void_p]
    lib.dovi_rpu_free_vdr_dm_data.restype  = None
    lib.dovi_rpu_free_vdr_dm_data.argtypes = [ctypes.POINTER(DoviVdrDmData)]
    return lib


def iter_rpu_nals(input_path, start_sec, duration_sec):
    cmd = ["ffmpeg", "-v", "error",
           "-ss", str(start_sec), "-t", str(duration_sec),
           "-i", input_path,
           "-map", "0:v:0", "-c:v", "copy",
           "-bsf:v", "hevc_mp4toannexb",
           "-f", "hevc", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buf  = b""
    CHUNK = 1 << 20
    starts = []
    while True:
        chunk = proc.stdout.read(CHUNK)
        if not chunk:
            break
        buf += chunk
        i = 0
        while i < len(buf) - 4:
            if buf[i:i+4] == b'\x00\x00\x00\x01':
                starts.append(i)
                if len(starts) >= 2:
                    nal = buf[starts[-2]+4 : starts[-1]]
                    if len(nal) >= 1 and ((nal[0] >> 1) & 0x3F) == 62:
                        yield nal
            i += 1
        if starts:
            buf = buf[starts[-1]:]
            starts = [0]
    proc.wait()
    if starts:
        nal = buf[starts[-1]+4:]
        if len(nal) >= 1 and ((nal[0] >> 1) & 0x3F) == 62:
            yield nal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="l1_data.csv")
    ap.add_argument("--start",    type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=99999.0)
    args = ap.parse_args()

    lib = load_lib()
    t0  = time.time()
    count = 0

    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_idx", "pts_approx", "l1_min_pq", "l1_max_pq", "l1_avg_pq", "scene_refresh"])

        for nal in iter_rpu_nals(args.input, args.start, args.duration):
            buf = (ctypes.c_uint8 * len(nal))(*nal)
            rpu = lib.dovi_parse_unspec62_nalu(buf, len(nal))
            if not rpu:
                count += 1
                continue
            err = lib.dovi_rpu_get_error(rpu)
            if err:
                lib.dovi_rpu_free(rpu)
                count += 1
                continue

            l1_min = l1_max = l1_avg = scene = None
            dm_ptr = lib.dovi_rpu_get_vdr_dm_data(rpu)
            if dm_ptr:
                dm = dm_ptr.contents
                scene = int(dm.scene_refresh_flag)
                l1 = dm.dm_data.level1
                if l1:
                    l1_min = l1.contents.min_pq
                    l1_max = l1.contents.max_pq
                    l1_avg = l1.contents.avg_pq
                lib.dovi_rpu_free_vdr_dm_data(dm_ptr)

            lib.dovi_rpu_free(rpu)
            pts = args.start + count / FPS
            writer.writerow([count, f"{pts:.6f}", l1_min, l1_max, l1_avg, scene])
            count += 1
            if count % 1000 == 0:
                print(f"  {count} frames  ({time.time()-t0:.1f}s)", flush=True)

    print(f"Done. {count} frames -> {args.output}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
