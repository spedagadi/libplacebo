"""
Batch Stage 1 manifest extraction for all Cal/Train/Val titles.

Stage 1 (default): --no-pixels, --sample-fps 0 (ALL frames) — fast RPU-only
  pass that captures every frame's polynomial + L1 + scene_refresh. Stratification
  and Stage 2 frame selection are applied to this base manifest.

Stage 2: --full-pixels — adds pixel decode (maxscl, histogram, SAT zones).
  Slower; run after Stage 1 to fill in pixel feature columns.

Parallel: --workers N (default 2) runs N titles concurrently. Keep ≤3 for
  a single HDD source drive; more workers thrash disk with random seeks.

Usage:
    python tools/batch_extract.py                          # Stage 1, all frames, 2 workers
    python tools/batch_extract.py --workers 3              # 3 parallel titles
    python tools/batch_extract.py --splits train val       # specific splits
    python tools/batch_extract.py --full-pixels --workers 1 # Stage 2 (slower, serial)
    python tools/batch_extract.py --dry-run                # print commands only
    python tools/batch_extract.py --title rush             # single title rerun
"""

import subprocess
import argparse
import sys
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

EXTRACTOR = "C:/Code/libplacebo/tools/dv_metadata_extract.py"
OUT_ROOT  = "F:/DTMModelData"

# ---------------------------------------------------------------------------
# Title registry  — (split, short_name, source_path, is_iso)
# ---------------------------------------------------------------------------
TITLES = [
    # ---- Calibration ----
    ("calibration", "everest",
     r"G:\Everest.2015.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HDR10P.HEVC.HYBRID.REMUX-FraMeSToR.mkv", False),
    ("calibration", "hurt_locker",
     r"G:\Toedliches.Kommando.The.Hurt.Locker.2008.German.Custom.Atmos.Dubbed.2160p.UHD.BluRay.DV.HDR.HEVC.Remux-QfG.mkv", False),
    ("calibration", "troy_dc",
     r"G:\Troy.2004.Directors.Cut.2160p.UHD.Blu-ray.Remux.DV.HDR.HEVC.DTS-HD.MA.5.1-CiNEPHiLES.mkv", False),
    ("calibration", "john_wick_4",
     r"G:\John.Wick.Kapitel.4.2023.German.Atmos.DL.2160p.UHD.BluRay.DV.HDR.HEVC.Remux-NIMA4K.mkv", False),
    ("calibration", "kingdom_of_heaven_dc",
     r"G:\Koenigreich.der.Himmel.2005.Directors.Cut.Roadshow.Version.German.DL.2160p.UHD.BluRay.DV.HDR.HEVC.Remux-QfG.mkv", False),

    # ---- Train ----
    ("train", "atomic_blonde",
     r"G:\Atomic.Blonde.2017.MULTi.COMPLETE.UHD.BLURAY-OLDHAM", False),
    ("train", "alien_romulus",
     r"G:\Alien.Romulus.2024.2160p.COMPLETE.UHD.BLURAY-DOUHD", False),
    ("train", "furiosa",
     r"G:\Furiosa.A.Mad.Max.Saga.2024.2160p.MULTi.COMPLETE.UHD.BLURAY-GLiMMER", False),
    ("train", "warfare",
     r"G:\Warfare.2025.2160p.USA.UHD.Blu-ray.DV.HDR.HEVC.TrueHD.7.1.Atmos-TMT", False),
    ("train", "spotlight",
     r"G:\Spotlight.2015.UHD.BluRay.2160p.HEVC.DTS-HD.MA5.1-MTeam", False),
    ("train", "zodiac",
     r"G:\Zodiac.2007.UHD.BluRay.2160p.HEVC.TrueHD5.1-CHDBits", False),
    ("train", "rush",
     r"G:\Rush.2013.2160p.UHD.Blu-ray.Remux.HEVC.DV.TrueHD.7.1.Atmos-HDT\Rush 2013 2160p UHD Blu-ray Remux HEVC DV TrueHD 7.1 Atmos-HDT.mkv", False),
    ("train", "wonder_woman_1984",
     r"G:\Wonder.Woman.1984.2020.2160p.CEE.UHD.Blu-ray.HDR.DV.HEVC.TrueHD.7.1.Atmos", False),
    ("train", "first_blood",
     r"G:\First.Blood.1982.2160p.USA.UHD.Blu-ray.DV.HDR.HEVC.TrueHD.7.1.Atmos-BLoz", False),
    ("train", "pacific_rim",
     r"G:\Pacific Rim 2013 Hybrid 2160p UHD Blu-ray Remux DoVi HDR HEVC TrueHD 7.1 Atmos.mkv", False),
    ("train", "prometheus",
     r"G:\Prometheus.2012.UHD.BluRay.2160p.DTS-HD.MA.7.1.DV.HEVC.HYBRID.REMUX-FraMeSToR.mkv", False),
    ("train", "the_creator",
     r"G:\The Creator 2023 Hybrid 2160p UHD Blu-ray Remux DoVi HDR HEVC TrueHD 7.1 Atmos.mkv", False),
    ("train", "kingdom_of_apes",
     r"G:\Kingdom.of.the.Planet.of.the.Apes.2024.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.HYBRID.REMUX-FraMeSToR.mkv", False),
    ("train", "28_years_later",
     r"G:\28.Years.Later.2025.2160p.USA.UHD.Blu-ray.DV.HDR.HEVC.TrueHD.7.1.Atmos-TMT", False),

    # ---- Val ----
    ("val", "how_to_train_your_dragon",
     r"G:\How.to.Train.Your.Dragon.2025.2160p.COMPLETE.UHD.BLURAY-B3LLUM", False),
    ("val", "mi_final_reckoning",
     r"G:\Mission.Impossible-The.Final.Reckoning.2025.2160p.UHD.Blu-ray.HEVC.TrueHD-Tasko", False),
    ("val", "invisible_man",
     r"G:\The.Invisible.Man.2020.UHD.BluRay.2160p.HEVC.TrueHD.Atmos.7.1-BeyondHD", False),
    ("val", "tron_legacy",
     r"G:\Tron.Legacy.2010.2160p.USA.UHD.Blu-ray.DV.HDR.HEVC.TrueHD.7.1.Atmos-TMT", False),
    ("val", "f1_movie",
     r"G:\F1.The.Movie.2025.2160p.USA.UHD.Blu-ray.DV.HDR.HEVC.TrueHD.7.1.Atmos-TMT", False),
    ("val", "weapons",
     r"G:\Weapons.2025.Hybrid.2160p.UHD.Blu-ray.Remux.DV.HDR10plus.HEVC.TrueHD.Atmos.7.1-CiNEPHiLES.mkv", False),
    ("val", "ballerina",
     r"G:\Ballerina.2025.2160p.USA.UHD.Blu-ray.DV.HDR.HEVC.TrueHD.7.1.Atmos", False),
]


def run_ps(cmd):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                       capture_output=True, text=True, timeout=60)
    return r.stdout.strip(), r.stderr.strip()


def mount_iso(iso_path):
    out, err = run_ps(f'$d = Mount-DiskImage -ImagePath "{iso_path}" -PassThru; ($d | Get-Volume).DriveLetter')
    drive = out.strip()
    if not drive or len(drive) != 1:
        raise RuntimeError(f"Failed to mount {iso_path}: {err}")
    time.sleep(2)
    return drive + ":\\"


def dismount_iso(iso_path):
    run_ps(f'Dismount-DiskImage -ImagePath "{iso_path}"')


def extract_title(split, name, source, is_iso, no_pixels, dry_run):
    out_dir = Path(OUT_ROOT) / split
    out_csv = out_dir / f"{name}.csv"

    if out_csv.exists():
        print(f"  [skip] {name} — already exists ({out_csv})")
        return True

    mount_point = None
    actual_source = source

    if is_iso:
        if dry_run:
            print(f"  [dry] Would mount: {source}")
            actual_source = "<mount_point>"
        else:
            print(f"  Mounting ISO: {Path(source).name}")
            actual_source = mount_iso(source)
            mount_point = actual_source

    # Stage 1: all frames (sample-fps 0) so stratification has the full manifest.
    # Stage 2: same rate but with pixel decode enabled.
    sample_fps = "0" if no_pixels else "1"
    cmd = [
        sys.executable, EXTRACTOR,
        actual_source,
        "-o", str(out_csv),
        "--sample-fps", sample_fps,
        "--chunk-secs", "60",
    ]
    if no_pixels:
        cmd.append("--no-pixels")

    if dry_run:
        print(f"  [dry] {' '.join(str(c) for c in cmd)}")
        return True

    print(f"\n{'='*60}")
    print(f"  {split.upper()} | {name}")
    print(f"  Source: {actual_source}")
    print(f"  Output: {out_csv}")
    print(f"  Mode:   {'manifest (no pixels)' if no_pixels else 'full (pixels + RPU)'}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0
        ok = result.returncode == 0
        status = "OK" if ok else f"FAILED (rc={result.returncode})"
        print(f"  -> {status} in {elapsed/60:.1f} min")
        return ok
    finally:
        if mount_point:
            dismount_iso(source)
            print(f"  ISO dismounted.")


def main():
    ap = argparse.ArgumentParser(description="Batch Stage 1/2 extraction for all titles.")
    ap.add_argument("--splits", nargs="+",
                    choices=["calibration", "train", "val", "test"],
                    default=["calibration", "train", "val"],
                    help="Splits to process (default: calibration train val)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without running")
    ap.add_argument("--full-pixels", action="store_true",
                    help="Stage 2: include pixel decode (slower). Default is Stage 1 manifest only.")
    ap.add_argument("--title", metavar="NAME",
                    help="Process only this title (short name, e.g. 'rush')")
    ap.add_argument("--workers", type=int, default=2,
                    help="Parallel titles (default 2; keep ≤3 for single HDD source)")
    args = ap.parse_args()

    no_pixels = not args.full_pixels
    titles = [(s, n, p, i) for s, n, p, i in TITLES if s in args.splits]
    if args.title:
        titles = [(s, n, p, i) for s, n, p, i in titles if n == args.title]
        if not titles:
            print(f"Title '{args.title}' not found. Available: {[n for _,n,_,_ in TITLES]}")
            sys.exit(1)

    mode = "MANIFEST Stage 1 — all frames, no pixels" if no_pixels else "FULL PIXELS Stage 2 — 1fps + pixel decode"
    workers = 1 if args.dry_run else min(args.workers, len(titles))
    print(f"\nBatch extraction — {mode}")
    print(f"Splits: {args.splits}  |  Titles: {len(titles)}  |  Workers: {workers}")
    print(f"Output: {OUT_ROOT}\n")

    ok_count = 0
    fail_count = 0
    t_start = time.time()

    def _run(item):
        split, name, source, is_iso = item
        return name, extract_title(split, name, source, is_iso, no_pixels, args.dry_run)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, t): t[1] for t in titles}
        for fut in as_completed(futures):
            name, ok = fut.result()
            if ok:
                ok_count += 1
            else:
                fail_count += 1

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f} min. OK={ok_count}  Failed={fail_count}")


if __name__ == "__main__":
    main()
