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
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

EXTRACTOR = "C:/Code/libplacebo/tools/dv_metadata_extract.py"
OUT_ROOT  = "F:/DTMModelData"

# ---------------------------------------------------------------------------
# Title registry  — (split, short_name, source_path, is_iso)
# ---------------------------------------------------------------------------
# Aug 2026: BDMV disc titles removed — confirmed 100% identity luma polynomials
# (HDR10 base layer already tone-mapped in mastering; DV is colour-matrix-only
# on disc). P5 streaming corpus is now primary — real per-scene polynomials
# authored directly against the streaming encode. See ml/README.md.
#
# Source paths are folders of per-episode MKVs under G:\Dataset\ (real release
# folder names, confirmed on disk Aug 2026). extract_title() detects a loose
# episode folder (vs a single file or BDMV disc) and iterates episodes itself.
#
# Note: Witcher downloaded as S04, not the originally planned S02 — S04 is
# what's on disk, kept as-is (see ml/README.md).

TITLES = [
    # ---- Train ----
    ("train", "andor_s02",
     r"G:\Dataset\Andor.S02.2160p.DSNP.WEB-DL.DDP5.1.DV.H.265-NTb", False),
    ("train", "euphoria_s03",
     r"G:\Dataset\Euphoria.US.S03.2160p.HMAX.WEB-DL.DDP5.1.DV.H.265-NTb", False),
    ("train", "for_all_mankind_s05",
     r"G:\Dataset\For.All.Mankind.S05.2160p.ATVP.WEB-DL.DDP5.1.DV.H.265-NTb", False),
    ("train", "house_of_the_dragon_s03",
     r"G:\Dataset\House.of.the.Dragon.S03.2160p.HMAX.WEB-DL.DDP5.1.DV.H.265-NTb", False),
    ("train", "mindhunter_s01",
     r"G:\Dataset\Mindhunter.S01.2160p.NF.WEB-DL.DDP5.1.DV.H.265-Kitsune", False),
    ("train", "monarch_s02",
     r"G:\Dataset\Monarch.Legacy.of.Monsters.S02.2160p.ATVP.WEB-DL.DDP5.1.DV.H.265-NTb", False),
    ("train", "sandman_s01",
     r"G:\Dataset\The.Sandman.S01.2160p.NF.WEB-DL.DDP.5.1.Atmos.DV.H.265-CHDWEB", False),
    ("train", "stranger_things_s05",
     r"G:\Dataset\Stranger.Things.S05.2160p.NF.WEB-DL.DDP5.1.DV.H.265-NTb", False),
    ("train", "ted_lasso_s03",
     r"G:\Dataset\Ted.Lasso.S03.2160p.ATVP.WEB-DL.DDP5.1.DoVi.H.265-NTb", False),
    ("train", "the_last_of_us_s02",
     r"G:\Dataset\The.Last.of.Us.S02.2160p.MAX.WEB-DL.DDP5.1.DV.x265-NTb", False),
    ("train", "the_mandalorian_s01",
     r"G:\Dataset\The.Mandalorian.S01.2160p.DSNP.WEB-DL.DDP5.1.Atmos.DV.HEVC-MZABI", False),
    ("train", "the_witcher_s04",
     r"G:\Dataset\The.Witcher.S04.2160p.NF.WEB-DL.DDP5.1.DV.H.265-NTb", False),
    # Nature documentaries — bright outdoor S-curve content (fills bright/S-curve training gap)
    # No stratification cap applied — full extraction to maximise boost-curve coverage
    ("train", "our_living_world_s01",
     r"G:\Dataset\Our.Living.World.S01.2160p.NF.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX", False),
    ("train", "our_oceans_s01",
     r"G:\Dataset\Our.Oceans.(2024).S01.(2160p.NF.WEB-DL.H265.DV.DDP.Atmos.5.1.English.-.HONE)", False),

    # ---- Val ----
    ("val", "born_to_be_wild_s01",
     r"G:\Dataset\Born.to.Be.Wild.2025.S01.2160p.ATVP.WEB-DL.DDP5.1.DV.HEVC-NTb", False),
    ("val", "our_planet_s01",
     r"G:\Dataset\Our.Planet.2019.S01.2160p.NF.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX", False),
    ("val", "prehistoric_planet_s03",
     r"G:\Dataset\Prehistoric.Planet.2022.S03.2160p.ATVP.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX", False),
    ("val", "rings_of_power_s02",
     r"G:\Dataset\The.Lord.of.the.Rings.The.Rings.of.Power.S02.2160p.AMZN.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX", False),
    ("val", "wondla_s03",
     r"G:\Dataset\WondLa.S03.2160p.ATVP.WEB-DL.DDP5.1.DV.HEVC-BYNDR", False),
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


def _is_complete(csv_path: Path, min_rows: int = 1000) -> bool:
    """Heuristic: a CSV with >min_rows is likely fully extracted."""
    try:
        with open(csv_path, "r") as f:
            count = sum(1 for _ in f)
        return count > min_rows
    except Exception:
        return False


def _concat_episodes(ep_dir: Path, out_csv: Path):
    """Concatenate per-episode CSVs into one title CSV.

    Each episode's scene_id restarts at 0, so a plain concat would collide
    scene_ids across episodes. Tag every row with an 'episode' column instead
    of touching scene_id — GroupKFold should group on (episode, scene_id).
    """
    ep_csvs = sorted(ep_dir.glob("*.csv"))
    if not ep_csvs:
        return
    writer = None
    with open(out_csv, "w", newline="") as out_f:
        for ep_csv in ep_csvs:
            with open(ep_csv, "r", newline="") as in_f:
                reader = csv.DictReader(in_f)
                if reader.fieldnames is None:
                    continue
                if writer is None:
                    writer = csv.DictWriter(out_f, fieldnames=["episode"] + list(reader.fieldnames))
                    writer.writeheader()
                for row in reader:
                    row["episode"] = ep_csv.stem
                    writer.writerow(row)
    print(f"  [concat] {len(ep_csvs)} episode CSV(s) -> {out_csv}")


def extract_episode_dir(split, name, source_dir, no_pixels, dry_run, nvdec=False):
    """Iterate loose per-episode MKVs in source_dir (P5 streaming season folders
    have no BDMV structure, so the extractor must run once per episode file)."""
    episodes = sorted(source_dir.glob("*.mkv")) or sorted(source_dir.glob("*.mp4"))
    if not episodes:
        print(f"  [skip]   {name} — no .mkv/.mp4 files found in {source_dir}")
        return False

    out_dir = Path(OUT_ROOT) / split
    out_csv = out_dir / f"{name}.csv"
    ep_dir = out_dir / f"{name}_episodes"
    if not dry_run:
        ep_dir.mkdir(parents=True, exist_ok=True)

    sample_fps = "0" if no_pixels else "1"
    all_ok = True
    for ep in episodes:
        ep_csv = ep_dir / f"{ep.stem}.csv"
        resume = False
        if not dry_run and ep_csv.exists():
            size = ep_csv.stat().st_size
            if size <= 512:
                ep_csv.unlink()
            elif _is_complete(ep_csv, min_rows=2000):
                print(f"  [done]   {name}/{ep.stem} — complete, skipping")
                continue
            else:
                resume = True

        cmd = [sys.executable, EXTRACTOR, str(ep), "-o", str(ep_csv),
               "--sample-fps", sample_fps, "--chunk-secs", "60"]
        if no_pixels:
            cmd.append("--no-pixels")
        if resume:
            cmd.append("--resume")
        if not no_pixels and nvdec:
            cmd.append("--nvdec")

        if dry_run:
            print(f"  [dry] {' '.join(str(c) for c in cmd)}")
            continue

        print(f"\n{'='*60}")
        print(f"  {split.upper()} | {name} / {ep.stem}")
        print(f"{'='*60}")
        t0 = time.time()
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0
        ok = result.returncode == 0
        print(f"  -> {'OK' if ok else f'FAILED (rc={result.returncode})'} in {elapsed/60:.1f} min")
        all_ok = all_ok and ok

    if dry_run:
        return True

    _concat_episodes(ep_dir, out_csv)
    return all_ok


def extract_title(split, name, source, is_iso, no_pixels, dry_run, nvdec=False):
    out_dir = Path(OUT_ROOT) / split
    out_csv = out_dir / f"{name}.csv"

    source_path = Path(source)
    if not is_iso and source_path.is_dir() and not any(source_path.rglob("*.m2ts")) \
            and (any(source_path.glob("*.mkv")) or any(source_path.glob("*.mp4"))):
        return extract_episode_dir(split, name, source_path, no_pixels, dry_run, nvdec=nvdec)

    resume = False
    if out_csv.exists():
        size = out_csv.stat().st_size
        if size <= 512:
            out_csv.unlink()
        elif _is_complete(out_csv):
            print(f"  [done]   {name} — complete ({size//1024}KB), skipping")
            return True
        else:
            print(f"  [resume] {name} — partial CSV {size//1024}KB, resuming from last row")
            resume = True

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
    if resume:
        cmd.append("--resume")
    if not no_pixels and nvdec:
        cmd.append("--nvdec")

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
    ap.add_argument("--nvdec", action="store_true",
                    help="Use NVDEC hardware decode for Stage 2 (~4x faster on RTX cards).")
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
        return name, extract_title(split, name, source, is_iso, no_pixels,
                                   args.dry_run, nvdec=args.nvdec)

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
