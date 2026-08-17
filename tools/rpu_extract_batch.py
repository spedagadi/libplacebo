"""
Phase 0a: Batch RPU Extraction

Extract RPU binary files from all episodes in parallel, caching them for fast re-use.

Why separate extraction from parsing:
  1. Disk I/O bottleneck: Reading 9GB videos takes 90-120s per episode
  2. Parallelization: Run 4-8 dovi_tool processes simultaneously
  3. Caching: Small RPU files (~20 MB) can be reused for iterations
  4. Resume: Skip already-extracted episodes

Usage:
    # Extract all 14 titles in parallel (4 workers)
    python tools/rpu_extract_batch.py \
        --output-dir F:/DTMModelData/rpu/ \
        --workers 4

    # Extract single title
    python tools/rpu_extract_batch.py \
        --output-dir F:/DTMModelData/rpu/ \
        --title ted_lasso_s03 \
        --workers 4
"""
import subprocess
import argparse
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

DOVI_TOOL = r"C:\Users\Sateesh\Downloads\dovi_tool-2.3.3-x86_64-pc-windows-msvc\dovi_tool.exe"
DATASET_ROOT = r"G:\Dataset"

# Title registry (same as rpu_stratify.py)
TITLES = {
    # Category 1: Dark, Gritty & Shadow-Heavy
    "the_last_of_us_s02": "The.Last.of.Us.S02.2160p.MAX.WEB-DL.DDP5.1.DV.x265-NTb",
    "andor_s02": "Andor.S02.2160p.DSNP.WEB-DL.DDP5.1.DV.H.265-NTb",
    "house_of_the_dragon_s03": "House.of.the.Dragon.S03.2160p.HMAX.WEB-DL.DDP5.1.DV.H.265-NTb",

    # Category 2: Ultra-Vibrant, Saturated & Neon
    "stranger_things_s05": "Stranger.Things.S05.2160p.NF.WEB-DL.DDP5.1.DV.H.265-NTb",
    "monarch_s02": "Monarch.Legacy.of.Monsters.S02.2160p.ATVP.WEB-DL.DDP5.1.DV.H.265-NTb",
    "the_witcher_s04": "The.Witcher.S04.2160p.NF.WEB-DL.DDP5.1.DV.H.265-NTb",

    # Category 3: High Contrast, CGI & Specular Peaks
    "rings_of_power_s02": "The.Lord.of.the.Rings.The.Rings.of.Power.S02.2160p.AMZN.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX",
    "the_mandalorian_s01": "The.Mandalorian.S01.2160p.DSNP.WEB-DL.DDP5.1.Atmos.DV.HEVC-MZABI",
    "prehistoric_planet_s03": "Prehistoric.Planet.2022.S03.2160p.ATVP.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX",
    "for_all_mankind_s05": "For.All.Mankind.S05.2160p.ATVP.WEB-DL.DDP5.1.DV.H.265-NTb",

    # Category 4: Natural, Daylight & Cinematic Neutral
    "ted_lasso_s03": "Ted.Lasso.S03.2160p.ATVP.WEB-DL.DDP5.1.DoVi.H.265-NTb",
    "born_to_be_wild_s01": "Born.to.Be.Wild.2025.S01.2160p.ATVP.WEB-DL.DDP5.1.DV.HEVC-NTb",
    "our_living_world_s01": "Our.Living.World.S01.2160p.NF.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX",
    "our_oceans_s01": "Our.Oceans.(2024).S01.(2160p.NF.WEB-DL.H265.DV.DDP.Atmos.5.1.English.-.HONE)",
    "our_planet_s01": "Our.Planet.2019.S01.2160p.NF.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX",
    "wondla_s03": "WondLa.S03.2160p.ATVP.WEB-DL.DDP5.1.DV.HEVC-BYNDR",

    # New titles (added 2026-08-15)
    "euphoria_s03": "Euphoria.US.S03.2160p.HMAX.WEB-DL.DDP5.1.DV.H.265-NTb",
    "mindhunter_s01": "Mindhunter.S01.2160p.NF.WEB-DL.DDP5.1.DV.H.265-Kitsune",
    "sandman_s01": "The.Sandman.S01.2160p.NF.WEB-DL.DDP.5.1.Atmos.DV.H.265-CHDWEB",
}


def log(msg):
    """Thread-safe logging with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def extract_rpu(mkv_path, output_rpu, title_name, episode_stem):
    """
    Extract RPU binary from a single episode.
    Returns: (success: bool, time_seconds: float, error_msg: str)
    """
    file_size_mb = mkv_path.stat().st_size / (1024 * 1024)

    # Skip if already exists and non-empty
    if output_rpu.exists() and output_rpu.stat().st_size > 1_000_000:
        rpu_size_mb = output_rpu.stat().st_size / (1024 * 1024)
        return True, 0.0, f"Already exists ({rpu_size_mb:.1f} MB)"

    log(f"  [{title_name}] {episode_stem} ({file_size_mb:.0f} MB) - Starting extraction...")

    t0 = time.time()
    try:
        result = subprocess.run(
            [DOVI_TOOL, "extract-rpu", str(mkv_path), "-o", str(output_rpu)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1200
        )
        elapsed = time.time() - t0

        if result.returncode != 0 or not output_rpu.exists():
            return False, elapsed, f"dovi_tool failed (returncode: {result.returncode})"

        rpu_size_mb = output_rpu.stat().st_size / (1024 * 1024)
        throughput = file_size_mb / elapsed if elapsed > 0 else 0
        log(f"  [{title_name}] {episode_stem} - DONE: {rpu_size_mb:.1f} MB in {elapsed:.0f}s ({throughput:.0f} MB/s)")
        return True, elapsed, None

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return False, elapsed, "Timeout (1200s exceeded)"
    except Exception as e:
        elapsed = time.time() - t0
        return False, elapsed, f"Exception: {type(e).__name__}: {e}"




def main():
    ap = argparse.ArgumentParser(description="Phase 0a: Batch RPU extraction")
    ap.add_argument("-o", "--output-dir", required=True,
                    help="Output directory for .rpu files")
    ap.add_argument("--title", help="Process single title only")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel dovi_tool processes (default: 4)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(DATASET_ROOT)

    # Select titles to process
    if args.title:
        if args.title not in TITLES:
            log(f"Error: Unknown title '{args.title}'")
            log(f"Available: {list(TITLES.keys())}")
            sys.exit(1)
        titles_to_scan = {args.title: TITLES[args.title]}
    else:
        titles_to_scan = TITLES

    log(f"Phase 0a: Batch RPU Extraction")
    log(f"Output directory: {output_dir}")
    log(f"Titles: {len(titles_to_scan)}")
    log(f"Workers: {args.workers}")
    log("")

    # Collect all episodes across all titles
    all_episodes = []
    for short_name, folder_name in titles_to_scan.items():
        title_dir = dataset_root / folder_name
        if not title_dir.exists():
            log(f"[SKIP] {short_name} - folder not found: {title_dir}")
            continue

        episodes = sorted(title_dir.glob("*.mkv")) or sorted(title_dir.glob("*.mp4"))
        log(f"[{short_name}] Found {len(episodes)} episodes")

        for ep in episodes:
            output_rpu = output_dir / f"{short_name}_{ep.stem}.rpu"
            all_episodes.append((ep, output_rpu, short_name, ep.stem))

    log(f"\nTotal episodes to process: {len(all_episodes)}")
    log(f"Starting extraction with {args.workers} workers...\n")

    # Process in parallel (using threads for I/O-bound work)
    t_start = time.time()
    success_count = 0
    error_count = 0
    skip_count = 0
    total_extraction_time = 0.0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {}
        for ep_args in all_episodes:
            mkv_path, output_rpu, title_name, episode_stem = ep_args
            future = executor.submit(extract_rpu, mkv_path, output_rpu, title_name, episode_stem)
            futures[future] = ep_args

        for future in as_completed(futures):
            success, elapsed, error = future.result()

            if error and "Already exists" in error:
                skip_count += 1
            elif success:
                success_count += 1
                total_extraction_time += elapsed
            else:
                error_count += 1
                ep_args = futures[future]
                log(f"  [ERROR] {ep_args[2]} / {ep_args[3]}: {error}")

    elapsed_total = time.time() - t_start

    log("")
    log("=" * 70)
    log("Phase 0a Complete")
    log(f"  Extracted: {success_count}")
    log(f"  Skipped (cached): {skip_count}")
    log(f"  Errors: {error_count}")
    log(f"  Total wall-clock time: {elapsed_total/60:.1f} minutes")
    if success_count > 0:
        avg_time = total_extraction_time / success_count
        log(f"  Average extraction time: {avg_time:.0f} seconds")
        log(f"  Parallel speedup: {total_extraction_time/elapsed_total:.1f}x")
    log(f"  RPU files: {output_dir}")
    log("=" * 70)


if __name__ == "__main__":
    main()
