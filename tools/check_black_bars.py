"""
Check L5 active area offsets from RPU across all titles to understand
black bar prevalence. Also analyse the val dataset's zone features
to see how much black bars contaminate top/bottom SAT cells.
"""
import os
DATA_DIR = os.environ.get("DTM_DATA_DIR", "F:/DTMModelData")

import subprocess, json, tempfile
from pathlib import Path
import pandas as pd, numpy as np

DOVI = r"C:\Users\Sateesh\Downloads\dovi_tool-2.3.3-x86_64-pc-windows-msvc\dovi_tool.exe"
RPU_DIR = Path("" + DATA_DIR + "/rpu")

print("L5 ACTIVE AREA OFFSETS (black bar detection)")
print("=" * 65)

results = []
for rpu in sorted(RPU_DIR.glob("*.rpu")):
    if rpu.stat().st_size < 1_000_000:
        continue
    # Sample first frame only
    r = subprocess.run([DOVI, "info", "-i", str(rpu), "-f", "0"],
                       capture_output=True, text=True, timeout=10)
    try:
        data = json.loads(r.stdout.split('\n',1)[1])
        dm = data.get("vdr_dm_data", {})
        blocks = dm.get("cmv29_metadata", {}).get("ext_metadata_blocks", [])
        l5 = next((b["Level5"] for b in blocks if "Level5" in b), None)
        if l5:
            top    = l5.get("active_area_top_offset", 0)
            bottom = l5.get("active_area_bottom_offset", 0)
            left   = l5.get("active_area_left_offset", 0)
            right  = l5.get("active_area_right_offset", 0)

            title = "_".join(rpu.stem.split("_")[:2])
            results.append({"title": title, "top": top, "bottom": bottom,
                           "left": left, "right": right,
                           "has_bars": top > 0 or bottom > 0})
    except Exception:
        pass

df = pd.DataFrame(results)

# Group by title prefix
print(f"\n{'Title':<30} {'top':>6} {'bottom':>8} {'left':>6} {'right':>7} {'bars?':>6}")
print("-" * 65)
for title, grp in df.groupby("title"):
    t = grp["top"].mean()
    b = grp["bottom"].mean()
    bars = (grp["has_bars"].mean() * 100)
    print(f"  {title:<28} {t:>6.0f} {b:>8.0f} {grp['left'].mean():>6.0f} "
          f"{grp['right'].mean():>7.0f} {bars:>5.0f}%")

print(f"\nTitles WITH black bars: {(df.groupby('title')['has_bars'].any()).sum()} / {df['title'].nunique()}")

# Analyse zone contamination in val dataset
print("\n\nZONE CONTAMINATION ANALYSIS (val dataset)")
print("=" * 65)
val = pd.read_csv("" + DATA_DIR + "/val/val_dataset.csv")

# Compare top row vs middle row vs bottom row zone means
# zone_mean_r0 = top, zone_mean_r1 = middle, zone_mean_r2 = bottom
top_cols = [f"zone_mean_r0_c{c}" for c in range(3)]
mid_cols  = [f"zone_mean_r1_c{c}" for c in range(3)]
bot_cols  = [f"zone_mean_r2_c{c}" for c in range(3)]

avail_top = [c for c in top_cols if c in val.columns]
avail_mid = [c for c in mid_cols if c in val.columns]
avail_bot = [c for c in bot_cols if c in val.columns]

if avail_top and avail_mid:
    top_mean = val[avail_top].mean(axis=1).mean()
    mid_mean = val[avail_mid].mean(axis=1).mean()
    bot_mean = val[avail_bot].mean(axis=1).mean() if avail_bot else 0

    print(f"Mean luma by row:")
    print(f"  Top row (r0): {top_mean:.4f}")
    print(f"  Mid row (r1): {mid_mean:.4f}")
    print(f"  Bot row (r2): {bot_mean:.4f}")
    print(f"  Top/Mid ratio: {top_mean/mid_mean:.3f}  (1.0 = no bars, <0.3 = heavy bars)")

    # Detect likely letterboxed scenes: top row <<< middle row
    ratio = val[avail_top].mean(axis=1) / (val[avail_mid].mean(axis=1) + 1e-6)
    letterboxed = (ratio < 0.15).sum()
    print(f"\nScenes likely letterboxed (top < 15% of mid): {letterboxed}/{len(val)} ({100*letterboxed/len(val):.1f}%)")
    print(f"Per title:")
    for title in val['title'].unique():
        t_df = val[val['title']==title]
        r = t_df[avail_top].mean(axis=1) / (t_df[avail_mid].mean(axis=1) + 1e-6)
        lb = (r < 0.15).sum()
        print(f"  {title:<15} {lb:>5}/{len(t_df):>5} ({100*lb/len(t_df):.1f}%)")
