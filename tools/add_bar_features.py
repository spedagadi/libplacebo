"""
Post-process train/val datasets to add top_bar_norm and bottom_bar_norm features.
Uses the verified L5 active area offsets from RPU metadata.
Title keys match the 'title' column in the datasets.
"""
import os
DATA_DIR = os.environ.get("DTM_DATA_DIR", "F:/DTMModelData")

import pandas as pd
from pathlib import Path

# Verified L5 offsets: title_key -> (top_px/2160, bottom_px/2160)
# Consistent per-title (confirmed from all episodes)
TITLE_L5 = {
    # Train titles
    "andor":       (276/2160, 276/2160),  # 2.39:1 cinematic
    "euphoria":    (0.0, 0.0),            # 16:9 native
    "for":         (0.0, 0.0),            # 16:9 native
    "house":       (0.0, 0.0),            # 16:9 native
    "mindhunter":  (206/2160, 206/2160),  # ~2.20:1
    "monarch":     (0.0, 0.0),            # 16:9 native
    "sandman":     (276/2160, 276/2160),  # 2.39:1 (E01-E10), E11 is 0
    "stranger":    (120/2160, 120/2160),  # ~2.00:1
    "ted":         (0.0, 0.0),            # 16:9 native
    "the":         (0.0, 0.0),            # The Last of Us, 16:9 native
    "mandalorian": (276/2160, 276/2160),  # 2.39:1 cinematic
    "witcher":     (120/2160, 120/2160),  # ~2.00:1
    # Val titles
    "born":        (0.0, 0.0),            # 16:9 native
    "our":         (0.0, 0.0),            # 16:9 native
    "prehistoric": (0.0, 0.0),            # 16:9 native
    "rings":       (276/2160, 276/2160),  # 2.39:1 cinematic
    "wondla":      (0.0, 0.0),            # 16:9 animated
}

for split in ["val", "train"]:
    f = Path(f"" + DATA_DIR + "/{split}/{split}_dataset.csv")
    if not f.exists():
        print(f"{split}: not found, skipping")
        continue

    df = pd.read_csv(f)
    if "top_bar_norm" in df.columns:
        print(f"{split}: already has bar features, skipping")
        continue

    print(f"{split}: {len(df):,} scenes — adding top_bar_norm and bottom_bar_norm...", flush=True)

    df["top_bar_norm"]    = df["title"].map(lambda t: TITLE_L5.get(t, (0.0, 0.0))[0])
    df["bottom_bar_norm"] = df["title"].map(lambda t: TITLE_L5.get(t, (0.0, 0.0))[1])

    unknown = df[~df["title"].isin(TITLE_L5)]["title"].unique()
    if len(unknown):
        print(f"  WARNING: unknown titles defaulting to 0: {unknown}")

    df.to_csv(f, index=False)
    print(f"  Saved {f}  ({f.stat().st_size/1024/1024:.1f} MB)")
    print(f"  top_bar_norm values: {dict(df.groupby('title')['top_bar_norm'].first())}")
