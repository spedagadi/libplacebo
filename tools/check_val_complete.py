import os
DATA_DIR = os.environ.get("DTM_DATA_DIR", "F:/DTMModelData")
import pandas as pd
from pathlib import Path

manifest = pd.read_csv("" + DATA_DIR + "/stratification_manifest.csv")
val_titles = {'born', 'our', 'prehistoric', 'rings', 'wondla'}
val_manifest = manifest[manifest['title'].isin(val_titles)]

df = pd.read_csv("" + DATA_DIR + "/val/val_dataset.csv")

print(f"MANIFEST: {len(val_manifest):,} scenes across {val_manifest['title'].nunique()} titles")
print(f"DATASET:  {len(df):,} scenes across {df['title'].nunique()} titles")
print(f"Coverage: {len(df)/len(val_manifest)*100:.1f}%")

print("\nPer-title coverage:")
print(f"{'Title':<15} {'Manifest':>10} {'Extracted':>10} {'Missing':>10} {'OK?':>6}")
print("-" * 55)
all_ok = True
for title in sorted(val_titles):
    expected = len(val_manifest[val_manifest['title'] == title])
    got = len(df[df['title'] == title])
    missing = expected - got
    ok = missing == 0
    if not ok:
        all_ok = False
    print(f"  {title:<13} {expected:>10} {got:>10} {missing:>10} {'OK' if ok else 'MISSING':>6}")

print("-" * 55)
print(f"\nVal dataset {'COMPLETE' if all_ok else 'INCOMPLETE'}")

# Quick feature quality check
print(f"\nFeature quality:")
print(f"  NaN count:      {df.isna().sum().sum()}")
print(f"  maxscl range:   [{df['maxscl'].min():.4f}, {df['maxscl'].max():.4f}]")
print(f"  l1_max_pq range:[{df['l1_max_pq'].min():.0f}, {df['l1_max_pq'].max():.0f}]")
print(f"  Poly valid:     {(df['poly_num_segs']>0).sum():,}/{len(df):,}")
