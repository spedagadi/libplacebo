import os
DATA_DIR = os.environ.get("DTM_DATA_DIR", "F:/DTMModelData")
import pandas as pd, numpy as np

for split in ["val", "train"]:
    f = f"" + DATA_DIR + "/{split}/{split}_dataset.csv"
    df = pd.read_csv(f)
    print(f"\n{split.upper()}: {len(df):,} scenes | {df['title'].nunique()} titles | {len(df.columns)} cols")

    issues = []

    # NaN check on key columns
    for col in ['maxscl','average_maxrgb','l1_max_pq','poly_num_segs','seg0_c0']:
        if col in df.columns and df[col].isna().any():
            issues.append(f"NaN in {col}")

    # Range check pixels [0,1]
    for col in ['maxscl','average_maxrgb','fraction_bright_pixels']:
        if col in df.columns:
            if df[col].min() < -0.01 or df[col].max() > 1.01:
                issues.append(f"{col} out of [0,1]: [{df[col].min():.3f}, {df[col].max():.3f}]")

    # L1 should be non-zero
    if 'l1_max_pq' in df.columns:
        zeros = (df['l1_max_pq'] == 0).sum()
        if zeros > 0:
            issues.append(f"l1_max_pq has {zeros} zeros")

    # All scenes should have valid poly
    if 'poly_num_segs' in df.columns:
        bad = (df['poly_num_segs'] == 0).sum()
        if bad > 0:
            issues.append(f"{bad} scenes with poly_num_segs=0")

    # Trim columns present
    for pq in ['2081','2851','3079']:
        col = f'trim_{pq}_slope'
        if col not in df.columns:
            issues.append(f"missing {col}")

    if issues:
        print(f"  ISSUES: {issues}")
    else:
        print(f"  [OK] No schema issues")

    # Key stats
    print(f"  maxscl:     [{df['maxscl'].min():.4f}, {df['maxscl'].max():.4f}]  mean={df['maxscl'].mean():.4f}")
    print(f"  l1_max_pq:  [{df['l1_max_pq'].min():.0f}, {df['l1_max_pq'].max():.0f}]  mean={df['l1_max_pq'].mean():.0f}")
    print(f"  poly valid: {(df['poly_num_segs']>0).sum()}/{len(df)}")
    corr = df['maxscl'].corr(df['l1_max_pq']/4095.0)
    print(f"  maxscl/L1 corr: {corr:.3f}")
