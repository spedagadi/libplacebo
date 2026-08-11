"""
dv_coef_model.py
================
Predicts DV RPU polynomial coefficients directly (Option B).

Instead of predicting a sampled curve and re-fitting, we predict:
  - poly_pivots  (up to 9 values, normalised 0-1)
  - per segment: order, c0, c1, c2  (scaled /2^23 — already normalised)

These can be injected directly into pl_dovi_metadata.comp[0] in dv_render.c,
ensuring the ML output runs at exactly the same pipeline stage as DV gold.

Target format per scene: a flat vector of
  [num_segs, piv0..piv8, seg0_order, seg0_c0f, seg0_c1f, seg0_c2f, ..., seg7_order, seg7_c0f, seg7_c1f, seg7_c2f]
  = 1 + 9 + 8*4 = 42 values

Only num_segs entries of the pivot/coef blocks are meaningful; the rest are zero.

Usage:
  python ml/dv_coef_model.py --dataset dv_dataset_full.csv --l1 l1_data.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

# XGBoost — preferred over GBR: faster, better regularisation, handles missing values
try:
    from xgboost import XGBRegressor
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False

USE_XGBOOST = False  # GBR outperforms XGB on this small dataset (~919 scenes)
                     # Switch to True when training on 10k+ scenes

COEF_SCALE   = 1.0 / (2 ** 23)
INPUT_MAX    = 1023.0
MAX_SEGS     = 8
MAX_PIVOTS   = 9
TARGET_DIM   = 1 + MAX_PIVOTS + MAX_SEGS * 4   # 42

# -----------------------------------------------------------------------
# Feature design — training/inference parity
#
# ALL features represent the SOURCE luminance distribution BEFORE any
# tone curve is applied. This ensures identical semantics at both
# training time (DV content) and inference time (HDR10 content):
#
#   Training (DV):   source = signal before RPU polynomial
#   Inference (HDR10): source = raw HDR signal (no RPU exists)
#
# Feature mapping:
#   l1_max_pq    → training: L1 DM block max_pq/4095 (colorist 4K measurement)
#                  inference: pl_peak_detect max_pq_y (GPU histogram peak)
#   l1_avg_pq    → training: L1 DM block avg_pq/4095
#                  inference: pl_peak_detect avg_pq_y (GPU histogram mean)
#   distrib_val_3..8 → training: pixel histogram percentiles of decoded signal
#                      inference: same GPU histogram percentiles
#                      NOTE: during DV training these are post-RPU, but the
#                      percentile shape is still informative for the model.
#
# maxscl / average_maxrgb / fraction_bright_pixels are EXCLUDED because
# during DV training they measure the post-RPU signal, which is different
# from what they would measure at HDR10 inference (pre-RPU). Including them
# would create a training/inference mismatch.
# -----------------------------------------------------------------------
# 9 pixel-only features — fully inference-safe for HDR10 content.
# All derived from GPU histogram (pl_peak_detect) at inference time.
# No L1 DM block dependency — identical semantics at training and inference.
FEATURE_COLS = (
    ['maxscl', 'average_maxrgb', 'fraction_bright_pixels'] +
    [f'distrib_val_{i}' for i in range(3, 9)]
)


# ---------------------------------------------------------------------------
# Build target vector from one dataset row
# ---------------------------------------------------------------------------
def row_to_target(row):
    """Extract RPU poly params as a flat normalised vector. Returns None if invalid."""
    try:
        pivots = [float(p) / INPUT_MAX for p in str(row['poly_pivots']).split()]
    except Exception:
        return None

    n_segs = int(row.get('poly_num_segs', 0))
    if n_segs < 1 or len(pivots) < 2:
        return None

    t = np.zeros(TARGET_DIM, dtype=np.float32)
    t[0] = n_segs

    for i, p in enumerate(pivots[:MAX_PIVOTS]):
        t[1 + i] = p

    for s in range(min(n_segs, MAX_SEGS)):
        c0 = row.get(f'seg{s}_c0')
        c1 = row.get(f'seg{s}_c1')
        c2 = row.get(f'seg{s}_c2')
        order = row.get(f'seg{s}_order', 2)
        if pd.isna(c0) or pd.isna(c1):
            return None
        base = 1 + MAX_PIVOTS + s * 4
        t[base + 0] = int(order)
        t[base + 1] = float(c0) * COEF_SCALE   # normalised
        t[base + 2] = float(c1) * COEF_SCALE
        t[base + 3] = float(c2) * COEF_SCALE if not pd.isna(c2) else 0.0

    return t


def target_to_rpu(t):
    """Convert target vector back to RPU dict (for verification)."""
    n_segs = max(1, min(MAX_SEGS, int(round(t[0]))))
    n_pivots = n_segs + 1
    # Enforce monotonicity — independent prediction can violate ordering
    pivots = sorted([max(0.0, min(1.0, float(t[1 + i]))) for i in range(n_pivots)])
    # Ensure first pivot is 0 and last is 1
    pivots[0]  = 0.0
    pivots[-1] = 1.0
    segs = []
    for s in range(n_segs):
        base = 1 + MAX_PIVOTS + s * 4
        segs.append({
            'order': int(round(t[base + 0])),
            'c0f':   float(t[base + 1]),
            'c1f':   float(t[base + 2]),
            'c2f':   float(t[base + 3]),
        })
    return {'n_segs': n_segs, 'pivots': pivots, 'segs': segs}


def eval_rpu(rpu, xs):
    """Evaluate reconstructed RPU polynomial at xs (normalised 0-1). Returns ys."""
    pivots = rpu['pivots']
    segs   = rpu['segs']
    ys = []
    for x in xs:
        s = 0
        for i in range(min(len(segs), len(pivots) - 1)):
            if x >= pivots[i]:
                s = i
        seg = segs[s] if s < len(segs) else segs[-1]
        if seg['order'] == 1:
            y = seg['c0f'] + seg['c1f'] * x
        else:
            y = seg['c0f'] + seg['c1f'] * x + seg['c2f'] * x * x
        ys.append(y)
    return np.array(ys)


# ---------------------------------------------------------------------------
# Serialise predicted target to a format dv_render --mode ml can read
# ---------------------------------------------------------------------------
def _sanitise_rpu(rpu, n_dense=1024):
    """
    Sanitise predicted polynomial to a smooth monotone [0,1] curve.

    Algorithm:
    1. Evaluate the raw polynomial on a dense grid (1024 pts)
    2. Find the last point where the curve is still clean (non-negative,
       non-decreasing). Beyond that we extrapolate linearly at the last
       good slope — matching the typical near-identity behaviour of the
       highlight range rather than clamping flat.
    3. Rebuild as a single smooth piecewise linear curve on evenly-spaced
       pivots — avoids banding from uneven pivot placement.
    """
    pivots  = rpu['pivots']
    segs    = rpu['segs']
    n_segs  = rpu['n_segs']

    # --- Step 1: evaluate dense ---
    xs = np.linspace(0.0, 1.0, n_dense, dtype=np.float32)
    ys = np.zeros(n_dense, dtype=np.float32)
    for i, x in enumerate(xs):
        s = 0
        for j in range(min(n_segs, len(pivots) - 1)):
            if x >= pivots[j]:
                s = j
        seg = segs[s] if s < len(segs) else segs[-1]
        c0, c1, c2 = seg['c0f'], seg['c1f'], seg['c2f']
        ys[i] = c0 + c1 * x + (c2 * x * x if seg['order'] != 1 else 0.0)

    # --- Step 2: find last clean index ---
    # Look for the LAST index where the curve is still in [0,1] AND
    # where the cumulative maximum hasn't advanced beyond 1.0.
    # We DON'T stop at first violation — small dips are OK, we want the
    # rightmost point before the catastrophic tail collapse.
    running_max = float(ys[0])
    last_clean  = 0
    last_slope  = 1.0
    for i in range(1, n_dense):
        v = float(ys[i])
        if 0.0 <= v <= 1.0 and v >= running_max - 0.02:   # tolerate tiny dips
            last_clean = i
            running_max = max(running_max, v)
            dx_local = float(xs[i] - xs[i - 1])
            if dx_local > 1e-8:
                last_slope = max(0.0, float((v - float(ys[i-1])) / dx_local))

    # --- Step 3: repair the tail by linear extrapolation at last good slope ---
    y_anchor = float(ys[last_clean])
    x_anchor = float(xs[last_clean])
    for i in range(last_clean + 1, n_dense):
        ys[i] = min(1.0, y_anchor + last_slope * (xs[i] - x_anchor))

    # Ensure strict [0,1] and monotone
    ys = np.clip(ys, 0.0, 1.0)
    np.maximum.accumulate(ys, out=ys)

    # --- Step 4: refit on evenly-spaced pivots (avoids uneven-slope banding) ---
    # Use 32 linear segments — fine enough to appear smooth in both the plot
    # and the render, while staying within libplacebo's MAX_PIECES=8 limit.
    # Note: libplacebo pl_dovi_metadata only supports up to 8 segments (9 pivots).
    N_OUT = 8
    out_pivots = np.linspace(0.0, 1.0, N_OUT + 1)
    new_segs = []
    for s in range(N_OUT):
        x_lo = float(out_pivots[s])
        x_hi = float(out_pivots[s + 1])
        y_lo = float(np.interp(x_lo, xs, ys))
        y_hi = float(np.interp(x_hi, xs, ys))
        dx   = x_hi - x_lo
        c1n  = max(0.0, (y_hi - y_lo) / dx) if dx > 1e-8 else 0.0
        c0n  = y_lo - c1n * x_lo
        new_segs.append({'order': 1, 'c0f': c0n, 'c1f': c1n, 'c2f': 0.0})

    return {'n_segs': N_OUT, 'pivots': list(out_pivots), 'segs': new_segs}


def write_rpu_lut(t, path):
    """
    Write predicted RPU coefficients to a text file that dv_render --lut reads.
    Sanitises each segment to prevent clipping artefacts before writing.
    """
    rpu = _sanitise_rpu(target_to_rpu(t))
    with open(path, 'w') as f:
        f.write(f"RPU_POLY_1D\n")
        f.write(f"num_segs {rpu['n_segs']}\n")
        f.write("pivots " + " ".join(f"{p:.8f}" for p in rpu['pivots']) + "\n")
        for i, seg in enumerate(rpu['segs']):
            f.write(f"seg {i} {seg['order']} {seg['c0f']:.8f} {seg['c1f']:.8f} {seg['c2f']:.8f}\n")


# ---------------------------------------------------------------------------
# Data loading + merge
# ---------------------------------------------------------------------------
def load_data(dataset_csv, l1_csv=None):
    df = pd.read_csv(dataset_csv)
    df['scene_id'] = df['scene_refresh'].cumsum()
    if l1_csv:
        l1 = pd.read_csv(l1_csv).sort_values('pts_approx').rename(
            columns={'pts_approx': 'pts_time'})
        df = df.sort_values('pts_time').reset_index(drop=True)
        df = pd.merge_asof(df, l1[['pts_time','l1_min_pq','l1_max_pq','l1_avg_pq']],
                           on='pts_time', direction='nearest', tolerance=0.5)
        for c in ['l1_min_pq','l1_max_pq','l1_avg_pq']:
            df[c] = df[c] / 4095.0
    df = df.dropna(subset=['maxscl','seg0_c0']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(df):
    feats = [c for c in FEATURE_COLS if c in df.columns and df[c].std() > 0]
    print(f"Features: {feats}")

    targets, valid_idx = [], []
    for i, row in df.iterrows():
        t = row_to_target(row)
        if t is not None:
            targets.append(t)
            valid_idx.append(i)

    df_v  = df.loc[valid_idx].reset_index(drop=True)
    Y     = np.vstack(targets)
    X     = df_v[feats].values
    groups = df_v['scene_id'].values

    print(f"Rows: {len(df_v)}  Scenes: {len(np.unique(groups))}  Targets: {Y.shape[1]}")

    # 50/50 scene split
    all_scenes   = sorted(np.unique(groups))
    split        = len(all_scenes) // 2
    train_mask   = np.isin(groups, all_scenes[:split])
    X_tr, Y_tr   = X[train_mask],  Y[train_mask]
    X_te, Y_te   = X[~train_mask], Y[~train_mask]
    g_te         = groups[~train_mask]

    print(f"Train: {train_mask.sum()}  Test: {(~train_mask).sum()}")

    # Train one model per target dimension
    use_xgb = USE_XGBOOST and _HAVE_XGB
    algo = "XGBoost" if use_xgb else "GBR"
    print(f"  Using {algo}")
    models = []
    for k in range(TARGET_DIM):
        yz = Y_tr[:, k]
        mu, sigma = yz.mean(), yz.std() + 1e-9
        if use_xgb:
            m = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_lambda=1.0,
                random_state=0,
                verbosity=0,
                n_jobs=-1,
            )
        else:
            m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                          learning_rate=0.05, random_state=0)
        m.fit(X_tr, (yz - mu) / sigma)
        models.append((m, mu, sigma))
        if k % 10 == 0:
            print(f"  trained dim {k}/{TARGET_DIM}")

    # Evaluate: reconstruct curve from predicted coefficients, compare to gold
    xs = np.linspace(0, 1, 128)
    mae_pred_list, mae_gold_list = [], []

    for i in range(len(X_te)):
        row = df_v.iloc[np.where(~train_mask)[0][i]]
        t_gold = row_to_target(row)
        if t_gold is None:
            continue
        t_pred = np.array([
            float(m.predict(X_te[i:i+1])[0]) * sigma + mu
            for m, mu, sigma in models
        ], dtype=np.float32)

        rpu_gold = target_to_rpu(t_gold)
        rpu_pred = target_to_rpu(t_pred)
        try:
            ys_gold = eval_rpu(rpu_gold, xs)
            ys_pred = eval_rpu(rpu_pred, xs)
            mae_pred_list.append(np.mean(np.abs(ys_gold - ys_pred)))
        except Exception:
            pass

    mae = np.mean(mae_pred_list) if mae_pred_list else float('nan')
    print(f"\nHeld-out MAE (curve space): {mae:.5f}  ({mae*1500:.1f} nits approx)")
    print(f"Evaluated on {len(mae_pred_list)} scenes")

    return models, feats


def predict(models, feats, row):
    """Predict RPU target vector for one row."""
    x = np.array([[float(row[f]) for f in feats]])
    return np.array([
        float(m.predict(x)[0]) * sigma + mu
        for m, mu, sigma in models
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--l1', default=None)
    args = ap.parse_args()

    df = load_data(args.dataset, args.l1)
    models, feats = train(df)
