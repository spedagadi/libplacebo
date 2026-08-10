"""
dv_curve_model.py
=================
Foundation code for the "unplugged" ML dynamic tone-mapping experiment.

What this does (and only this — it's a starting point for your copilot to extend):
  1. Loads a DV-extracted dataset (one row per sampled frame; RPU-derived features + labels).
  2. Reconstructs the per-scene tone-mapping CURVE from the DV RPU piecewise polynomial
     (this is the corrected reconstruction: 10-bit input domain, respects per-segment order,
      handles single-segment / order-1 cases). Verified: produces 100% monotonic curves.
  3. De-duplicates frames -> scenes (via scene_refresh) so we model per-scene, not per-frame.
  4. Runs a scene-grouped baseline regressor (features -> curve) and reports improvement over
     the mean-curve baseline + R2. This is the litmus test.

KNOWN-GOOD RESULT (single title, ~2062 frames / ~576 scenes):
    buggy reconstruction : ~15% over baseline, R2 ~0.08   (wrong domain/order  -> noisy label)
    fixed reconstruction : ~21% over baseline, R2 ~0.35   (this file's eval_curve)

IMPORTANT CAVEATS (read before trusting results):
  * LABEL SEMANTICS NOT EXTERNALLY VALIDATED. The 10-bit domain + per-segment-order eval
    yields monotonic, plausible curves, but has NOT been cross-checked against libplacebo's
    or dovi_tool's actual applied DV curve. Do that before scaling to many titles.
    (See `TODO: validate_against_reference` below.)
  * COEFFICIENT SCALING is inferred (input normalized as x/1023). If your extractor used a
    different convention, adjust `eval_curve`.
  * SINGLE TITLE ONLY. R2=0.35 is *within-title* signal. Generalization needs many titles.
  * NO SPATIAL FEATURES yet. Current features are global (RPU-derived). The likely lever to
    push R2 above 0.35 is spatial features (integral-image highlight concentration), which
    require DECODED FRAMES (not in this dataset).

Extension points for your copilot are marked with `# COPILOT:`.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------------------------------------------
# 1. CURVE RECONSTRUCTION  (the corrected version — the key fix that lifted R2 0.08->0.35)
# --------------------------------------------------------------------------------------
def eval_curve(row, n_samples=16, input_max=1023.0):
    """
    Evaluate the DV RPU piecewise polynomial for one row into a sampled tone curve.

    The RPU stores, per scene:
        poly_pivots : space-separated segment boundaries in a 10-bit code domain (0..1023)
        poly_num_segs : number of segments
        seg{i}_order  : polynomial order of segment i (1=linear, 2=quadratic)
        seg{i}_c0/c1/c2 : coefficients for segment i

    We evaluate at `n_samples` input points spanning [pivots[0], pivots[-1]], normalizing
    the input as x/input_max before applying coefficients.

    Returns a length-`n_samples` np.ndarray of output values, or None if the row is unusable.

    # COPILOT: if you validate against libplacebo/dovi_tool and find the scaling convention
    #          differs (e.g. input not normalized, or coefficients fixed-point scaled),
    #          adjust the `xn = x / input_max` line and/or the coefficient application here.
    """
    try:
        pivots = [float(p) for p in str(row['poly_pivots']).split()]
    except Exception:
        return None
    n_seg = int(row['poly_num_segs'])
    if len(pivots) < 2:
        return None
    lo, hi = pivots[0], pivots[-1]
    if hi <= lo:
        return None

    # Coefficients are raw fixed-point integers; scale = 1/2^coef_log2_denom = 1/2^23.
    # (libavutil/dovi_meta.h: int64_t poly_coef[8][3]; libplacebo scales by 1/(1<<denom))
    coef_scale = 1.0 / (2 ** 23)

    xs = np.linspace(lo, hi, n_samples)
    ys = []
    for x in xs:
        # locate the segment this x falls into
        s = 0
        for i in range(min(n_seg, len(pivots) - 1)):
            if x >= pivots[i]:
                s = i
        order = row.get(f'seg{s}_order', 2)
        c0 = row.get(f'seg{s}_c0')
        c1 = row.get(f'seg{s}_c1')
        c2 = row.get(f'seg{s}_c2')
        if pd.isna(c0) or pd.isna(c1):
            return None
        xn = x / input_max                       # normalize 10-bit input to [0, 1]
        if order == 1 or pd.isna(c2):
            y = (c0 + c1 * xn) * coef_scale      # linear segment
        else:
            y = (c0 + c1 * xn + c2 * xn * xn) * coef_scale  # quadratic segment
        ys.append(y)
    return np.array(ys)


def reconstruct_all_curves(df, n_samples=16):
    """
    Reconstruct curves for every row. Returns:
        idx    : list of row indices with a valid curve
        curves : (len(idx), n_samples) array of sampled curves
    Also prints a monotonicity check (a tone curve must be non-decreasing; ~100% expected).
    """
    curves = {}
    for i, r in df.iterrows():
        c = eval_curve(r, n_samples=n_samples)
        if c is not None and not np.any(np.isnan(c)) and not np.any(np.isinf(c)):
            curves[i] = c
    idx = list(curves.keys())
    Y = np.vstack([curves[i] for i in idx]) if idx else np.empty((0, n_samples))

    if len(Y):
        mono = np.mean([(np.diff(Y[i]) >= -1e-6).all() for i in range(len(Y))])
        print(f"[reconstruct] valid curves: {len(idx)}/{len(df)} | monotonic: {100*mono:.0f}%")
    return idx, Y


# --------------------------------------------------------------------------------------
# 2. DATA PREP
# --------------------------------------------------------------------------------------
# Global, RPU-derived features available in the dataset. These are the "cheap" features.
# NOTE on inference-time availability: at live capture time the RPU is STRIPPED, so features
# must ultimately be computed FROM PIXELS the same way at train & inference (parity). The
# RPU-derived maxscl/distrib here are fine for this offline litmus test, and double as a
# CROSS-CHECK target for a future pixel-based extractor (pixel peak should ~match RPU maxscl).
FEATURE_COLS = ['maxscl', 'average_maxrgb', 'fraction_bright_pixels'] + \
               [f'distrib_val_{i}' for i in range(3, 9)]   # low percentiles are ~constant; skip 0-2


def prepare(df):
    """
    Clean, assign scene ids, reconstruct curves, and return everything the model needs.
    Returns dict with X (features), Y (curves), groups (scene ids), df_valid.
    """
    df = df.dropna(subset=['maxscl', 'seg0_c0']).reset_index(drop=True)
    # scene id = cumulative count of scene-cut flags -> frames in the same shot share an id
    df['scene_id'] = df['scene_refresh'].cumsum()

    idx, Y = reconstruct_all_curves(df)
    df_valid = df.iloc[idx].reset_index(drop=True)

    feats = [c for c in FEATURE_COLS if c in df_valid.columns and df_valid[c].std() > 0]
    X = df_valid[feats].values
    groups = df_valid['scene_id'].values

    print(f"[prepare] rows={len(df_valid)} scenes={len(np.unique(groups))} features={feats}")
    return dict(X=X, Y=Y, groups=groups, feats=feats, df=df_valid)


# --------------------------------------------------------------------------------------
# 3. BASELINE MODEL + LITMUS TEST
# --------------------------------------------------------------------------------------
def litmus(data, sample_pts=(0, 2, 4, 6, 8, 10, 12, 14), n_splits=5):
    """
    Scene-grouped K-fold. Predicts the curve (at `sample_pts`) from features, standardized.
    Compares against the mean-curve baseline. Reports improvement% and R2.

    Grouping by scene_id ensures NO scene appears in both train and test (no leakage) —
    critical, because frames in a shot are near-duplicates.

    # COPILOT: swap GradientBoostingRegressor for XGBoost/LightGBM, or a torch MLP head,
    #          here. Keep the GroupKFold(groups=scene_id) — do not use a random frame split.
    """
    X, Y, groups = data['X'], data['Y'], data['groups']
    Yt = Y[:, list(sample_pts)]
    Ytz = (Yt - Yt.mean(0)) / (Yt.std(0) + 1e-9)   # standardize per curve-point for fair MAE

    gkf = GroupKFold(n_splits=n_splits)
    mdl_err, base_err, r2s = [], [], []
    for tr, te in gkf.split(X, groups=groups):
        preds = []
        for k in range(Ytz.shape[1]):
            m = GradientBoostingRegressor(n_estimators=150, max_depth=3,
                                          learning_rate=0.05, random_state=0)
            m.fit(X[tr], Ytz[tr, k])
            preds.append(m.predict(X[te]))
        P = np.vstack(preds).T
        mdl_err.append(mean_absolute_error(Ytz[te], P))
        base_err.append(mean_absolute_error(Ytz[te], np.tile(Ytz[tr].mean(0), (len(te), 1))))
        r2s.append(r2_score(Ytz[te], P))

    g, b, r2 = np.mean(mdl_err), np.mean(base_err), np.mean(r2s)
    print(f"\n[litmus] model MAE={g:.4f} | mean-baseline MAE={b:.4f} "
          f"| improvement={100*(1-g/b):.1f}% | R2={r2:.3f}")
    return dict(model_mae=g, baseline_mae=b, improvement=1 - g / b, r2=r2)


def feature_importance(data, curve_point=8, n_splits=5):
    """
    Fit on a single representative curve point and report feature importances +
    per-fold R2. Sensible importances (peak/maxscl high) = signal is real, not artifact.
    """
    X, Y, groups, feats = data['X'], data['Y'], data['groups'], data['feats']
    t = Y[:, curve_point]
    tz = (t - t.mean()) / (t.std() + 1e-9)
    gkf = GroupKFold(n_splits=n_splits)
    r2s, imps = [], []
    for tr, te in gkf.split(X, groups=groups):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, random_state=0)
        m.fit(X[tr], tz[tr])
        r2s.append(r2_score(tz[te], m.predict(X[te])))
        imps.append(m.feature_importances_)
    print(f"\n[importance] point={curve_point} R2={np.mean(r2s):.3f}")
    for f, im in sorted(zip(feats, np.mean(imps, 0)), key=lambda z: -z[1]):
        print(f"    {f:24s} {im:.3f}")
    return dict(r2=np.mean(r2s), importances=dict(zip(feats, np.mean(imps, 0))))


# --------------------------------------------------------------------------------------
# 4. TODO STUBS for your copilot  (not implemented — deliberate handoff points)
# --------------------------------------------------------------------------------------
def validate_against_reference(df, frame_indices):
    """
    # COPILOT / TODO — THE MOST IMPORTANT NEXT STEP.
    Cross-check our reconstructed curve (eval_curve) against a TRUSTED reference for the
    SAME frames: either libplacebo's applied DV curve, or dovi_tool's interpretation.
    Overlay/compare. If they match -> label validated (R2 is on a clean target).
    If not -> eval_curve's domain/scaling convention is still wrong; fix it here.
    This gates trusting any R2 number and gates scaling to many titles.
    """
    raise NotImplementedError("Cross-check reconstructed curve vs libplacebo/dovi_tool output.")


def visualize_curves(df, row_indices, n_samples=64):
    """
    # COPILOT / TODO — plotting.
    For each row, call eval_curve(row, n_samples) and plot input(0..1023 normalized) vs output.
    Overlay several scenes to see how much the curve actually varies across the movie.
    (Earlier finding: within one title the curves may vary only modestly — worth visualizing.)
    """
    raise NotImplementedError("Plot eval_curve outputs; overlay multiple scenes.")


def libplacebo_computepeak_baseline(df):
    """
    # COPILOT / TODO — THE REAL 'is-it-better-than-libplacebo' TEST (Interpretation B).
    For each scene, get libplacebo's METADATA-FREE curve (its --hdr-compute-peak choice),
    then measure whose curve is closer to the DV gold-standard curve: libplacebo's heuristic
    vs. our ML prediction. This quantifies the project's value proposition.
    """
    raise NotImplementedError("Compare ML curve vs libplacebo compute-peak curve vs DV gold.")


# --------------------------------------------------------------------------------------
# 5. ENTRY POINT
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dv_dataset_full.csv"
    df = pd.read_csv(path)
    print(f"[load] {path}: shape={df.shape}")
    data = prepare(df)
    litmus(data)
    feature_importance(data)