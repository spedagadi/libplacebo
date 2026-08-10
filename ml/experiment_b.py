"""
experiment_b.py — Experiment B: ML vs libplacebo spline vs DV gold
===================================================================
For every scene in the dataset:
  1. DV gold curve       — reconstructed from RPU polynomial coefficients
  2. libplacebo spline   — called directly via libplacebo-360.dll ctypes
                           (pl_tone_map_generate with pl_tone_map_spline)
  3. ML predicted curve  — GBR model trained on L1 + pixel features

Metrics reported per-scene and in aggregate:
  - MAE and max-error vs DV gold (in normalised PQ, 0-1 range)
  - Which method is closer to DV gold, scene by scene
  - Histogram of errors

Run:
    python ml/experiment_b.py --dataset <dv_dataset_full.csv> --l1 <l1_data.csv>
"""

import argparse
import ctypes
import numpy as np
import pandas as pd
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
N_PTS      = 256
COEF_SCALE = 1.0 / (2**23)
INPUT_MAX  = 1023.0

LIBPLACEBO_DLL = "C:/msys64/ucrt64/bin/libplacebo-360.dll"

# PQ constants (src/colorspace.h)
PQ_M1 = 2610.0/4096/4;  PQ_M2 = 2523.0/4096*128
PQ_C1 = 3424.0/4096;    PQ_C2 = 2413.0/4096*32;  PQ_C3 = 2392.0/4096*32
SDR_WHITE = 203.0

def nits_to_pq(nits):
    x = (nits / 10000.0) ** PQ_M1
    return ((PQ_C1 + PQ_C2 * x) / (1.0 + PQ_C3 * x)) ** PQ_M2

OUTPUT_MAX_PQ = nits_to_pq(SDR_WHITE)   # ~0.5807
OUTPUT_MIN_PQ = nits_to_pq(0.005)       # near-black

PL_HDR_PQ = 3   # enum pl_hdr_scaling


# ---------------------------------------------------------------------------
# libplacebo ctypes structs  (exact layout from v7.360.1 headers)
# ---------------------------------------------------------------------------
class PlCieXY(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

class PlRawPrimaries(ctypes.Structure):
    _fields_ = [("red", PlCieXY), ("green", PlCieXY),
                ("blue", PlCieXY), ("white", PlCieXY)]

class PlHdrBezier(ctypes.Structure):
    _fields_ = [("target_luma", ctypes.c_float), ("knee_x", ctypes.c_float),
                ("knee_y", ctypes.c_float), ("anchors", ctypes.c_float * 15),
                ("num_anchors", ctypes.c_uint8)]

class PlHdrMetadata(ctypes.Structure):
    _fields_ = [("prim", PlRawPrimaries), ("min_luma", ctypes.c_float),
                ("max_luma", ctypes.c_float), ("max_cll", ctypes.c_float),
                ("max_fall", ctypes.c_float), ("scene_max", ctypes.c_float * 3),
                ("scene_avg", ctypes.c_float), ("ootf", PlHdrBezier),
                ("max_pq_y", ctypes.c_float), ("avg_pq_y", ctypes.c_float)]

class PlToneMapConstants(ctypes.Structure):
    _fields_ = [("knee_adaptation", ctypes.c_float), ("knee_minimum", ctypes.c_float),
                ("knee_maximum", ctypes.c_float), ("knee_default", ctypes.c_float),
                ("knee_offset", ctypes.c_float), ("slope_tuning", ctypes.c_float),
                ("slope_offset", ctypes.c_float), ("spline_contrast", ctypes.c_float),
                ("reinhard_contrast", ctypes.c_float), ("linear_knee", ctypes.c_float),
                ("exposure", ctypes.c_float)]

class PlToneMapParams(ctypes.Structure):
    _fields_ = [("function", ctypes.c_void_p), ("constants", PlToneMapConstants),
                ("input_scaling", ctypes.c_int), ("output_scaling", ctypes.c_int),
                ("lut_size", ctypes.c_size_t), ("input_min", ctypes.c_float),
                ("input_max", ctypes.c_float), ("input_avg", ctypes.c_float),
                ("output_min", ctypes.c_float), ("output_max", ctypes.c_float),
                ("hdr", PlHdrMetadata), ("param", ctypes.c_float)]


def _load_libplacebo():
    lib = ctypes.CDLL(LIBPLACEBO_DLL)
    lib.pl_tone_map_generate.restype  = None
    lib.pl_tone_map_generate.argtypes = [ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(PlToneMapParams)]
    spline_addr = ctypes.addressof(ctypes.c_void_p.in_dll(lib, "pl_tone_map_spline"))
    return lib, spline_addr

_LP_LIB, _LP_SPLINE = _load_libplacebo()


def libplacebo_spline(input_max_pq, input_avg_pq, xs,
                      output_max_pq=OUTPUT_MAX_PQ, output_min_pq=OUTPUT_MIN_PQ):
    """
    Call pl_tone_map_generate(pl_tone_map_spline) from libplacebo-360.dll.
    xs is used only for its length (lut_size) and min/max bounds.
    Returns ys evaluated at the same xs grid via linear interpolation of the LUT.
    """
    lut_size = len(xs)
    params = PlToneMapParams()
    ctypes.memset(ctypes.addressof(params), 0, ctypes.sizeof(params))
    params.function       = _LP_SPLINE
    params.input_scaling  = PL_HDR_PQ
    params.output_scaling = PL_HDR_PQ
    params.lut_size       = lut_size
    params.input_min      = float(xs[0])
    params.input_max      = float(xs[-1])
    params.input_avg      = input_avg_pq
    params.output_min     = output_min_pq
    params.output_max     = output_max_pq
    # PL_TONE_MAP_CONSTANTS defaults
    params.constants.knee_adaptation   = 0.4
    params.constants.knee_minimum      = 0.1
    params.constants.knee_maximum      = 0.8
    params.constants.knee_default      = 0.4
    params.constants.knee_offset       = 1.0
    params.constants.slope_tuning      = 1.5
    params.constants.slope_offset      = 0.2
    params.constants.spline_contrast   = 0.5
    params.constants.reinhard_contrast = 0.5
    params.constants.linear_knee       = 0.3
    params.constants.exposure          = 1.0

    lut = (ctypes.c_float * lut_size)()
    _LP_LIB.pl_tone_map_generate(lut, ctypes.byref(params))
    return np.array(list(lut))


# ---------------------------------------------------------------------------
# 1. DV gold curve
# ---------------------------------------------------------------------------
def eval_dv_curve(row, xs):
    """Evaluate DV RPU piecewise polynomial at given xs (PQ 0-1 normalised)."""
    try:
        pivots = [float(p) / INPUT_MAX for p in str(row["poly_pivots"]).split()]
    except Exception:
        return None
    n_seg = int(row["poly_num_segs"])
    if len(pivots) < 2 or pivots[-1] <= pivots[0]:
        return None

    ys = []
    for x in xs:
        s = 0
        for i in range(min(n_seg, len(pivots) - 1)):
            if x >= pivots[i]:
                s = i
        order = row.get(f"seg{s}_order", 2)
        c0 = row.get(f"seg{s}_c0")
        c1 = row.get(f"seg{s}_c1")
        c2 = row.get(f"seg{s}_c2")
        if pd.isna(c0) or pd.isna(c1):
            return None
        if order == 1 or pd.isna(c2):
            y = (c0 + c1 * x) * COEF_SCALE
        else:
            y = (c0 + c1 * x + c2 * x * x) * COEF_SCALE
        ys.append(y)
    return np.array(ys)


# ---------------------------------------------------------------------------
# 3. ML model  (scene-grouped GBR, same as dv_curve_model but returns per-scene)
# ---------------------------------------------------------------------------
FEATURE_COLS = (
    ["maxscl", "average_maxrgb", "fraction_bright_pixels"] +
    [f"distrib_val_{i}" for i in range(3, 9)] +
    ["l1_min_pq", "l1_max_pq", "l1_avg_pq"]
)

SAMPLE_PTS = list(range(0, N_PTS, N_PTS // 16))   # 16 training points


def train_ml_model(df_train, sample_pts):
    """Train one GBR per curve sample point. Returns list of fitted models."""
    feats = [c for c in FEATURE_COLS if c in df_train.columns and df_train[c].std() > 0]
    X = df_train[feats].values
    models = []
    for k in sample_pts:
        ys = df_train[f"_curve_{k}"].values
        yz = (ys - ys.mean()) / (ys.std() + 1e-9)
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, random_state=0)
        m.fit(X, yz)
        models.append((m, ys.mean(), ys.std()))
    return models, feats


def predict_ml_curve(models, feats, row_feats, xs, sample_pts):
    """Predict curve at all xs by interpolating between trained sample points."""
    X = np.array([[row_feats[f] for f in feats]])
    preds = []
    for m, mu, sigma in models:
        preds.append(float(m.predict(X)[0]) * sigma + mu)
    # interpolate to full grid
    return np.interp(xs, [xs[k] for k in sample_pts], preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_and_merge(dataset_path, l1_path):
    df = pd.read_csv(dataset_path)
    l1 = pd.read_csv(l1_path)
    df_s = df.sort_values("pts_time").reset_index(drop=True)
    l1_s = l1.sort_values("pts_approx").rename(columns={"pts_approx": "pts_time"})
    merged = pd.merge_asof(df_s, l1_s[["pts_time","l1_min_pq","l1_max_pq","l1_avg_pq"]],
                           on="pts_time", direction="nearest", tolerance=0.5)
    for col in ["l1_min_pq","l1_max_pq","l1_avg_pq"]:
        merged[col] = merged[col] / 4095.0
    merged["scene_id"] = merged["scene_refresh"].cumsum()
    merged = merged.dropna(subset=["maxscl","seg0_c0","l1_max_pq"]).reset_index(drop=True)
    return merged


def build_scene_table(df, xs):
    """
    For each scene (first frame), reconstruct the DV gold curve and store
    pre-computed curve values at sample pts as columns _curve_{k}.
    Returns scene-level DataFrame.
    """
    scenes = []
    sample_pts_idx = SAMPLE_PTS
    for scene_id, grp in df.groupby("scene_id"):
        row = grp.iloc[0]
        curve = eval_dv_curve(row, xs)
        if curve is None or np.any(np.isnan(curve)):
            continue
        entry = row.to_dict()
        for k in sample_pts_idx:
            entry[f"_curve_{k}"] = curve[k]
        entry["_xs"] = xs
        entry["_dv_curve"] = curve
        scenes.append(entry)
    return pd.DataFrame(scenes)


def run(dataset_path, l1_path, output_dir="."):
    print("Loading data...")
    df = load_and_merge(dataset_path, l1_path)
    print(f"  {len(df)} frames, {df['scene_id'].nunique()} scenes")

    xs = np.linspace(0, 1, N_PTS)

    print("Building scene table + DV gold curves...")
    scenes = build_scene_table(df, xs)
    print(f"  {len(scenes)} valid scenes")

    # --- Compute libplacebo spline curves ---
    print("Computing libplacebo spline curves...")
    spline_curves = []
    for _, row in scenes.iterrows():
        imax = float(row["l1_max_pq"])
        iavg = float(row["l1_avg_pq"])
        if imax <= 0:
            spline_curves.append(None)
            continue
        c = libplacebo_spline(imax, iavg, xs)
        spline_curves.append(c)
    scenes["_spline_curve"] = spline_curves

    # --- Train ML model (leave-one-scene-out not practical for per-scene;
    #     use 5-fold GroupKFold so test scenes are truly held out) ---
    print("Training ML model (GroupKFold)...")
    feats = [c for c in FEATURE_COLS if c in scenes.columns and scenes[c].std() > 0]
    groups = scenes["scene_id"].values
    ml_preds = np.full((len(scenes), N_PTS), np.nan)

    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(scenes, groups=groups)):
        df_tr = scenes.iloc[tr]
        models, used_feats = train_ml_model(df_tr, SAMPLE_PTS)
        for i in te:
            row = scenes.iloc[i]
            row_feats = {f: float(row[f]) for f in used_feats}
            ml_preds[i] = predict_ml_curve(models, used_feats, row_feats, xs, SAMPLE_PTS)
        print(f"  fold {fold+1}/5 done")

    # --- Compute per-scene errors ---
    print("\nComputing errors...")
    results = []
    dv_curves = np.vstack(scenes["_dv_curve"].values)

    for i, (_, row) in enumerate(scenes.iterrows()):
        dv  = dv_curves[i]
        spl = spline_curves[i]
        ml  = ml_preds[i]

        if spl is None or np.any(np.isnan(ml)):
            continue

        spl_mae  = float(np.mean(np.abs(dv - spl)))
        spl_max  = float(np.max(np.abs(dv - spl)))
        ml_mae   = float(np.mean(np.abs(dv - ml)))
        ml_max   = float(np.max(np.abs(dv - ml)))

        results.append({
            "scene_id":     row["scene_id"],
            "pts_time":     row["pts_time"],
            "l1_max_pq":    row["l1_max_pq"],
            "l1_avg_pq":    row["l1_avg_pq"],
            "spline_mae":   spl_mae,
            "spline_max":   spl_max,
            "ml_mae":       ml_mae,
            "ml_max":       ml_max,
            "ml_wins":      int(ml_mae < spl_mae),
            "ml_improvement": (spl_mae - ml_mae) / max(spl_mae, 1e-9),
        })

    res = pd.DataFrame(results)
    res.to_csv(f"{output_dir}/experiment_b_results.csv", index=False)

    # --- Summary ---
    print("\n" + "="*60)
    print("EXPERIMENT B RESULTS")
    print("="*60)
    print(f"Scenes evaluated: {len(res)}")
    print()
    print(f"{'Metric':<30} {'libplacebo spline':>18} {'ML prediction':>14}")
    print("-"*64)
    print(f"{'Mean MAE (PQ 0-1)':<30} {res['spline_mae'].mean():>18.5f} {res['ml_mae'].mean():>14.5f}")
    print(f"{'Median MAE':<30} {res['spline_mae'].median():>18.5f} {res['ml_mae'].median():>14.5f}")
    print(f"{'Mean max-error':<30} {res['spline_max'].mean():>18.5f} {res['ml_max'].mean():>14.5f}")
    print(f"{'% scenes ML wins':<30} {100*res['ml_wins'].mean():>17.1f}%")
    print(f"{'Mean ML improvement':<30} {100*res['ml_improvement'].mean():>17.1f}%")
    print()

    # Express MAE in nits at typical operating point (PQ 0.5 ~ 100 nits)
    # d(nits)/d(PQ) at PQ=0.5 ≈ 10000 * m1*m2 * ... ≈ ~1500 nits/PQ_unit
    # Use a simpler approximation: 0.01 PQ ≈ 15 nits at midtones
    PQ_TO_NITS_APPROX = 1500.0
    print(f"Approximate nit-space MAE:")
    print(f"  libplacebo spline: {res['spline_mae'].mean()*PQ_TO_NITS_APPROX:.1f} nits")
    print(f"  ML prediction:     {res['ml_mae'].mean()*PQ_TO_NITS_APPROX:.1f} nits")
    print()

    # Error buckets
    for thresh, label in [(0.005,"<0.5% PQ"), (0.01,"<1% PQ"), (0.05,"<5% PQ"), (0.1,"<10% PQ")]:
        spl_pct = 100*(res["spline_mae"] < thresh).mean()
        ml_pct  = 100*(res["ml_mae"]     < thresh).mean()
        print(f"  Scenes within {label}:  spline={spl_pct:.0f}%  ML={ml_pct:.0f}%")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment B: ML vs libplacebo Spline vs DV Gold", fontsize=13)

    # 1. MAE distribution
    ax = axes[0, 0]
    bins = np.linspace(0, max(res["spline_mae"].quantile(0.99),
                              res["ml_mae"].quantile(0.99)) * 1.1, 40)
    ax.hist(res["spline_mae"], bins=bins, alpha=0.6, label="libplacebo spline", color="#e07040")
    ax.hist(res["ml_mae"],     bins=bins, alpha=0.6, label="ML prediction",     color="#4080e0")
    ax.axvline(res["spline_mae"].mean(), color="#e07040", linestyle="--", linewidth=1.5)
    ax.axvline(res["ml_mae"].mean(),     color="#4080e0", linestyle="--", linewidth=1.5)
    ax.set_xlabel("MAE vs DV gold (PQ 0-1)")
    ax.set_ylabel("Scene count")
    ax.set_title("Error distribution")
    ax.legend()

    # 2. Max-error distribution
    ax = axes[0, 1]
    bins2 = np.linspace(0, max(res["spline_max"].quantile(0.99),
                               res["ml_max"].quantile(0.99)) * 1.1, 40)
    ax.hist(res["spline_max"], bins=bins2, alpha=0.6, label="libplacebo spline", color="#e07040")
    ax.hist(res["ml_max"],     bins=bins2, alpha=0.6, label="ML prediction",     color="#4080e0")
    ax.set_xlabel("Max error vs DV gold (PQ 0-1)")
    ax.set_ylabel("Scene count")
    ax.set_title("Worst-case error distribution")
    ax.legend()

    # 3. MAE vs l1_max_pq scatter
    ax = axes[1, 0]
    ax.scatter(res["l1_max_pq"], res["spline_mae"], alpha=0.3, s=8,
               label="spline", color="#e07040")
    ax.scatter(res["l1_max_pq"], res["ml_mae"],     alpha=0.3, s=8,
               label="ML",      color="#4080e0")
    ax.set_xlabel("l1_max_pq (normalised)")
    ax.set_ylabel("MAE vs DV gold")
    ax.set_title("Error vs scene peak brightness")
    ax.legend()

    # 4. Single median scene: 3 curves only — DV gold, spline, ML
    ax = axes[1, 1]
    # Pick median scene by spline MAE — ensures interesting content, not a dark/trivial scene
    res_reset = res.reset_index(drop=True)
    q  = res_reset["spline_mae"].quantile(0.5)
    si = (res_reset["spline_mae"] - q).abs().idxmin()

    scene_id  = res_reset.loc[si, "scene_id"]
    scene_row = scenes[scenes["scene_id"] == scene_id].iloc[0]
    scene_pos = scenes.index.get_loc(scenes[scenes["scene_id"] == scene_id].index[0])

    dv_curve  = dv_curves[scene_pos]
    spl_curve = spline_curves[scene_pos]
    ml_curve  = ml_preds[scene_pos]

    try:
        pivots = [float(p) / INPUT_MAX for p in str(scene_row["poly_pivots"]).split()]
        x_lo, x_hi = pivots[0], pivots[-1]
    except Exception:
        x_lo, x_hi = 0.0, 1.0
    xs_scene   = np.linspace(x_lo, x_hi, N_PTS)
    spl_interp = np.interp(xs_scene, xs, spl_curve)
    ml_interp  = np.interp(xs_scene, xs, ml_curve)

    t = float(res_reset.loc[si, "pts_time"])
    ax.plot(xs_scene, dv_curve,   color="#f5c518", linewidth=2.5, linestyle="-",  label=f"DV gold (reference)")
    ax.plot(xs_scene, spl_interp, color="#e07040", linewidth=1.8, linestyle="--", label=f"libplacebo spline  MAE={res_reset.loc[si,'spline_mae']:.4f}")
    ax.plot(xs_scene, ml_interp,  color="#4488ff", linewidth=1.8, linestyle=":",  label=f"ML prediction      MAE={res_reset.loc[si,'ml_mae']:.4f}")

    ax.set_xlabel("Input PQ (normalised 0-1)")
    ax.set_ylabel("Output PQ (normalised 0-1)")
    ax.set_title(f"Median scene  (t={t:.0f}s)\nDV gold = target  |  closer = better")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{output_dir}/experiment_b.png"
    plt.savefig(out_path, dpi=140)
    print(f"\nPlot saved: {out_path}")
    print(f"Results CSV: {output_dir}/experiment_b_results.csv")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--l1",      required=True)
    ap.add_argument("--out",     default=".")
    args = ap.parse_args()
    run(args.dataset, args.l1, args.out)
