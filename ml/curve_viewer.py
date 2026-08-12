"""
curve_viewer.py — DV tone-mapping curve explorer
=================================================
Shows 3 curves per scene:
  - DV gold    : reconstructed from RPU polynomial (the reference)
  - libplacebo : pl_tone_map_spline called via libplacebo-360.dll
  - ML pred    : GBR model trained on pixel + L1 features

Run:
    streamlit run ml/curve_viewer.py -- --csv path/to/dv_dataset_full.csv --l1 path/to/l1_data.csv
"""

import sys
import ctypes
import subprocess
import tempfile
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COEF_SCALE  = 1.0 / (2 ** 23)
INPUT_MAX   = 1023.0
N_PTS       = 256

LIBPLACEBO_DLL = "C:/msys64/ucrt64/bin/libplacebo-360.dll"

PQ_M1 = 2610.0/4096/4;  PQ_M2 = 2523.0/4096*128
PQ_C1 = 3424.0/4096;    PQ_C2 = 2413.0/4096*32;  PQ_C3 = 2392.0/4096*32
SDR_WHITE = 203.0

def nits_to_pq(n):
    x = (n / 10000.0) ** PQ_M1
    return ((PQ_C1 + PQ_C2 * x) / (1.0 + PQ_C3 * x)) ** PQ_M2

OUTPUT_MAX_PQ = nits_to_pq(SDR_WHITE)
OUTPUT_MIN_PQ = nits_to_pq(0.005)
PL_HDR_PQ = 3

# Feature cols — mirrors dv_coef_model.FEATURE_COLS (full 27 raw SAT zones)
# Training auto-selects base9/derived20/full27 based on scene count
_SAT_ROWS, _SAT_COLS = 3, 3
FEATURE_COLS = (
    ["maxscl", "average_maxrgb", "fraction_bright_pixels"] +
    [f"distrib_val_{i}" for i in range(3, 9)] +
    [f"zone_mean_r{r}_c{c}" for r in range(_SAT_ROWS) for c in range(_SAT_COLS)] +
    [f"zone_max_r{r}_c{c}"  for r in range(_SAT_ROWS) for c in range(_SAT_COLS)]
)
N_SAMPLE_PTS = 16
SAMPLE_IDXS  = list(range(0, N_PTS, N_PTS // N_SAMPLE_PTS))

DISTRIB_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# ---------------------------------------------------------------------------
# libplacebo ctypes structs  (v7.360.1 exact layout)
# ---------------------------------------------------------------------------
class PlCieXY(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]
class PlRawPrimaries(ctypes.Structure):
    _fields_ = [("red", PlCieXY), ("green", PlCieXY), ("blue", PlCieXY), ("white", PlCieXY)]
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

@st.cache_resource
def load_libplacebo():
    lib = ctypes.CDLL(LIBPLACEBO_DLL)
    lib.pl_tone_map_generate.restype  = None
    lib.pl_tone_map_generate.argtypes = [ctypes.POINTER(ctypes.c_float),
                                          ctypes.c_void_p]
    spline_addr = ctypes.addressof(ctypes.c_void_p.in_dll(lib, "pl_tone_map_spline"))
    return lib, spline_addr


_TM_SYMBOLS = {
    "spline":    "pl_tone_map_spline",
    "st2094-10": "pl_tone_map_st2094_10",
    "st2094-40": "pl_tone_map_st2094_40",
    "bt2390":    "pl_tone_map_bt2390",
}

def spline_curve(l1_max_pq, l1_avg_pq, xs,
                 spline_contrast=0.5, knee_adaptation=0.4, slope_tuning=1.5,
                 tone_mapper="spline"):
    """Call a libplacebo tone mapper via DLL. xs defines the input grid."""
    lib, _spline_addr = load_libplacebo()
    sym = _TM_SYMBOLS.get(tone_mapper, "pl_tone_map_spline")
    try:
        tm_addr = ctypes.addressof(ctypes.c_void_p.in_dll(lib, sym))
    except Exception:
        tm_addr = _spline_addr
    lut_size = len(xs)
    params = PlToneMapParams()
    ctypes.memset(ctypes.addressof(params), 0, ctypes.sizeof(params))
    params.function       = tm_addr
    params.input_scaling  = PL_HDR_PQ
    params.output_scaling = PL_HDR_PQ
    params.lut_size       = lut_size
    params.input_min      = float(xs[0])
    params.input_max      = float(xs[-1])
    params.input_avg      = float(l1_avg_pq)
    params.output_min     = OUTPUT_MIN_PQ
    params.output_max     = OUTPUT_MAX_PQ
    params.constants.knee_adaptation   = knee_adaptation
    params.constants.knee_minimum      = 0.1
    params.constants.knee_maximum      = 0.8
    params.constants.knee_default      = 0.4
    params.constants.knee_offset       = 1.0
    params.constants.slope_tuning      = slope_tuning
    params.constants.slope_offset      = 0.2
    params.constants.spline_contrast   = spline_contrast
    params.constants.reinhard_contrast = 0.5
    params.constants.linear_knee       = 0.3
    params.constants.exposure          = 1.0
    lut = (ctypes.c_float * lut_size)()
    lib.pl_tone_map_generate(lut, ctypes.addressof(params))
    return np.array(list(lut))


# ---------------------------------------------------------------------------
# DV gold curve
# ---------------------------------------------------------------------------
def dv_gold_curve(row, xs):
    """Evaluate RPU piecewise polynomial. xs in normalised PQ (0-1). Returns ys or None."""
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
        c0 = row.get(f"seg{s}_c0")
        c1 = row.get(f"seg{s}_c1")
        c2 = row.get(f"seg{s}_c2")
        order = row.get(f"seg{s}_order", 2)
        if pd.isna(c0) or pd.isna(c1):
            return None
        if order == 1 or pd.isna(c2):
            y = (c0 + c1 * x) * COEF_SCALE
        else:
            y = (c0 + c1 * x + c2 * x * x) * COEF_SCALE
        ys.append(y)
    return np.array(ys)


# ---------------------------------------------------------------------------
# ML model  — trained once on full dataset, cached
# ---------------------------------------------------------------------------
@st.cache_resource
def train_model(dataset_csv, l1_csv):
    """
    Train model using dv_coef_model.train() — auto-selects feature set
    (base9 / derived14 / full27) based on training data size.
    Returns (models, feature_cols, train_scene_ids, held_out_scene_ids).
    """
    import dv_coef_model as coef

    df = coef.load_data(dataset_csv, l1_csv)

    # Build targets for 50/50 scene split
    targets, valid_idx = [], []
    for i, row in df.iterrows():
        t = coef.row_to_target(row)
        if t is not None:
            targets.append(t)
            valid_idx.append(i)

    df_v   = df.loc[valid_idx].reset_index(drop=True)
    groups = df_v["scene_id"].values
    all_scenes   = sorted(np.unique(groups))
    split        = len(all_scenes) // 2
    train_scenes = set(all_scenes[:split])
    held_scenes  = set(all_scenes[split:])

    # Force reload to pick up FEATURE_SET changes without Streamlit restart
    import importlib
    importlib.reload(coef)

    # Delegate to coef.train() which handles auto feature selection
    models, feats = coef.train(df)

    return models, feats, train_scenes, held_scenes


def ml_predict(models, feats, row):
    """Predict RPU target vector for one row. Returns flat numpy array or None."""
    import dv_coef_model as coef
    try:
        x = np.array([[float(row[f]) for f in feats]])
    except Exception:
        return None
    return np.array([
        float(m.predict(x)[0]) * sigma + mu
        for m, mu, sigma in models
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset(dataset_csv, l1_csv):
    import dv_coef_model as coef
    df = coef.load_data(dataset_csv, l1_csv if l1_csv else None)
    if len(df) == 0:
        # Stage 1 manifest CSV — pixel features are NaN, load_data drops all rows.
        # Load raw and compute only scene_id so the viewer can show DV gold curves.
        import pandas as pd
        df = pd.read_csv(dataset_csv)
        df = df[df["poly_pivots"].notna() & df["seg0_c0"].notna()].reset_index(drop=True)
        if "scene_refresh" in df.columns:
            df["scene_id"] = df["scene_refresh"].fillna(0).cumsum().astype(int)
        else:
            df["scene_id"] = range(len(df))
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def build_three_curves(row, models, feats, show_pivots,
                       spline_contrast=0.5, knee_adaptation=0.4, slope_tuning=1.5,
                       tone_mapper="spline"):
    xs = np.linspace(0, 1, N_PTS)

    # DV gold — only valid over [pivot_min, pivot_max]
    try:
        pivots_norm = [float(p) / INPUT_MAX for p in str(row["poly_pivots"]).split()]
        x_lo, x_hi = pivots_norm[0], pivots_norm[-1]
    except Exception:
        x_lo, x_hi = 0.0, 1.0
    xs_dv = np.linspace(x_lo, x_hi, N_PTS)
    dv    = dv_gold_curve(row, xs_dv)

    # libplacebo spline — full xs range using l1_max_pq as input ceiling
    l1_max = float(row.get("l1_max_pq") or 0)
    l1_avg = float(row.get("l1_avg_pq") or 0)
    if l1_max > 0:
        xs_spl = np.linspace(0, l1_max, N_PTS)
        spl    = spline_curve(l1_max, l1_avg, xs_spl,
                              spline_contrast=spline_contrast,
                              knee_adaptation=knee_adaptation,
                              slope_tuning=slope_tuning,
                              tone_mapper=tone_mapper)
    else:
        xs_spl, spl = None, None

    # ML prediction — show the raw predicted polynomial for the curve plot
    # (smooth, unmodified) — the sanitised version is only used for rendering
    ml = None
    if models:
        ml_t = ml_predict(models, feats, row)
        if ml_t is not None:
            import dv_coef_model as coef
            rpu_raw = coef.target_to_rpu(ml_t)
            rpu_safe = coef._sanitise_rpu(rpu_raw)
            try:
                # Plot the sanitised (safe) curve — what actually gets rendered
                ml = np.array(coef.eval_rpu(rpu_safe, xs))
            except Exception:
                ml = None

    fig = go.Figure()

    # Identity
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="rgba(150,150,150,0.3)", dash="dot", width=1),
        name="identity", showlegend=True,
        hoverinfo="skip",
    ))

    # libplacebo spline
    if spl is not None:
        fig.add_trace(go.Scatter(
            x=xs_spl.tolist(), y=spl.tolist(), mode="lines",
            line=dict(color="#e07840", width=2, dash="dash"),
            name=f"libplacebo {tone_mapper}",
            hovertemplate="spline<br>in=%{x:.3f}<br>out=%{y:.3f}<extra></extra>",
        ))

    # ML prediction
    if ml is not None:
        fig.add_trace(go.Scatter(
            x=xs.tolist(), y=ml.tolist(), mode="lines",
            line=dict(color="#4488ff", width=2, dash="dot"),
            name="ML prediction",
            hovertemplate="ML<br>in=%{x:.3f}<br>out=%{y:.3f}<extra></extra>",
        ))

    # DV gold — drawn last so it's on top
    if dv is not None:
        fig.add_trace(go.Scatter(
            x=xs_dv.tolist(), y=dv.tolist(), mode="lines",
            line=dict(color="#f5c518", width=3),
            name="DV gold (reference)",
            hovertemplate="DV gold<br>in=%{x:.3f}<br>out=%{y:.3f}<extra></extra>",
        ))
        # Pivot markers
        if show_pivots:
            pv_ys = np.interp(pivots_norm, xs_dv, dv)
            fig.add_trace(go.Scatter(
                x=pivots_norm, y=pv_ys.tolist(), mode="markers",
                marker=dict(color="#ffffff", size=7, symbol="circle",
                            line=dict(color="#f5c518", width=1.5)),
                name="pivots", showlegend=True,
                hovertemplate="pivot<br>in=%{x:.3f}<br>out=%{y:.3f}<extra></extra>",
            ))

    # MAE annotations
    annotations = []
    if dv is not None and spl is not None:
        spl_on_dv = np.interp(xs_dv, xs_spl, spl)
        mae_spl = float(np.mean(np.abs(dv - spl_on_dv)))
        annotations.append(f"spline MAE={mae_spl:.4f}")
    if dv is not None and ml is not None:
        ml_on_dv = np.interp(xs_dv, xs, ml)
        mae_ml = float(np.mean(np.abs(dv - ml_on_dv)))
        annotations.append(f"ML MAE={mae_ml:.4f}")

    fig.update_layout(
        title="Tone Mapping Curves  (normalised PQ in → out)" +
              (f"  |  {' | '.join(annotations)}" if annotations else ""),
        xaxis_title="Input PQ (0–1)",
        yaxis_title="Output PQ (0–1)",
        xaxis=dict(range=[0, 1], gridcolor="rgba(80,80,80,0.3)"),
        yaxis=dict(range=[0, max(OUTPUT_MAX_PQ * 1.05, 0.65)],
                   gridcolor="rgba(80,80,80,0.3)"),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", orientation="v",
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
        height=460,
        margin=dict(l=55, r=20, t=50, b=50),
    )
    return fig


def build_luminance_figure(row):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Luminance CDF", "Scene stats"),
                        column_widths=[0.55, 0.45],
                        specs=[[{"type": "xy"}, {"type": "table"}]])

    pcts = [row.get(f"distrib_pct_{i}") for i in range(9)]
    vals = [row.get(f"distrib_val_{i}") for i in range(9)]
    valid = [(p, v) for p, v in zip(pcts, vals)
             if p is not None and v is not None
             and not (isinstance(p, float) and np.isnan(p))
             and not (isinstance(v, float) and np.isnan(v))]
    if valid:
        ps, vs = zip(*valid)
        fig.add_trace(go.Bar(x=[f"p{int(p)}" for p in ps],
                             y=[float(v) for v in vs],
                             marker_color="rgba(99,200,255,0.7)"), row=1, col=1)

    def fmt(v, decimals=4):
        try:
            return f"{float(v):.{decimals}f}"
        except Exception:
            return "—"

    l1_max = row.get("l1_max_pq")
    l1_avg = row.get("l1_avg_pq")
    labels = ["maxscl", "avg_maxrgb", "frac_bright",
              "l1_max_pq", "l1_avg_pq",
              "src_min_pq", "src_max_pq", "scene_refresh"]
    values = [
        fmt(row.get("maxscl")),
        fmt(row.get("average_maxrgb")),
        fmt(row.get("fraction_bright_pixels")),
        fmt(l1_max),
        fmt(l1_avg),
        str(int(row.get("source_min_pq", 0) or 0)),
        str(int(row.get("source_max_pq", 0) or 0)),
        str(int(row.get("scene_refresh", 0))),
    ]
    fig.add_trace(go.Table(
        header=dict(values=["<b>Field</b>", "<b>Value</b>"],
                    fill_color="#0f3460", font=dict(color="white")),
        cells=dict(values=[labels, values],
                   fill_color=[["#1a1a2e"] * len(labels)],
                   font=dict(color="#e0e0e0")),
    ), row=1, col=2)

    fig.update_layout(plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
                      font=dict(color="#e0e0e0"), height=260,
                      margin=dict(l=50, r=20, t=35, b=35), showlegend=False)
    fig.update_yaxes(title_text="PQ luma", gridcolor="rgba(80,80,80,0.3)", row=1, col=1)
    return fig


def build_timeline(df, frame_idx):
    window = 300
    lo = max(0, frame_idx - window // 2)
    hi = min(len(df), lo + window)
    sub = df.iloc[lo:hi]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["pts_time"].tolist(),
                             y=sub["maxscl"].tolist(), mode="lines",
                             line=dict(color="rgba(99,200,255,0.7)", width=1.2),
                             name="maxscl"))
    cuts = sub[sub["scene_refresh"] == 1]
    if not cuts.empty:
        fig.add_trace(go.Scatter(x=cuts["pts_time"].tolist(),
                                 y=cuts["maxscl"].tolist(), mode="markers",
                                 marker=dict(color="rgba(255,120,50,0.9)", size=6,
                                             symbol="triangle-up"),
                                 name="scene cut"))
    cur = df.iloc[frame_idx]
    fig.add_vline(x=float(cur["pts_time"]),
                  line=dict(color="white", width=1.5, dash="dash"))
    fig.update_layout(title="maxscl timeline  (triangle = scene cut)",
                      xaxis_title="time (s)", yaxis_title="maxscl",
                      plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
                      font=dict(color="#e0e0e0"),
                      legend=dict(bgcolor="rgba(0,0,0,0.3)"),
                      height=190, margin=dict(l=50, r=20, t=35, b=35))
    return fig


# ---------------------------------------------------------------------------
# Frame rendering via dv_render (libplacebo D3D11 pipeline)
# ---------------------------------------------------------------------------

DV_RENDER_EXE = "C:/Code/libplacebo/build/dv_render.exe"
_DV_RENDER_ENV = None

def _get_env():
    global _DV_RENDER_ENV
    if _DV_RENDER_ENV is None:
        import os
        env = os.environ.copy()
        env["PATH"] = "C:/msys64/ucrt64/bin;" + env.get("PATH", "")
        _DV_RENDER_ENV = env
    return _DV_RENDER_ENV


@st.cache_data(max_entries=32)
def render_frame(video_path: str, pts_time: float, mode: str,
                 width: int = 960, height: int = 540,
                 lut_path: str = None,
                 l1_max_pq: float = 0.0,
                 l1_avg_pq: float = 0.0,
                 out_nits: int = 203,
                 spline_contrast: float = 0.0,
                 knee_adaptation: float = 0.0,
                 slope_tuning: float = 0.0,
                 perceptual_strength: float = 0.8,
                 gamut_expansion: bool = False) -> np.ndarray:
    """
    Call dv_render.exe to render one frame through the full libplacebo pipeline.
    Returns RGB uint8 (H, W, 3) numpy array, or None on failure.
    """
    cmd = [
        DV_RENDER_EXE,
        "--input",  video_path,
        "--pts",    f"{pts_time:.6f}",
        "--mode",   mode,
        "--width",  str(width),
        "--height", str(height),
    ]
    if lut_path:
        cmd += ["--lut", lut_path]
    if l1_max_pq > 0:
        cmd += ["--l1-max", f"{l1_max_pq:.6f}", "--l1-avg", f"{l1_avg_pq:.6f}"]
    if out_nits != 203:
        cmd += ["--out-nits", str(out_nits)]
    cmd += ["--spline-contrast",      f"{spline_contrast:.3f}"]
    cmd += ["--knee-adaptation",      f"{knee_adaptation:.3f}"]
    cmd += ["--slope-tuning",         f"{slope_tuning:.3f}"]
    cmd += ["--perceptual-strength",  f"{perceptual_strength:.3f}"]
    cmd += ["--gamut-expansion",      "1" if gamut_expansion else "0"]

    try:
        r = subprocess.run(cmd, capture_output=True, env=_get_env(), timeout=60)
    except Exception as e:
        return None

    expected = width * height * 3
    if len(r.stdout) != expected:
        return None

    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


def _write_rpu_lut(t, path):
    """Write predicted RPU polynomial target vector to RPU_POLY_1D file."""
    import dv_coef_model as coef
    coef.write_rpu_lut(t, path)


def build_frame_panel(video_path, pts_time, row, models, feats,
                      width=960, height=540, out_nits=203,
                      spline_contrast=0.0, knee_adaptation=0.0, slope_tuning=0.0,
                      tone_mapper="spline",
                      perceptual_strength=0.8, gamut_expansion=False):
    """
    Render DV gold / libplacebo spline / ML prediction via dv_render.exe.
    All three go through the same libplacebo D3D11 pipeline — only the tone
    curve differs. Returns (images_dict, diffs_dict) or None.
    """
    images = {}

    # L1 metadata for spline mode
    l1_max = float(row.get("l1_max_pq") or 0)
    l1_avg = float(row.get("l1_avg_pq") or 0)

    # --- DV gold: map_dovi=true — libplacebo applies RPU ycc_to_rgb matrix
    #     + RPU polynomial. This is the reference/gold standard. ---
    sc = dict(spline_contrast=spline_contrast,
              knee_adaptation=knee_adaptation,
              slope_tuning=slope_tuning,
              perceptual_strength=perceptual_strength,
              gamut_expansion=gamut_expansion)

    img_gold = render_frame(video_path, pts_time, "gold", width, height,
                            out_nits=out_nits, **sc)
    if img_gold is not None:
        images["DV gold"] = img_gold

    img_spline = render_frame(video_path, pts_time, tone_mapper, width, height,
                              l1_max_pq=l1_max, l1_avg_pq=l1_avg,
                              out_nits=out_nits, **sc)
    if img_spline is not None:
        images[f"libplacebo {tone_mapper}"] = img_spline

    if models:
        ml_t = ml_predict(models, feats, row)
        if ml_t is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".rpu", delete=False, mode="w")
            tmp.close()
            _write_rpu_lut(ml_t, tmp.name)
            img_ml = render_frame(video_path, pts_time, "ml", width, height,
                                  lut_path=tmp.name, l1_max_pq=l1_max,
                                  l1_avg_pq=l1_avg, out_nits=out_nits, **sc)
            os.unlink(tmp.name)
            if img_ml is not None:
                images["ML prediction"] = img_ml

    if not images:
        return None

    # Difference heatmaps vs DV gold
    diffs = {}
    if "DV gold" in images:
        ref = images["DV gold"].astype(np.int16)
        for name in [f"libplacebo {tone_mapper}", "ML prediction"]:
            if name in images:
                diffs[name] = np.abs(images[name].astype(np.int16) - ref).mean(axis=2)

    return images, diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="DV Curve Viewer", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown("<style>.block-container{padding-top:1rem}</style>",
                unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.title("DV Curve Viewer")

        default_csv   = ""
        default_l1    = ""
        default_video = ""
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "--video" and i + 1 < len(sys.argv):
                default_video = sys.argv[i + 1]; i += 2; continue
            if arg.endswith(".csv") and "l1" in arg.lower():
                default_l1 = arg
            elif arg.endswith(".csv"):
                default_csv = arg
            i += 1

        csv_path = st.text_input("Dataset CSV", value=default_csv,
                                 placeholder="dv_dataset_sat.csv")
        l1_path  = st.text_input("L1 CSV (optional, enables spline + ML)",
                                 value=default_l1,
                                 placeholder="l1_data.csv")

        import os
        if not csv_path or not os.path.exists(csv_path):
            st.info("Enter a valid dataset CSV path to begin.")
            return

        l1_csv = l1_path if (l1_path and os.path.exists(l1_path)) else None
        if l1_path and not l1_csv:
            st.warning("L1 CSV not found — spline and ML curves disabled.")

        df = load_dataset(csv_path, l1_csv)
        n  = len(df)
        scene_count = int(df["scene_refresh"].sum()) if "scene_refresh" in df.columns else (int(df["scene_id"].max()) if "scene_id" in df.columns and df["scene_id"].notna().any() else "?")
        st.caption(f"{n:,} frames  |  {scene_count} scenes  |  {float(df['pts_time'].max()):.0f}s")

        st.divider()

        # Train model first so held_scenes is available for the scene selector
        st.caption("Model status:")
        if l1_csv:
            with st.spinner("Training ML model on first 50% of scenes..."):
                models, feats, train_scenes, held_scenes = train_model(csv_path, l1_csv)
            import dv_coef_model as _coef
            st.caption(f"ML ready  |  {len(feats)} features  ({_coef.FEATURE_SET})  |  "
                       f"train={len(train_scenes)} scenes  held-out={len(held_scenes)} scenes")
        else:
            models, feats = None, []
            train_scenes, held_scenes = set(), set()

        st.divider()

        # nav_frame drives the slider default; updated by scene selectbox on_change
        if "nav_frame" not in st.session_state:
            st.session_state["nav_frame"] = 0

        all_scene_ids = sorted(df["scene_id"].dropna().unique().tolist())
        held_out_only = st.toggle("Held-out scenes only", value=True,
                                   help="Show only the 50% of scenes not used in training")
        if held_out_only and held_scenes:
            scenes = [s for s in all_scene_ids if s in held_scenes]
        else:
            scenes = all_scene_ids

        def _on_scene_change():
            sid = st.session_state["scene_select"]
            st.session_state["nav_frame"] = int(df[df["scene_id"] == sid].index[0])

        st.selectbox(
            "Jump to scene",
            options=scenes,
            key="scene_select",
            on_change=_on_scene_change,
            format_func=lambda s: f"Scene {int(s):04d}  (t={float(df[df.scene_id==s].iloc[0].pts_time):.1f}s)",
        )

        def _on_num_change():
            st.session_state["nav_frame"] = int(st.session_state["frame_num"])

        col_s, col_n = st.columns([3, 1])
        with col_s:
            frame_idx = st.slider("Frame", 0, n - 1,
                                  value=st.session_state["nav_frame"],
                                  key="frame_slider")
        with col_n:
            st.number_input("Index", 0, n - 1,
                            value=st.session_state["nav_frame"],
                            step=1, key="frame_num",
                            on_change=_on_num_change,
                            label_visibility="visible")
        # Merge: number_input change wins (handled by on_change above)
        frame_idx = st.session_state["nav_frame"]
        # Keep slider in sync with nav_frame
        if frame_idx != st.session_state.get("frame_slider", frame_idx):
            st.session_state["nav_frame"] = st.session_state["frame_slider"]
            frame_idx = st.session_state["frame_slider"]

        st.divider()
        show_pivots  = st.toggle("Show pivot points", value=True)
        tone_mapper  = st.selectbox(
            "Tone mapper (spline/reference mode)",
            options=["spline", "st2094-10", "st2094-40", "bt2390"],
            index=0,
            help=("spline=libplacebo default  "
                  "st2094-10=SMPTE rational EETF (DV L1 metadata)  "
                  "st2094-40=SMPTE Bezier (HDR10+ ootf)  "
                  "bt2390=ITU hermite spline"),
        )

        st.divider()
        video_path = st.text_input("Video file (for frame decode)", value=default_video)
        show_frames = st.toggle("Decode & show frames", value=True)
        out_nits    = st.select_slider(
            "Target display (nits)",
            options=[50, 100, 150, 203, 400, 600, 1000],
            value=203,
            help="50=projector  203=SDR monitor  1000=HDR display",
        )
        show_diff   = st.toggle("Show difference heatmap", value=True)

        st.divider()
        st.caption("Spline tone-map constants")
        spline_contrast  = st.slider("spline_contrast",  0.0, 1.5, 0.5, 0.05,
                                     help="0=linear, 0.5=default, 1.5=clip-like shoulder")
        knee_adaptation  = st.slider("knee_adaptation",  0.0, 1.0, 0.4, 0.05,
                                     help="0=fixed knee, 1=fully adapt to scene avg")
        slope_tuning     = st.slider("slope_tuning",     0.0, 4.0, 1.5, 0.1,
                                     help="Slope aggressiveness vs peak ratio")

        st.divider()
        st.caption("Colour volume (gamut) controls")
        perceptual_strength = st.slider("perceptual_strength", 0.0, 1.0, 0.8, 0.05,
                                        help="Chroma restoration after tone map — higher preserves colour saturation at low nits")
        gamut_expansion     = st.toggle("gamut_expansion", value=False,
                                        help="Allow chroma beyond source gamut — helps restore washed-out colours at 50 nits")

    # --- Main area ---
    cur = df.iloc[frame_idx]
    t   = float(cur["pts_time"])
    sid = int(cur["scene_id"])
    st.markdown(f"### Scene {sid:04d}  |  frame {int(cur.frame_idx):05d}  |  t={t:.2f}s")

    # Timeline
    st.plotly_chart(build_timeline(df, frame_idx), width="stretch",
                    key=f"timeline_{frame_idx}")

    # 3-curve plot + luminance panel
    col_curve, col_lum = st.columns([3, 2])
    with col_curve:
        st.plotly_chart(build_three_curves(cur, models, feats, show_pivots,
                                           spline_contrast=spline_contrast,
                                           knee_adaptation=knee_adaptation,
                                           slope_tuning=slope_tuning,
                                           tone_mapper=tone_mapper),
                        width="stretch", key=f"curves_{frame_idx}")
    with col_lum:
        st.plotly_chart(build_luminance_figure(cur), width="stretch",
                        key=f"lum_{frame_idx}")

    # Coefficients expander
    with st.expander("RPU polynomial coefficients", expanded=False):
        try:
            pivots = [float(p) for p in str(cur["poly_pivots"]).split()]
        except Exception:
            pivots = []
        rows = []
        for i in range(int(cur.get("poly_num_segs", 0))):
            c0 = cur.get(f"seg{i}_c0")
            c1 = cur.get(f"seg{i}_c1")
            c2 = cur.get(f"seg{i}_c2")
            p_lo = f"{pivots[i]/INPUT_MAX:.4f}" if i < len(pivots) else "?"
            p_hi = f"{pivots[i+1]/INPUT_MAX:.4f}" if i+1 < len(pivots) else "?"
            rows.append({
                "seg": i, "input range": f"[{p_lo}, {p_hi}]",
                "order": int(cur.get(f"seg{i}_order", 0)),
                "c0 (raw)": int(c0) if c0 and not pd.isna(c0) else None,
                "c1 (raw)": int(c1) if c1 and not pd.isna(c1) else None,
                "c2 (raw)": int(c2) if c2 and not pd.isna(c2) else None,
                "c0f": f"{float(c0)*COEF_SCALE:.6f}" if c0 and not pd.isna(c0) else None,
                "c1f": f"{float(c1)*COEF_SCALE:.6f}" if c1 and not pd.isna(c1) else None,
                "c2f": f"{float(c2)*COEF_SCALE:.6f}" if c2 and not pd.isna(c2) else None,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # --- Frame panel ---
    if show_frames and video_path and __import__("os").path.exists(video_path):
        with st.spinner(f"Decoding frame at t={t:.2f}s  ({out_nits} nits, {tone_mapper})..."):
            result = build_frame_panel(video_path, t, cur, models, feats,
                                       out_nits=out_nits,
                                       spline_contrast=spline_contrast,
                                       knee_adaptation=knee_adaptation,
                                       slope_tuning=slope_tuning,
                                       tone_mapper=tone_mapper,
                                       perceptual_strength=perceptual_strength,
                                       gamut_expansion=gamut_expansion)

        if result is None:
            st.error("Frame decode failed — check video path and ffmpeg.")
        else:
            images, diffs = result
            st.markdown("#### Tone-mapped frames")
            cols = st.columns(len(images))
            for col, (name, img) in zip(cols, images.items()):
                with col:
                    st.caption(name)
                    st.image(img, width="stretch", clamp=True)

            if show_diff and diffs:
                st.markdown("#### Difference vs DV gold  (brighter = larger error)")
                import matplotlib.pyplot as plt
                dcols = st.columns(len(diffs))
                for col, (name, diff) in zip(dcols, diffs.items()):
                    with col:
                        st.caption(f"{name}  (mean={diff.mean():.1f} counts)")
                        fig, ax = plt.subplots(figsize=(6, 3.4))
                        ax.imshow(diff, cmap="inferno", vmin=0, vmax=30)
                        ax.axis("off")
                        st.pyplot(fig, width="stretch")
                        plt.close(fig)
    elif show_frames:
        st.info("Enter a video file path in the sidebar to enable frame decoding.")


if __name__ == "__main__":
    main()
