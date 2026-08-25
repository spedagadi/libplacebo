"""
model_curve_explorer.py — ML vs spline vs gold curve diagnostic viewer.

Shows best/worst performing scenes per cell type.
Select a cell → sorted dropdown of best/worst scenes → see three curves.

Run:
    streamlit run ml/model_curve_explorer.py
"""

import sys, os, json, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(__file__))
import dv_mlp_model as mlp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINTS = {
    "Run 29 (shadow+perceptual hinge)":   r"F:\DTMModelData\ckpt_mcp_run29_best.pt",
    "Run 28 (BO-optimised, MAE 0.09255)": r"F:\DTMModelData\ckpt_mcp_run28_best.pt",
    "Run 27 (maxscl-baseline)":           r"F:\DTMModelData\ckpt_mcp_run27_best.pt",
}
# HDR10 tab adds XGBoost as a separate option not available in DV val set
HDR10_EXTRA_CHECKPOINTS = {
    "XGBoost delta (MAE 0.06094)": r"xgb://F:\DTMModelData\xgb_delta_model.pkl",
}
GATE_THRESHOLDS = {}
VAL_CSV       = r"F:\DTMModelData\val\val_dataset.csv"
N_SHOW        = None   # None = show ALL scenes (sorted by advantage)
DV_RENDER_EXE = r"C:\Code\libplacebo\build\tools\dv_render.exe"
DATASET_ROOTS = [r"D:\Jdownloader\Dataset", r"G:\Dataset"]

# Map title_key -> video folder (same as stage2_pixel_extract.py)
TITLE_KEY_TO_VIDEO_FOLDER = {
    "andor_s02":"Andor.S02.2160p.DSNP.WEB-DL.DDP5.1.DV.H.265-NTb",
    "bad_batch_s03":"Star.Wars.The.Bad.Batch.S03.2160p.DSNP.WEB-DL.DDP5.1.DoVi.H.265-NTb",
    "born_to_be_wild_s01":"Born.to.Be.Wild.2025.S01.2160p.ATVP.WEB-DL.DDP5.1.DV.HEVC-NTb",
    "euphoria_s03":"Euphoria.US.S03.2160p.HMAX.WEB-DL.DDP5.1.DV.H.265-NTb",
    "for_all_mankind_s05":"For.All.Mankind.S05.2160p.ATVP.WEB-DL.DDP5.1.DV.H.265-NTb",
    "house_of_the_dragon_s03":"House.of.the.Dragon.S03.2160p.HMAX.WEB-DL.DDP5.1.DV.H.265-NTb",
    "mindhunter_s01":"Mindhunter.S01.2160p.NF.WEB-DL.DDP5.1.DV.H.265-Kitsune",
    "monarch_s02":"Monarch.Legacy.of.Monsters.S02.2160p.ATVP.WEB-DL.DDP5.1.DV.H.265-NTb",
    "our_living_world_s01":"Our.Living.World.S01.2160p.NF.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX",
    "our_oceans_s01":"Our.Oceans.(2024).S01.(2160p.NF.WEB-DL.H265.DV.DDP.Atmos.5.1.English.-.HONE)",
    "our_planet_s01":"Our.Planet.2019.S01.2160p.NF.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX",
    "penguin_s01":"The.Penguin.S01.2160p.MAX.WEB-DL.DDP5.1.DoVi.x265-NTb",
    "prehistoric_planet_s03":"Prehistoric.Planet.2022.S03.2160p.ATVP.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX",
    "rings_of_power_s02":"The.Lord.of.the.Rings.The.Rings.of.Power.S02.2160p.AMZN.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX",
    "sandman_s01":"The.Sandman.S01.2160p.NF.WEB-DL.DDP.5.1.Atmos.DV.H.265-CHDWEB",
    "shogun_s01":"Shogun.2024.S01.2160p.DSNP.WEB-DL.DDP5.1.DV.H.265-Kitsune",
    "silo_s03":"Silo.S03E01.Who.Are.You.2160p.ATVP.WEB-DL.DDP5.1.DV.H.265",
    "stranger_things_s05":"Stranger.Things.S05.2160p.NF.WEB-DL.DDP5.1.DV.H.265-NTb",
    "tales_empire_s01":"Star.Wars.Tales.of.the.Empire.S01.REPACK.2160p.DSNP.WEB-DL.DDP5.1.DoVi.HEVC-NTb",
    "ted_lasso_s03":"Ted.Lasso.S03.2160p.ATVP.WEB-DL.DDP5.1.DoVi.H.265-NTb",
    "the_last_of_us_s02":"The.Last.of.Us.S02.2160p.MAX.WEB-DL.DDP5.1.DV.x265-NTb",
    "the_mandalorian_s01":"The.Mandalorian.S01.2160p.DSNP.WEB-DL.DDP5.1.Atmos.DV.HEVC-MZABI",
    "the_witcher_s04":"The.Witcher.S04.2160p.NF.WEB-DL.DDP5.1.DV.H.265-NTb",
    "wondla_s03":"WondLa.S03.2160p.ATVP.WEB-DL.DDP5.1.DV.HEVC-BYNDR",
}

_COEF_SCALE = 1.0 / (2**23)

def curve_to_rpu_linear(curve256, n_segs=8):
    """Convert 256-pt ML curve to piecewise linear RPU_POLY_1D (same format as DV gold).
    Piecewise linear (order=1, c2=0) with evenly-spaced pivots — C0 continuous."""
    xs = np.linspace(0, 1, 256)
    pivots = np.linspace(0, 1, n_segs + 1)
    segs = []
    for i in range(n_segs):
        x0, x1 = pivots[i], pivots[i+1]
        y0 = float(np.interp(x0, xs, curve256))
        y1 = float(np.interp(x1, xs, curve256))
        c1 = (y1 - y0) / (x1 - x0)
        c0 = y0 - c1 * x0
        segs.append((c0, c1))
    return pivots, segs

def write_rpu_linear(pivots, segs, path):
    with open(path, 'w') as f:
        f.write("RPU_POLY_1D\n")
        f.write(f"num_segs {len(segs)}\n")
        f.write("pivots " + " ".join(f"{p:.8f}" for p in pivots) + "\n")
        for i, (c0, c1) in enumerate(segs):
            f.write(f"seg {i} 1 {c0:.8f} {c1:.8f} 0.00000000\n")

def find_video(title_key, episode_stem):
    folder = TITLE_KEY_TO_VIDEO_FOLDER.get(title_key)
    if not folder: return None
    for root in DATASET_ROOTS:
        title_dir = os.path.join(root, folder)
        if not os.path.isdir(title_dir): continue
        for ext in (".mkv", ".mp4"):
            p = os.path.join(title_dir, episode_stem + ext)
            if os.path.exists(p): return p
        for f in os.listdir(title_dir):
            if episode_stem[:35].lower() in f.lower():
                return os.path.join(title_dir, f)
    return None

def render_frame_lut(video_path, pts_time, mode, ml_curve=None,
                     l1_max=0.0, l1_avg=0.0, width=720, height=405, out_nits=143):
    """Call dv_render.exe and return RGB uint8 array or None."""
    env = os.environ.copy()
    env['PATH'] = r'C:\Code\libplacebo\build\src;C:\msys64\ucrt64\bin;' + env.get('PATH','')
    cmd = [DV_RENDER_EXE, "--input", video_path, "--pts", f"{pts_time:.6f}",
           "--mode", mode, "--width", str(width), "--height", str(height),
           "--out-nits", str(out_nits)]
    if l1_max > 0:
        cmd += ["--l1-max", f"{l1_max:.6f}", "--l1-avg", f"{l1_avg:.6f}"]
    lut_tmp = None
    if mode == "ml" and ml_curve is not None:
        # Convert 256-pt curve to DV RPU piecewise linear format — same pipeline as gold
        pivots, segs = curve_to_rpu_linear(ml_curve, n_segs=8)
        lut_tmp = tempfile.NamedTemporaryFile(suffix=".rpu", delete=False, mode='w')
        write_rpu_linear(pivots, segs, lut_tmp.name)
        lut_tmp.close()
        cmd += ["--lut", lut_tmp.name]
    try:
        r = subprocess.run(cmd, capture_output=True, env=env, timeout=30)
        expected = width * height * 3
        if len(r.stdout) == expected:
            return np.frombuffer(r.stdout, dtype=np.uint8).reshape(height, width, 3).copy()
    except Exception:
        pass
    finally:
        if lut_tmp and os.path.exists(lut_tmp.name): os.unlink(lut_tmp.name)
    return None

xs = np.linspace(0, 1, 256)

# ---------------------------------------------------------------------------
# Cache: load dataset + run inference once per checkpoint
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and running inference...")
def load_results(ckpt_path: str, run_name: str = ""):
    if ckpt_path.startswith("xgb://"):
        # XGBoost model — return empty results (DV val set eval not supported)
        import pandas as _pd
        empty = _pd.DataFrame(columns=["title_key","episode","scene_id","cell","pts_time",
                                        "ml_mae","ml_gate_mae","spline_mae","ml_mt_mae",
                                        "spl_mt_mae","advantage","mt_advantage"])
        return empty, np.zeros((0,256)), np.zeros((0,256)), np.zeros((0,256))
    mlp.BAR_FEATURE_COLS = []
    df = pd.read_csv(VAL_CSV)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_cols = ckpt["feat_cols"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_residual    = any("residual_mcp_dec" in k for k in ckpt["model_state"])
    is_residual_l1 = any("residual_l1_dec"  in k for k in ckpt["model_state"])

    # Derive MCP_K from checkpoint mcp_head.bias shape BEFORE creating the model.
    # This avoids stale module globals from a previously loaded checkpoint.
    ckpt_mcp_k = len(ckpt["model_state"]["mcp_head.bias"])   # e.g. 8 for Run14, 12 for ResidualL1
    mlp.MCP_K         = ckpt_mcp_k
    mlp.MCP_K_SHAPE   = ckpt_mcp_k - 1
    mlp.MCP_N_KNOTS   = ckpt_mcp_k   # = MCP_K_SHAPE + 1

    ckpt_knots  = ckpt["model_state"].get("mcp_eval.x_knots")
    ckpt_k_eval = (len(ckpt_knots) - 1) if ckpt_knots is not None else (ckpt_mcp_k - 1)

    model = mlp.DVPolyMLP(pixel_dim=len(feat_cols), embed_dim=8, hidden_dim=128,
                          dropout=0.0, use_tier_embed=False, has_trim_head=False)
    model.mcp_eval = mlp.MonotoneControlPoints(k=ckpt_k_eval)

    if is_residual:
        model.set_residual_mode([feat_cols.index(f"spline_k{i}") for i in range(8)])
    if is_residual_l1:
        model.residual_l1_dec = mlp.ResidualL1MCP(k=ckpt_mcp_k)

    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    if is_residual:    model.residual_mcp_dec = model.residual_mcp_dec.to(device)
    if is_residual_l1: model.residual_l1_dec  = model.residual_l1_dec.to(device)
    model.eval()

    ds = mlp.DVNRDataset(df, feat_cols)

    # Run inference
    ml_curves   = []
    gold_curves = []
    spline_curves = []

    with torch.no_grad():
        for s in range(0, len(ds), 512):
            e = min(s+512, len(ds))
            feats = torch.tensor(ds.X[s:e], dtype=torch.float32, device=device)
            tiers = torch.zeros(e-s, dtype=torch.long, device=device)
            mcp, _, _ = model(feats, tiers)
            if is_residual_l1:
                spline_q = torch.tensor(ds.spline_q_raw[s:e], dtype=torch.float32, device=device)
                pred, _  = model.residual_l1_dec(mcp, spline_q)
                pred = pred.cpu().numpy()
            elif is_residual:
                pred = model.mcp_eval_residual(mcp, feats).cpu().numpy()
            else:
                pred = model.mcp_eval(mcp).cpu().numpy()
            ml_curves.append(pred)
            gold_curves.append(ds.curves[s:e])
            spline_curves.append(ds.splines[s:e])

    ml_arr     = np.vstack(ml_curves)      # [N, 256]
    gold_arr   = np.vstack(gold_curves)    # [N, 256]
    spline_arr = np.vstack(spline_curves)  # [N, 256]

    # Always compute gated version (threshold=0.25) for comparison in summary table
    GATE_T = 0.30   # 0.30 leaves expansion content untouched (p95 legitimate = 0.11)
    ml_arr_raw   = ml_arr.copy()   # keep raw predictions for ML MAE column
    gate_mask    = (ml_arr_raw - spline_arr) > GATE_T
    ml_arr_gated = np.where(gate_mask, spline_arr, ml_arr_raw)
    n_gated      = gate_mask.any(axis=1).sum()
    print(f"  Gate@{GATE_T}: {n_gated:,} scenes would be gated", flush=True)

    # Apply gate to primary curves shown in plots if configured for this run
    gate_thresh = GATE_THRESHOLDS.get(run_name)
    if gate_thresh is not None:
        ml_arr = ml_arr_gated  # gated curves shown in plot

    # Midtone-weighted MAE: 3x weight on x in [0.2, 0.7] — where colorist decisions are visible
    _mt_w = np.where((xs >= 0.2) & (xs <= 0.7), 3.0, 1.0)
    _mt_w = _mt_w / _mt_w.mean()   # normalise so overall scale stays comparable

    ml_mae      = np.abs(ml_arr_raw   - gold_arr).mean(axis=1)
    ml_gate_mae = np.abs(ml_arr_gated - gold_arr).mean(axis=1)
    spline_mae  = np.abs(spline_arr   - gold_arr).mean(axis=1)
    ml_mt_mae   = (np.abs(ml_arr_raw - gold_arr) * _mt_w).mean(axis=1)
    spl_mt_mae  = (np.abs(spline_arr - gold_arr) * _mt_w).mean(axis=1)
    advantage   = spline_mae - ml_mae  # positive = ML wins

    # Build results frame (include l1 metadata for rendering, pts_time for seeking)
    keep_cols = ["title_key","episode","scene_id","cell","pts_time"]
    for c in ["l1_max_pq","l1_avg_pq"]:
        if c in ds.df_valid.columns: keep_cols.append(c)
    results = ds.df_valid[keep_cols].copy().reset_index(drop=True)
    results["ml_mae"]      = ml_mae
    results["ml_gate_mae"] = ml_gate_mae
    results["spline_mae"]  = spline_mae
    results["ml_mt_mae"]   = ml_mt_mae
    results["spl_mt_mae"]  = spl_mt_mae
    results["advantage"]    = advantage                       # regular MAE: + = ML wins
    results["mt_advantage"] = spl_mt_mae - ml_mt_mae         # midtone MAE: + = ML wins

    return results, ml_arr, gold_arr, spline_arr


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Curve Explorer", layout="wide")
st.title("ML vs Spline vs Gold — Curve Explorer")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    run_name = st.selectbox("Model checkpoint", list(CHECKPOINTS.keys()))
    ckpt_path = CHECKPOINTS[run_name]

    results, ml_arr, gold_arr, spline_arr = load_results(ckpt_path, run_name)

    # Cell selector
    cells = sorted(results["cell"].unique(),
                   key=lambda c: -(results["cell"]==c).sum())
    cell = st.selectbox("Cell type", cells,
                        format_func=lambda c: f"{c}  ({(results['cell']==c).sum():,} scenes)")

    mode = st.radio("Show", ["Worst (ML < spline)", "Best (ML > spline)"], index=0)
    sort_metric = st.radio("Sort by", ["Midtone MAE (shape)", "Regular MAE"], index=0)

# ---------------------------------------------------------------------------
# Tabs — use tab objects as containers to avoid re-indenting all content
# ---------------------------------------------------------------------------
_tab_dv, _tab_hdr = st.tabs(["DV Val Set", "HDR10 MKV"])
_C = _tab_dv   # active container: DV tab

# Filter to selected cell
cell_df = results[results["cell"] == cell].copy()
adv_col = "mt_advantage" if "Midtone" in sort_metric else "advantage"
label_adv = "MT-adv" if "Midtone" in sort_metric else "adv"

if "Worst" in mode:
    cell_df = cell_df.sort_values(adv_col)
    n_bad = (cell_df[adv_col] < 0).sum()
    title_str = f"{len(cell_df):,} scenes — Worst first ({n_bad:,} ML loses) — sorted by {label_adv}"
else:
    cell_df = cell_df.sort_values(adv_col, ascending=False)
    n_good = (cell_df[adv_col] > 0).sum()
    title_str = f"{len(cell_df):,} scenes — Best first ({n_good:,} ML wins) — sorted by {label_adv}"

_C.subheader(f"{cell}  —  {title_str}")

def scene_label(row):
    ep = row["episode"][:50] if len(row["episode"]) > 50 else row["episode"]
    mt_adv = row["mt_advantage"]
    adv    = row["advantage"]
    return (f"{row['title_key']} | {ep} | scene {row['scene_id']}  "
            f"MT-adv={mt_adv:+.4f}  adv={adv:+.4f}  "
            f"ML={row['ml_mae']:.4f} Spl={row['spline_mae']:.4f}")

labels = cell_df.apply(scene_label, axis=1).tolist()
idx_map = dict(zip(labels, cell_df.index.tolist()))

selected_label = st.selectbox("Scene", labels)
sel_idx = idx_map[selected_label]

row = results.loc[sel_idx]
ml_curve     = ml_arr[sel_idx]
gold_curve   = gold_arr[sel_idx]
spline_curve = spline_arr[sel_idx]

# ---------------------------------------------------------------------------
# Curve plot
# ---------------------------------------------------------------------------
col1, col2 = _C.columns([2, 1])

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=gold_curve,   name="Gold (DV poly)",  line=dict(color="#2ecc71", width=2)))
    fig.add_trace(go.Scatter(x=xs, y=spline_curve, name="Spline (libplacebo)", line=dict(color="#3498db", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=ml_curve,     name=f"ML ({run_name[:6]})", line=dict(color="#e74c3c", width=2)))
    fig.add_trace(go.Scatter(x=xs, y=xs, name="Identity", line=dict(color="#999", width=1, dash="dot")))

    fig.update_layout(
        title=f"Tone mapping curves — {row['cell']}",
        xaxis_title="Input signal (PQ [0,1])",
        yaxis_title="Output signal (PQ [0,1])",
        legend=dict(x=0.02, y=0.98),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Deviation plot (curve - identity)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xs, y=gold_curve-xs,   name="Gold deviation",   line=dict(color="#2ecc71", width=2)))
    fig2.add_trace(go.Scatter(x=xs, y=spline_curve-xs, name="Spline deviation", line=dict(color="#3498db", width=2, dash="dash")))
    fig2.add_trace(go.Scatter(x=xs, y=ml_curve-xs,     name="ML deviation",     line=dict(color="#e74c3c", width=2)))
    fig2.add_hline(y=0, line_dash="dot", line_color="#999")
    fig2.update_layout(
        title="Deviation from identity (positive = lift, negative = compress)",
        xaxis_title="Input PQ", yaxis_title="Deviation",
        legend=dict(x=0.02, y=0.02), height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.metric("ML MAE vs gold",     f"{row['ml_mae']:.5f}")
    st.metric("Spline MAE vs gold", f"{row['spline_mae']:.5f}")
    delta = row['advantage']
    st.metric("ML advantage",       f"{delta:+.5f}",
              delta=f"{'ML wins' if delta > 0 else 'Spline wins'}",
              delta_color="normal" if delta > 0 else "inverse")

    st.divider()
    st.write("**Scene info**")
    st.write(f"Title: `{row['title_key']}`")
    ep = row['episode']
    st.write(f"Episode: `{ep[:60]}{'...' if len(ep)>60 else ''}`")
    st.write(f"Scene ID: `{row['scene_id']}`")
    st.write(f"Cell: `{row['cell']}`")

    st.divider()
    st.write("**Cell stats**")
    full_cell = results[results["cell"] == cell]
    st.write(f"Scenes in cell: {len(full_cell):,}")
    st.write(f"ML wins: {(full_cell['advantage']>0).sum():,} ({(full_cell['advantage']>0).mean()*100:.1f}%)")
    st.write(f"Mean ML MAE: {full_cell['ml_mae'].mean():.5f}")
    st.write(f"Mean Spl MAE: {full_cell['spline_mae'].mean():.5f}")

# ---------------------------------------------------------------------------
# Frame rendering panel
_C.divider()
_C.subheader("Rendered frames")

video_path = find_video(row["title_key"], row["episode"])
if video_path:
    pts_time = float(row["pts_time"])
    l1_max   = float(results.loc[sel_idx, "l1_max_pq"]) / 4095.0 if "l1_max_pq" in results.columns else 0.0
    l1_avg   = float(results.loc[sel_idx, "l1_avg_pq"]) / 4095.0 if "l1_avg_pq" in results.columns else 0.0
    ml_curve = ml_arr[sel_idx]   # [256] float32

    rc1, rc2 = _C.columns([3, 1])
    with rc2:
        target_nits = st.slider("Target peak (nits)", min_value=48, max_value=400,
                                value=143, step=1,
                                help="out-nits passed to dv_render.exe — 143 is the training target")

    with rc1:
        if st.button("Render Gold / Spline / ML  (3-5 sec)"):
            with st.spinner("Rendering frames..."):
                img_gold   = render_frame_lut(video_path, pts_time, "gold",
                                              l1_max=l1_max, l1_avg=l1_avg, out_nits=target_nits)
                img_spline = render_frame_lut(video_path, pts_time, "spline",
                                              l1_max=l1_max, l1_avg=l1_avg, out_nits=target_nits)
                img_ml     = render_frame_lut(video_path, pts_time, "ml",
                                              ml_curve=ml_curve, l1_max=l1_max, l1_avg=l1_avg,
                                              out_nits=target_nits)
            fc1, fc2, fc3 = st.columns(3)
            cap = f"  ({target_nits} nits)"
            if img_gold   is not None: fc1.image(img_gold,   caption="Gold (DV poly)" + cap,      use_container_width=True)
            else:                      fc1.warning("Gold render failed")
            if img_spline is not None: fc2.image(img_spline, caption="Spline (libplacebo)" + cap, use_container_width=True)
            else:                      fc2.warning("Spline render failed")
            if img_ml     is not None: fc3.image(img_ml,     caption=f"ML{cap}",                  use_container_width=True)
            else:                      fc3.warning("ML render failed")
else:
    _C.info(f"Video not found on disk for {row['title_key']} — copy to D:\\Jdownloader\\Dataset to enable rendering")

# ---------------------------------------------------------------------------
# Cell summary table
# ---------------------------------------------------------------------------
_C.divider()
_C.subheader("All cells — summary")
summary = results.groupby("cell").agg(
    N=("ml_mae","count"),
    ml_mae=("ml_mae","mean"),
    spline_mae=("spline_mae","mean"),
    ml_mt_mae=("ml_mt_mae","mean"),
    spl_mt_mae=("spl_mt_mae","mean"),
    win_pct=("advantage",    lambda x: (x>0).mean()*100),
    mt_win_pct=("mt_advantage", lambda x: (x>0).mean()*100),
).reset_index().sort_values("N", ascending=False)
summary["vs_spline%"]    = ((summary["spline_mae"]  - summary["ml_mae"])    / summary["spline_mae"]  * 100).round(1)
summary["mt_vs_spline%"] = ((summary["spl_mt_mae"]  - summary["ml_mt_mae"]) / summary["spl_mt_mae"]  * 100).round(1)

disp = summary.reset_index(drop=True).rename(columns={
    "cell":"Cell","N":"N scenes",
    "ml_mae":"ML MAE","spline_mae":"Spline MAE",
    "ml_mt_mae":"ML MT-MAE","spl_mt_mae":"Spl MT-MAE",
    "win_pct":"Win%","mt_win_pct":"MT Win%",
    "vs_spline%":"vs Spline%","mt_vs_spline%":"MT vs Spline%"
}).copy()
for c in ["ML MAE","Spline MAE","ML MT-MAE","Spl MT-MAE"]:
    disp[c] = disp[c].map("{:.5f}".format)
for c in ["Win%","MT Win%","vs Spline%","MT vs Spline%"]:
    disp[c] = disp[c].map("{:+.1f}%".format)
_C.dataframe(disp, use_container_width=True)
_C.caption("MT-MAE = midtone-weighted MAE (3x weight on PQ [0.2-0.7]). MT Win% = % scenes where ML beats spline on midtone shape. If MT Win% < Win%, ML is winning via endpoint match not midtone shape.")

# ---------------------------------------------------------------------------
# Title × Cell distribution
# ---------------------------------------------------------------------------
_C.divider()
_C.subheader("Val set — title x cell distribution (scene counts)")

pivot = (results
    .groupby(["title_key", "cell"])
    .size()
    .unstack(fill_value=0)
    .sort_index())

# Add row total
pivot["TOTAL"] = pivot.sum(axis=1)
# Sort columns: cells by total scenes desc, then TOTAL last
cell_cols = [c for c in pivot.columns if c != "TOTAL"]
cell_cols_sorted = sorted(cell_cols, key=lambda c: -pivot[c].sum())
pivot = pivot[cell_cols_sorted + ["TOTAL"]]

# Colour-map each cell column independently so sparse cells are still readable
_C.dataframe(
    pivot.style.background_gradient(cmap="Blues", axis=0, subset=cell_cols_sorted),
    use_container_width=True,
    height=min(60 + len(pivot) * 35, 700),
)

# ---------------------------------------------------------------------------
# TAB 2 — HDR10 MKV Viewer  (use proper with block for reliable rendering)
# ---------------------------------------------------------------------------
BASELINE_EXE = r"C:\Code\libplacebo\build\tools\libplacebo_baseline_eval.exe"
DAEMON_BIN   = r"C:\Code\libplacebo\build\tools\pl_extract_features_daemon.exe"
_SPLINE_X_KNOTS = np.linspace(0, 1, 8, dtype=np.float32)

def _get_frame_count(video_path):
    try:
        rd = subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
                             '-show_entries','format=duration','-of','csv=p=0', video_path],
                            capture_output=True, text=True, timeout=8)
        rf = subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
                             '-show_entries','stream=r_frame_rate','-of','csv=p=0', video_path],
                            capture_output=True, text=True, timeout=8)
        dur = float(rd.stdout.strip())
        fps_s = rf.stdout.strip().split(',')[0]
        fps = float(fps_s.split('/')[0]) / float(fps_s.split('/')[1]) if '/' in fps_s else float(fps_s)
        return max(1, int(dur * fps)), fps
    except Exception:
        return 100000, 24.0

def _frame_to_pts(frame_idx, fps):
    return frame_idx / fps

def _nits_to_pq(nits):
    L = max(float(nits), 1e-6) / 10000.0
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lm1 = L ** m1
    return ((c1 + c2 * Lm1) / (1 + c3 * Lm1)) ** m2

def _write_ml_lut_from_curve(curve256, path, maxscl, out_nits, n_pts=512):
    """
    Convert 256-pt MCP curve to x,y LUT for --mode ml-lut.

    dv_render ml_tone_map() normalises input by dividing by input_max (= maxscl):
        xn = x_absolute_pq / maxscl
        output = yn * maxscl
    So LUT x must be in [0,1] where x=1.0 = scene peak (maxscl), and
    LUT y must also be normalised: y_lut = y_pq / maxscl.

    Without this, the render samples the wrong part of the curve and produces
    desaturated/incorrect output.
    """
    src_xs = np.linspace(0, 1, 256)          # MCP curve: absolute PQ [0,1]
    xn     = np.linspace(0, 1, n_pts, dtype=np.float32)  # LUT x: normalised to scene peak
    x_abs  = xn * maxscl                      # corresponding absolute PQ values

    # Sample MCP curve at the absolute PQ positions corresponding to each xn
    y_abs = np.interp(x_abs, src_xs, curve256).astype(np.float32)
    y_abs = np.maximum.accumulate(np.clip(y_abs, 0, None))

    # Normalise output by maxscl so output_pq = y_lut * maxscl
    yn = y_abs / max(maxscl, 1e-6)

    # Scale so yn at scene peak (xn=1) matches target display ceiling
    if out_nits > 0:
        target_pq = _nits_to_pq(out_nits)
        target_yn = target_pq / max(maxscl, 1e-6)
        peak_yn   = float(yn[-1])
        if peak_yn > 0.01:
            yn = np.clip(yn * (target_yn / peak_yn), 0, target_yn + 0.05)

    with open(path, 'w') as f:
        for x, y in zip(xn, yn):
            f.write(f'{x:.6f} {y:.6f}\n')

def _compute_spline_curve(l1_max, l1_avg, target_nits, n_pts=256):
    """Run libplacebo_baseline_eval and return (spline_256pt, spline_k8).
    spline_256pt: full curve over [0,1] for plotting.
    spline_k8: sampled at 8 uniform knot positions (model features).
    """
    identity = np.linspace(0, 1, n_pts, dtype=np.float32)
    if not Path(BASELINE_EXE).exists():
        return identity, _SPLINE_X_KNOTS.copy()
    inp = f"scene_id,maxscl,l1_avg_pq,target_nits\n0,{l1_max:.6f},{l1_avg:.6f},{target_nits}\n"
    try:
        _env = os.environ.copy()
        _env['PATH'] = r'C:\Code\libplacebo\build\src;C:\msys64\ucrt64\bin;' + _env.get('PATH', '')
        r = subprocess.run([BASELINE_EXE], input=inp, capture_output=True, text=True, timeout=5, env=_env)
        lines = r.stdout.strip().split('\n')
        if len(lines) >= 2:
            raw = np.array([float(x) for x in lines[1].split(',')[1:]], dtype=np.float32)
            lut_xs = np.linspace(0, l1_max, len(raw))
            eval_xs = np.linspace(0, 1, n_pts)

            # Above l1_max: extrapolate using the slope at the last two LUT points
            # rather than a hard constant — gives a more accurate view of what
            # libplacebo's spline actually does beyond the scene peak.
            if len(raw) >= 2 and l1_max < 1.0:
                dx = lut_xs[-1] - lut_xs[-2]
                dy = raw[-1] - raw[-2]
                slope = dy / dx if dx > 0 else 0.0
                # Extrapolate linearly but cap at 1.0 to avoid out-of-range values
                y_at_1 = float(np.clip(raw[-1] + slope * (1.0 - l1_max), 0.0, 1.0))
            else:
                y_at_1 = float(raw[-1])

            ext_xs = np.concatenate([lut_xs, [1.0]])
            ext_ys = np.concatenate([raw, [y_at_1]])

            spline_curve = np.interp(eval_xs, ext_xs, ext_ys).astype(np.float32)
            spline_k = np.interp(_SPLINE_X_KNOTS, ext_xs, ext_ys).astype(np.float32)
            return spline_curve, spline_k
    except Exception:
        pass
    return identity, _SPLINE_X_KNOTS.copy()

def _get_daemon():
    """Get a healthy daemon — restarts automatically if the process has died."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
    from libplacebo_daemon_client import SustainedFeatureExtractor
    key = "_hdr10_daemon"
    ext = st.session_state.get(key)
    if ext is None or ext.process is None or ext.process.poll() is not None:
        # Dead or missing — create fresh instance
        if ext is not None:
            try: ext.stop()
            except Exception: pass
        ext = SustainedFeatureExtractor()
        ext.start()
        st.session_state[key] = ext
    return ext

def _infer_mcp_hdr10(features, feat_cols, feat_mean, feat_std, model, device, target_nits):
    """Run inference for HDR10 frame. Returns (256-pt ml_curve, l1_max, l1_avg, spline_curve, spline_k)."""
    l1_max = float(features.get('maxscl', 0.5))
    l1_avg = float(features.get('average_maxrgb', 0.3))
    spline_curve, spline_k = _compute_spline_curve(l1_max, l1_avg, target_nits)

    # spline_k* and spline_km_* both use maxscl-based spline at inference
    # (Run 27 feat_cols has spline_km_*; older runs had spline_k*)
    spline_lookup = {}
    for i in range(8):
        v = float(spline_k[i])
        spline_lookup[f'spline_k{i}']   = v   # old naming (Run 14 compat)
        spline_lookup[f'spline_km_{i}'] = v   # new naming (Run 27)

    feat_vals = []
    for col in feat_cols:
        if col in features:
            feat_vals.append(float(features[col]))
        elif col in spline_lookup:
            feat_vals.append(spline_lookup[col])
        elif col == 'l1_max_pq':
            feat_vals.append(l1_max)   # proxy — only used for pre-Run27 checkpoints
        elif col == 'l1_avg_pq':
            feat_vals.append(l1_avg)   # proxy
        else:
            feat_vals.append(0.0)

    # Apply training normalisation
    raw = np.array(feat_vals, dtype=np.float32)
    if feat_mean is not None and feat_std is not None:
        n = min(len(raw), len(feat_mean))
        raw[:n] = (raw[:n] - feat_mean[:n]) / (feat_std[:n] + 1e-8)
    feats_t = torch.tensor([raw], dtype=torch.float32, device=device)
    with torch.no_grad():
        mcp_raw, _, _ = model(feats_t, torch.zeros(1, dtype=torch.long, device=device))
        curve = model.mcp_eval(mcp_raw)[0].cpu().numpy()

    return curve, l1_max, l1_avg, spline_curve, spline_k

def _render_hdr10(video_path, pts, mode, out_nits, lut_path=None, l1_max=0.0, l1_avg=0.0,
                  width=1280, height=720):
    """Render one frame. Retries at ±1/2/4s if the exact PTS is undecodable (HEVC GOP issue)."""
    env = os.environ.copy()
    env['PATH'] = r'C:\Code\libplacebo\build\src;C:\msys64\ucrt64\bin;' + env.get('PATH','')
    expected = width * height * 3

    for offset in [0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0]:
        p = max(0.0, pts + offset)
        cmd = [DV_RENDER_EXE, '--input', video_path, '--pts', f'{p:.3f}',
               '--mode', mode, '--width', str(width), '--height', str(height),
               '--out-nits', str(out_nits),
               '--l1-max', f'{l1_max:.6f}', '--l1-avg', f'{l1_avg:.6f}']
        if lut_path:
            cmd += ['--lut', lut_path]
        try:
            r = subprocess.run(cmd, capture_output=True, env=env, timeout=30)
            if len(r.stdout) == expected:
                return np.frombuffer(r.stdout, dtype=np.uint8).reshape(height, width, 3).copy()
        except Exception:
            pass
    return None

def _is_xgb(ckpt_path: str) -> bool:
    return ckpt_path.startswith("xgb://")

@st.cache_resource(show_spinner="Loading XGBoost model...")
def _load_xgb_model(pkl_path: str):
    import pickle
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feat_cols"], bundle["knot_indices"]

def _infer_xgb_hdr10(features, feat_cols, xgb_model, knot_indices, spline_k, l1_max, l1_avg, target_nits):
    """Run XGBoost delta inference for one HDR10 frame."""
    # Build raw feature vector (NO normalisation — trees are scale-invariant)
    spline_k_dict = {f"spline_km_{i}": float(spline_k[i]) for i in range(8)}
    spline_k_dict.update({f"spline_k{i}": float(spline_k[i]) for i in range(8)})
    fv = []
    for col in feat_cols:
        if col in features:         fv.append(float(features[col]))
        elif col in spline_k_dict:  fv.append(spline_k_dict[col])
        elif col == "l1_max_pq":    fv.append(l1_max)
        elif col == "l1_avg_pq":    fv.append(l1_avg)
        else:                       fv.append(0.0)
    X = np.array([fv], dtype=np.float32)

    # Predict 8 delta values
    delta_pred = xgb_model.predict(X)[0].astype(np.float32)  # [8]

    # Reconstruct corrected knots
    corrected_knots = np.clip(spline_k + delta_pred, 0.0, 1.0)
    corrected_knots = np.maximum.accumulate(corrected_knots)

    # Steffen interpolation → 256-pt curve
    from ml.train_xgb_delta import steffen_interp_numpy
    ml_curve = np.maximum.accumulate(np.clip(steffen_interp_numpy(corrected_knots), 0.0, 1.0))
    return ml_curve

@st.cache_resource(show_spinner="Loading HDR10 model...")
def _load_hdr10_model(ckpt_path_key):
    ckpt = torch.load(ckpt_path_key, map_location="cpu", weights_only=False)
    feat_cols  = ckpt["feat_cols"]
    feat_mean  = ckpt.get("feat_mean")
    feat_std   = ckpt.get("feat_std")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp.MCP_K       = len(ckpt["model_state"]["mcp_head.bias"])
    mlp.MCP_K_SHAPE = mlp.MCP_K - 1
    mlp.MCP_N_KNOTS = mlp.MCP_K
    model = mlp.DVPolyMLP(pixel_dim=len(feat_cols), dropout=0.0,
                           use_tier_embed=False, has_trim_head=False)
    ck = len(ckpt["model_state"].get("mcp_eval.x_knots", torch.zeros(8))) - 1
    model.mcp_eval = mlp.MonotoneControlPoints(k=ck)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    return model, feat_cols, feat_mean, feat_std, device

_HDR_LOG_FILE = "C:/tmp/hdr10_trace.log"

def _hlog(stage: str, msg: str):
    """Write to trace log file and Streamlit session log list."""
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] [{stage:20s}] {msg}"
    print(line, flush=True)
    try:
        with open(_HDR_LOG_FILE, "a", encoding="utf-8") as _f:
            _f.write(line + "\n")
    except Exception:
        pass
    log_list = st.session_state.get("hdr10_log", [])
    log_list.append(line)
    if len(log_list) > 40:
        log_list = log_list[-40:]
    st.session_state["hdr10_log"] = log_list

# ── HDR10 UI — use proper with block so all st.* calls route to the tab ──
with _tab_hdr:
    st.subheader("HDR10 MKV Viewer — Spline vs ML")

    # HDR10 checkpoint selector — includes XGBoost + all DV models
    _hdr10_ckpts = {**HDR10_EXTRA_CHECKPOINTS, **CHECKPOINTS}
    _hdr10_run = st.selectbox("HDR10 model", list(_hdr10_ckpts.keys()), key="hdr10_model")
    _hdr10_ckpt = _hdr10_ckpts[_hdr10_run]

    hdr_col_path, hdr_col_nits = st.columns([4, 1])
    with hdr_col_path:
        hdr_path = st.text_input("HDR10 MKV path", placeholder="G:/path/to/content.mkv",
                                  key="hdr10_path")
    with hdr_col_nits:
        hdr_nits = st.number_input("Target nits", min_value=48, max_value=2000,
                                    value=50, step=1, key="hdr10_nits")

    if hdr_path and Path(hdr_path).exists():
        n_frames, fps = _get_frame_count(hdr_path)
        st.caption(f"{Path(hdr_path).name}  |  ~{n_frames:,} frames  |  {fps:.3f} fps")

        hf1, hf2 = st.columns([3, 1])
        with hf1:
            hdr_frame = st.slider("Frame", 0, n_frames - 1, min(72500, n_frames - 1),
                                   step=24, key="hdr10_frame_slider")
        with hf2:
            hdr_frame = int(st.number_input("Frame (exact)", 0, n_frames - 1,
                                             value=hdr_frame, step=1, key="hdr10_frame_num"))

        pts_hdr = _frame_to_pts(hdr_frame, fps)
        st.caption(f"PTS: {pts_hdr:.3f}s  ({pts_hdr/60:.1f} min)")
        _hlog("INPUT", f"frame={hdr_frame}  pts={pts_hdr:.3f}s  nits={hdr_nits}")

        # ── Beta controls ────────────────────────────────────────────────────
        # auto_beta is computed DURING render from raw ML curve (correct for current frame)
        # and stored in cached result. Display it here from cache.
        cached_for_beta = st.session_state.get("hdr10_cache", {})
        beta_auto_display = cached_for_beta.get("beta_auto", 0.6)

        beta_row1, beta_row2, beta_row3 = st.columns([1, 1, 3])
        with beta_row1:
            st.metric("Auto β", f"{beta_auto_display:.2f}",
                      help="shadow gap mean(max(spline-ML,0)) in x<0.2 PQ / 0.05. Updates after each render.")
        with beta_row2:
            use_auto = st.checkbox("Use auto", value=True, key="hdr10_use_auto")
        with beta_row3:
            shadow_beta_manual = st.slider(
                "Manual β override", 0.0, 1.0,
                value=float(st.session_state.get("hdr10_beta_manual", beta_auto_display)),
                step=0.05, key="hdr10_beta_manual",
                help="0=pure ML, 0.6=dark scene default, 1.0=full lift to spline")

        # Effective beta — render_key includes it so changing manual β re-renders
        shadow_beta = beta_auto_display if use_auto else shadow_beta_manual
        render_key = (hdr_path, hdr_frame, int(hdr_nits), _hdr10_ckpt,
                      "auto" if use_auto else round(shadow_beta_manual, 2))
        cached     = st.session_state.get("hdr10_cache", {})

        # Feature extraction is cached separately from render (beta changes don't re-extract)
        feat_key = (hdr_path, hdr_frame, int(hdr_nits), _hdr10_ckpt)
        feat_cache = st.session_state.get("hdr10_feat_cache", {})

        if cached.get("key") != render_key:
            _hlog("RENDER_START", f"render_key={render_key}")
            with st.spinner("Extracting features + rendering (5-10 sec)..."):
                # Only re-extract if path/frame/nits/checkpoint changed (not just beta)
                if feat_cache.get("key") != feat_key:
                    _hlog("DAEMON_CALL", f"Calling daemon: frame={hdr_frame}  pts={pts_hdr:.3f}s  feat_target=100 nits")
                    try:
                        import concurrent.futures as _cf
                        daemon = _get_daemon()
                        _hlog("DAEMON_READY", f"Daemon pid={daemon.process.pid if daemon.process else 'none'}")
                        with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                            _fut = _pool.submit(daemon.extract_frame, hdr_path, pts_hdr, 100.0)
                            try:
                                features = _fut.result(timeout=20)
                            except _cf.TimeoutError:
                                features = None
                                _hlog("DAEMON_TIMEOUT", "extract_frame timed out after 20s")
                                st.error("Feature extraction timed out (>20s) — try a nearby frame.")
                    except Exception as e:
                        features = None
                        _hlog("DAEMON_ERROR", str(e))
                        st.error(f"Feature extraction failed: {e}")

                    if features is not None:
                        _hlog("DAEMON_RESULT", f"maxscl={features.get('maxscl',0):.4f}  "
                              f"avg={features.get('average_maxrgb',0):.4f}  "
                              f"zone_r1c1={features.get('zone_mean_3x3_r1_c1',0):.4f}  "
                              f"n_feats={len(features)}")
                    else:
                        _hlog("DAEMON_RESULT", "features=None (extraction failed)")
                    st.session_state["hdr10_feat_cache"] = {"key": feat_key, "features": features}
                else:
                    features = feat_cache.get("features")
                    _hlog("FEAT_CACHE_HIT", f"Reusing cached features: maxscl={features.get('maxscl',0):.4f}" if features else "FEAT_CACHE_HIT: None")

                result = {"key": render_key, "features": None,
                          "img_spline": None, "img_ml": None,
                          "ml_curve": None, "spline_curve": None,
                          "l1_max": 0.5, "l1_avg": 0.2}

                if features is not None:
                    _l1max = float(features.get("maxscl", 0.5))
                    _l1avg = float(features.get("average_maxrgb", 0.3))
                    _spline_curve, _spline_k = _compute_spline_curve(_l1max, _l1avg, hdr_nits)

                    if _is_xgb(_hdr10_ckpt):
                        # XGBoost delta inference — no PyTorch, no normalisation
                        _pkl_path = _hdr10_ckpt[len("xgb://"):]
                        _xgb_model, _feat_cols, _knot_idx = _load_xgb_model(_pkl_path)
                        sys.path.insert(0, os.path.dirname(__file__))
                        ml_curve = _infer_xgb_hdr10(
                            features, _feat_cols, _xgb_model, _knot_idx,
                            _spline_k, _l1max, _l1avg, hdr_nits)
                        l1_max, l1_avg, spline_curve, spline_k = _l1max, _l1avg, _spline_curve, _spline_k
                    else:
                        _model, _feat_cols, _feat_mean, _feat_std, _device = _load_hdr10_model(_hdr10_ckpt)
                        ml_curve, l1_max, l1_avg, spline_curve, spline_k = _infer_mcp_hdr10(
                            features, _feat_cols, _feat_mean, _feat_std, _model, _device, hdr_nits)

                    _hlog("ML_INFERENCE", f"l1_max={l1_max:.4f}  l1_avg={l1_avg:.4f}  "
                          f"ml_x01={np.interp(0.1,np.linspace(0,1,256),ml_curve):.4f}  "
                          f"ml_x02={np.interp(0.2,np.linspace(0,1,256),ml_curve):.4f}  "
                          f"spl_x01={np.interp(0.1,np.linspace(0,1,256),spline_curve):.4f}  "
                          f"spl_x02={np.interp(0.2,np.linspace(0,1,256),spline_curve):.4f}")

                    # ── Compute auto-beta from raw ML curve (before any correction) ──
                    _xs_ab = np.linspace(0, 1, 256, dtype=np.float32)
                    _sz    = _xs_ab < 0.2
                    _raw_gap   = float(np.mean(np.maximum(spline_curve[_sz] - ml_curve[_sz], 0)))
                    _scaled    = float(_raw_gap / 0.05)
                    beta_auto_computed = float(min(_scaled, 0.5))
                    _hlog("BETA_CALC", f"shadow_gap={_raw_gap:.5f}  scaled={_scaled:.3f}  "
                          f"beta_auto={beta_auto_computed:.3f}  use_auto={use_auto}  "
                          f"manual_beta={shadow_beta_manual:.2f}")
                    # Effective beta: auto mode uses this, manual uses user slider
                    effective_beta = beta_auto_computed if use_auto else shadow_beta_manual

                    # Expansion scene guard: if content peak < display ceiling (target_yn > 1.0),
                    # the ML's conservative curve is intentional (colorist low-key style).
                    # Pulling toward the spline's aggressive mathematical expansion destroys
                    # the cinematic lighting. Force beta=0 for expansion scenes.
                    target_pq_val = _nits_to_pq(hdr_nits)
                    target_yn_val = target_pq_val / max(l1_max, 1e-6)
                    if target_yn_val > 1.0 and use_auto:
                        _hlog("BETA_EXPANSION", f"Expansion scene target_yn={target_yn_val:.3f}>1.0 — forcing beta=0 (ML style preserved)")
                        effective_beta = 0.0

                    _hlog("BETA_APPLIED", f"effective_beta={effective_beta:.3f}  target_yn={target_yn_val:.3f}")

                    # ── Shadow lift ───────────────────────────────────────────
                    if effective_beta > 0.0:
                        shadow_mask = np.exp(-_xs_ab / 0.15)
                        shadow_gap  = np.maximum(spline_curve - ml_curve, 0.0)
                        ml_curve    = ml_curve + effective_beta * shadow_mask * shadow_gap
                        ml_curve    = np.maximum.accumulate(ml_curve)
                    # ─────────────────────────────────────────────────────────

                    lut_tmp = tempfile.NamedTemporaryFile(suffix=".lut", delete=False, mode='w')
                    lut_tmp.close()
                    _write_ml_lut_from_curve(ml_curve, lut_tmp.name, l1_max, hdr_nits)

                    img_spline = _render_hdr10(hdr_path, pts_hdr, "spline", hdr_nits,
                                                l1_max=l1_max, l1_avg=l1_avg)
                    img_ml     = _render_hdr10(hdr_path, pts_hdr, "ml-lut", hdr_nits,
                                                lut_path=lut_tmp.name, l1_max=l1_max, l1_avg=l1_avg)
                    os.unlink(lut_tmp.name)

                    spline_ok = img_spline is not None
                    ml_ok = img_ml is not None
                    _hlog("RENDER_DONE", f"spline={'OK mean='+f'{img_spline.mean():.1f}' if spline_ok else 'FAILED'}  "
                          f"ml={'OK mean='+f'{img_ml.mean():.1f}' if ml_ok else 'FAILED'}  "
                          f"beta_stored={beta_auto_computed:.3f}")
                    result.update({"features": features, "img_spline": img_spline,
                                    "img_ml": img_ml, "ml_curve": ml_curve,
                                    "spline_curve": spline_curve, "l1_max": l1_max,
                                    "l1_avg": l1_avg,
                                    "beta_auto": beta_auto_computed,
                                    "beta_used": effective_beta})

            st.session_state["hdr10_cache"] = result
            cached = result
            st.rerun()  # refresh display so Auto β metric shows current frame's value

        if cached.get("img_spline") is None and cached.get("img_ml") is None and cached.get("key") == render_key:
            st.warning("Renders failed — this PTS may be undecodable in this file. Try a nearby frame number.")
        elif cached.get("img_spline") is not None or cached.get("img_ml") is not None:
            l1_max = cached["l1_max"]; l1_avg = cached["l1_avg"]
            st.caption(f"maxscl={l1_max:.4f}  l1_avg={l1_avg:.4f}")
            hrc1, hrc2 = st.columns(2)
            if cached["img_spline"] is not None:
                hrc1.image(cached["img_spline"], caption=f"Spline ({hdr_nits} nits)", use_container_width=True)
            else:
                hrc1.warning("Spline render failed")
            if cached["img_ml"] is not None:
                hrc2.image(cached["img_ml"], caption=f"ML {run_name[:20]} ({hdr_nits} nits)", use_container_width=True)
            else:
                hrc2.warning("ML render failed")

            if cached["ml_curve"] is not None and cached["spline_curve"] is not None:
                ml_c = cached["ml_curve"]; spl_c = cached["spline_curve"]
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(x=xs, y=spl_c, name="Spline (libplacebo)",
                                            line=dict(color="#3498db", width=2, dash="dash")))
                fig_h.add_trace(go.Scatter(x=xs, y=ml_c, name=f"ML ({run_name[:20]})",
                                            line=dict(color="#e74c3c", width=2)))
                fig_h.add_trace(go.Scatter(x=xs, y=xs, name="Identity",
                                            line=dict(color="#999", width=1, dash="dot")))
                fig_h.add_vline(x=l1_max, line_dash="dash", line_color="#aaa",
                                 annotation_text=f"l1_max={l1_max:.3f}")
                mae_hdr = float(np.abs(ml_c - spl_c).mean())
                fig_h.update_layout(
                    title=f"Tone curves (HDR10)  |  ML vs Spline MAE={mae_hdr:.5f}",
                    xaxis_title="Input PQ", yaxis_title="Output PQ",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                    legend=dict(x=0.02, y=0.98), height=400)
                st.plotly_chart(fig_h, use_container_width=True)

    elif hdr_path:
        st.error(f"File not found: {hdr_path}")
    else:
        st.info("Enter an HDR10 MKV path above. Renders automatically when frame or nits change.")

    # ── Trace log ──
    hdr_log = st.session_state.get("hdr10_log", [])
    if hdr_log:
        with st.expander(f"Pipeline trace log ({len(hdr_log)} entries)", expanded=True):
            st.code("\n".join(hdr_log), language=None)
            if st.button("Clear log", key="hdr10_clear_log"):
                st.session_state["hdr10_log"] = []
                if Path(_HDR_LOG_FILE).exists():
                    open(_HDR_LOG_FILE, 'w').close()
                st.rerun()
