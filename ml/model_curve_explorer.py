"""
model_curve_explorer.py — ML vs spline vs gold curve diagnostic viewer.

Shows best/worst performing scenes per cell type.
Select a cell → sorted dropdown of best/worst scenes → see three curves.

Run:
    streamlit run ml/model_curve_explorer.py
"""

import sys, os, json, subprocess, tempfile
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
    "Run 14 (production)": r"F:\DTMModelData\ckpt_mcp_run14_best.pt",
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

# Scene selector
st.subheader(f"{cell}  —  {title_str}")

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
col1, col2 = st.columns([2, 1])

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
st.divider()
st.subheader("Rendered frames")

video_path = find_video(row["title_key"], row["episode"])
if video_path:
    pts_time = float(row["pts_time"])
    l1_max   = float(results.loc[sel_idx, "l1_max_pq"]) / 4095.0 if "l1_max_pq" in results.columns else 0.0
    l1_avg   = float(results.loc[sel_idx, "l1_avg_pq"]) / 4095.0 if "l1_avg_pq" in results.columns else 0.0
    ml_curve = ml_arr[sel_idx]   # [256] float32

    rc1, rc2 = st.columns([3, 1])
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
    st.info(f"Video not found on disk for {row['title_key']} — copy to D:\\Jdownloader\\Dataset to enable rendering")

# ---------------------------------------------------------------------------
# Cell summary table
# ---------------------------------------------------------------------------
st.divider()
st.subheader("All cells — summary")
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
st.dataframe(disp, use_container_width=True)
st.caption("MT-MAE = midtone-weighted MAE (3x weight on PQ [0.2-0.7]). MT Win% = % scenes where ML beats spline on midtone shape. If MT Win% < Win%, ML is winning via endpoint match not midtone shape.")

# ---------------------------------------------------------------------------
# Title × Cell distribution
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Val set — title x cell distribution (scene counts)")

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
st.dataframe(
    pivot.style.background_gradient(cmap="Blues", axis=0, subset=cell_cols_sorted),
    use_container_width=True,
    height=min(60 + len(pivot) * 35, 700),
)
