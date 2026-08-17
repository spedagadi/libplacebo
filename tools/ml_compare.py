"""
ml_compare.py
=============
Interactive DV polynomial comparison tool.

For a given video + frame (or PTS), renders the frame in four modes:
  gold     — DV RPU polynomial (ground truth)
  ml       — ML model predicted polynomial (ckpt11_best.pt)
  spline   — libplacebo pl_tone_map_spline
  st2094   — ST2094-40 tone mapping

Also plots the four tone curves on the same axes.

Usage:
    python3 tools/ml_compare.py --video <path.mkv> --frame <N> [--nits 143|1030|1669]
    python3 tools/ml_compare.py --video <path.mkv> --pts <seconds> [--nits 143]

    Interactive mode (loops asking for frames):
    python3 tools/ml_compare.py --video <path.mkv> --interactive [--nits 143]

Output:
    F:/DTMModelData/compare/<basename>_frame<N>_<nits>nits_curves.png
    F:/DTMModelData/compare/<basename>_frame<N>_<nits>nits_frames.png

Requirements:
    - build/tools/dv_render.exe  (built from tools/dv_render.c)
    - F:/DTMModelData/ckpt11_best.pt  (or --checkpoint <path>)
    - F:/DTMModelData/val/val_dataset.csv + train/train_dataset.csv
    - matplotlib, torch, pandas, numpy
"""

import sys, os, argparse, subprocess, struct, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from dv_coef_model import (load_data, row_to_target, target_to_rpu, eval_rpu,
                            write_rpu_lut)
from dv_mlp_model import DVPolyMLP

# ── Paths ────────────────────────────────────────────────────────────────────
DV_RENDER   = Path('c:/Code/libplacebo/build/tools/dv_render.exe')
BASELINE    = Path('c:/Code/libplacebo/build/tools/libplacebo_baseline_eval.exe')
TRAIN_CSV   = Path('F:/DTMModelData/train/train_dataset.csv')
VAL_CSV     = Path('F:/DTMModelData/val/val_dataset.csv')
CHECKPOINT  = Path('F:/DTMModelData/ckpt11_best.pt')
OUT_DIR     = Path('F:/DTMModelData/compare')
RENDER_W    = 1920
RENDER_H    = 1080

# ── DLL search path (libplacebo + MSYS2 runtime) ────────────────────────────
os.environ['PATH'] = (
    r'C:\Code\libplacebo\build\src;'
    r'C:\msys64\ucrt64\bin;'
    + os.environ.get('PATH', '')
)


# ── Load model once ──────────────────────────────────────────────────────────
_model_cache = None
_feat_cols_cache = None

def load_model(checkpoint=CHECKPOINT):
    global _model_cache, _feat_cols_cache
    if _model_cache is not None:
        return _model_cache, _feat_cols_cache
    ckpt = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    feat_cols = ckpt['feat_cols']
    model = DVPolyMLP(pixel_dim=len(feat_cols), dropout=0.3, use_tier_embed=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    _model_cache = model
    _feat_cols_cache = feat_cols
    print(f"  Model loaded: {checkpoint.name}  ep={ckpt.get('epoch')}  "
          f"val_loss={ckpt.get('val_loss', '?'):.5f}", flush=True)
    return model, feat_cols


# ── Dataset lookup ───────────────────────────────────────────────────────────
_dataset_cache = None

def load_dataset():
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache
    print("  Loading datasets...", flush=True)
    dfs = []
    for path in [TRAIN_CSV, VAL_CSV]:
        if path.exists():
            dfs.append(load_data(str(path)))
    _dataset_cache = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"  {len(_dataset_cache):,} scenes loaded", flush=True)
    return _dataset_cache


def find_scene_for_video_frame(video_path, frame_idx):
    """Find the dataset row whose episode matches the video and contains frame_idx."""
    df = load_dataset()
    if df.empty:
        return None
    # Match by episode stem
    stem = Path(video_path).stem
    # Try matching episode column (partial match)
    mask = df['episode'].str.contains(stem[:30], case=False, na=False)
    if mask.sum() == 0:
        # Fallback: match by title hint from path
        for word in stem.split('.')[:4]:
            if len(word) > 4:
                mask = df['episode'].str.contains(word, case=False, na=False)
                if mask.sum() > 0:
                    break
    if mask.sum() == 0:
        print(f"  WARNING: could not find episode '{stem}' in dataset", flush=True)
        return None
    ep_df = df[mask]
    # Find scene containing this frame
    row_mask = (ep_df['start_frame'] <= frame_idx) & (ep_df['end_frame'] >= frame_idx)
    if row_mask.sum() == 0:
        # Use nearest scene
        dists = (ep_df['rep_frame'] - frame_idx).abs()
        row = ep_df.iloc[dists.argmin()]
    else:
        row = ep_df[row_mask].iloc[0]
    return row


def frame_to_pts(video_path: str, frame_idx: int) -> float:
    """Convert frame index to PTS using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=10
        )
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        return frame_idx / fps
    except Exception:
        return frame_idx / 24.0  # fallback 24fps


# ── ML inference ─────────────────────────────────────────────────────────────
def run_ml_inference(row, checkpoint=CHECKPOINT):
    """
    Run ML model on a dataset row. Returns (target_42, rpu_lut_path).
    Writes RPU_POLY_1D to a temp file for dv_render.
    """
    model, feat_cols = load_model(checkpoint)
    feat_vals = [float(row.get(c, 0.0)) for c in feat_cols]
    feats = torch.tensor([feat_vals], dtype=torch.float32)
    tiers = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        pred_42, _ = model(feats, tiers)
    t = pred_42[0].numpy()

    # Write RPU_POLY_1D for dv_render
    lut_path = str(OUT_DIR / '_ml_poly.rpu_poly')
    write_rpu_lut(t, lut_path)
    return t, lut_path


# ── dv_render call ────────────────────────────────────────────────────────────
def render_frame(video_path: str, pts: float, mode: str, out_nits: float,
                 lut_path=None,
                 l1_max=None,
                 l1_avg=None):
    """
    Call dv_render and capture raw RGB8 output.
    Returns H×W×3 uint8 array or None on failure.
    """
    if not DV_RENDER.exists():
        print(f"  ERROR: {DV_RENDER} not found", flush=True)
        return None

    cmd = [
        str(DV_RENDER),
        '--input', video_path,
        '--pts', f'{pts:.6f}',
        '--mode', mode,
        '--width', str(RENDER_W),
        '--height', str(RENDER_H),
        '--out-nits', str(out_nits),
    ]
    if lut_path:
        cmd += ['--lut', lut_path]
    if l1_max is not None:
        cmd += ['--l1-max', f'{l1_max:.6f}']
    if l1_avg is not None:
        cmd += ['--l1-avg', f'{l1_avg:.6f}']

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        expected = RENDER_W * RENDER_H * 3
        if len(result.stdout) != expected:
            print(f"  WARNING: {mode} render returned {len(result.stdout)} bytes "
                  f"(expected {expected})", flush=True)
            if result.stderr:
                print(f"  stderr: {result.stderr.decode()[:200]}", flush=True)
            return None
        arr = np.frombuffer(result.stdout, dtype=np.uint8)
        return arr.reshape(RENDER_H, RENDER_W, 3)
    except subprocess.TimeoutExpired:
        print(f"  ERROR: {mode} render timed out", flush=True)
        return None
    except Exception as e:
        print(f"  ERROR: {mode} render failed: {e}", flush=True)
        return None


# ── Curve computation ─────────────────────────────────────────────────────────
def compute_libplacebo_curve(maxscl, l1_avg, out_nits, n_pts=256):
    """Call libplacebo_baseline_eval for a single scene."""
    if not BASELINE.exists():
        return None
    inp = f"scene_id,maxscl,l1_avg_pq,target_nits\n0,{maxscl:.6f},{l1_avg:.6f},{out_nits}\n"
    result = subprocess.run(
        [str(BASELINE)], input=inp, capture_output=True, text=True, timeout=5
    )
    lines = result.stdout.strip().split('\n')
    if len(lines) < 2:
        return None
    vals = [float(x) for x in lines[1].split(',')[1:]]
    # Resample to n_pts
    orig = np.array(vals)
    return np.interp(np.linspace(0, 1, n_pts), np.linspace(0, len(orig)-1, len(orig)), orig)


def compute_curves(row: pd.Series, ml_target: np.ndarray,
                   out_nits: float, n_pts: int = 256):
    """
    Returns dict of curve_name -> y values over xs=linspace(0, maxscl, n_pts).
    """
    maxscl   = float(row.get('maxscl', 0.5))
    l1_avg   = float(row.get('l1_avg_pq', 0.3))
    xs       = np.linspace(0.0, maxscl, n_pts)

    curves = {}
    curves['identity'] = xs.copy()

    # Gold RPU polynomial
    t = row_to_target(row)
    if t is not None:
        rpu = target_to_rpu(t)
        ys = eval_rpu(rpu, xs)
        if not np.any(np.isnan(ys)):
            curves['gold'] = ys

    # ML predicted polynomial
    if ml_target is not None:
        try:
            rpu_ml = target_to_rpu(ml_target)
            ys_ml  = eval_rpu(rpu_ml, xs)
            if not np.any(np.isnan(ys_ml)):
                curves['ml'] = ys_ml
        except Exception:
            pass

    # libplacebo spline
    lp = compute_libplacebo_curve(maxscl, l1_avg, out_nits, n_pts)
    if lp is not None:
        curves['libplacebo_spline'] = lp

    return xs, curves


# ── Plotting ──────────────────────────────────────────────────────────────────
CURVE_STYLE = {
    'gold':             dict(color='#FFD700', lw=2.5, ls='-',  label='Gold RPU (DV)'),
    'ml':               dict(color='#00BFFF', lw=2.5, ls='-',  label='ML model'),
    'libplacebo_spline':dict(color='#FF6347', lw=1.8, ls='--', label='libplacebo spline'),
    'identity':         dict(color='#888888', lw=1.2, ls=':',  label='Identity (y=x)'),
}

MODE_LABEL = {
    'gold':    'Gold RPU',
    'ml':      'ML model',
    'spline':  'libplacebo spline',
    'st2094-40': 'ST2094-40',
}

def make_figure(video_path, frame_idx, out_nits, xs, curves, frames_dict, row):
    """Create and save the comparison figure."""
    basename = Path(video_path).stem[:30]
    title    = f"{basename}  frame={frame_idx}  target={out_nits:.0f} nits"

    # Scene metadata for subplot title
    maxscl  = float(row.get('maxscl', 0.5))
    gold_dev = float(np.mean(curves.get('gold', xs) - xs)) if 'gold' in curves else 0
    ml_dev   = float(np.mean(curves.get('ml', xs) - xs))   if 'ml'   in curves else 0

    # ── Figure 1: Rendered frames ──────────────────────────────────────────
    n_frames  = len(frames_dict)
    if n_frames > 0:
        fig_f, axes_f = plt.subplots(1, n_frames, figsize=(6*n_frames, 4))
        if n_frames == 1:
            axes_f = [axes_f]
        fig_f.suptitle(title, fontsize=11, y=1.01)
        for ax, (mode, img) in zip(axes_f, frames_dict.items()):
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'render failed', ha='center', va='center',
                        transform=ax.transAxes, color='red')
                ax.set_facecolor('#222')
            ax.set_title(MODE_LABEL.get(mode, mode), fontsize=10)
            ax.axis('off')
        fig_f.tight_layout()
        frames_path = OUT_DIR / f'{basename}_frame{frame_idx}_{int(out_nits)}nits_frames.png'
        fig_f.savefig(str(frames_path), dpi=150, bbox_inches='tight')
        plt.close(fig_f)
        print(f"  Frames saved: {frames_path}", flush=True)

    # ── Figure 2: Tone curves ──────────────────────────────────────────────
    fig_c, ax = plt.subplots(figsize=(8, 6))
    for name, ys in curves.items():
        style = CURVE_STYLE.get(name, dict(lw=1.5, label=name))
        ax.plot(xs, ys, **style)
    ax.set_xlabel('Input ICtCp-I signal (PQ, normalized)')
    ax.set_ylabel('Output ICtCp-I signal')
    ax.set_title(
        f'{title}\n'
        f'maxscl={maxscl:.3f}  '
        f'gold_dev={gold_dev:+.4f}  '
        f'ml_dev={ml_dev:+.4f}  '
        f'gap={ml_dev-gold_dev:+.4f}'
    )
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, maxscl * 1.05)
    fig_c.tight_layout()
    curves_path = OUT_DIR / f'{basename}_frame{frame_idx}_{int(out_nits)}nits_curves.png'
    fig_c.savefig(str(curves_path), dpi=150, bbox_inches='tight')
    plt.close(fig_c)
    print(f"  Curves saved: {curves_path}", flush=True)

    return frames_path, curves_path


# ── Main comparison logic ─────────────────────────────────────────────────────
def compare(video_path: str, frame_idx: int, out_nits: float,
            checkpoint=CHECKPOINT, skip_render=False):
    """Run full comparison for one frame."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}", flush=True)
    print(f"Video: {Path(video_path).name}  frame={frame_idx}  nits={out_nits:.0f}", flush=True)

    # 1. Find scene in dataset
    print("  Looking up scene in dataset...", flush=True)
    row = find_scene_for_video_frame(video_path, frame_idx)
    if row is None:
        print("  WARNING: scene not found — will use inference-only mode", flush=True)
    else:
        print(f"  Scene: {row.get('title','?')}  "
              f"frames [{row.get('start_frame','?')}-{row.get('end_frame','?')}]  "
              f"maxscl={row.get('maxscl', '?'):.4f}  "
              f"l1_max_pq={row.get('l1_max_pq','?'):.4f}", flush=True)

    # 2. Compute PTS
    pts = frame_to_pts(video_path, frame_idx)
    print(f"  PTS: {pts:.3f}s  (frame {frame_idx})", flush=True)

    # 3. ML inference
    print("  Running ML inference...", flush=True)
    ml_target, lut_path = None, None
    if row is not None:
        try:
            ml_target, lut_path = run_ml_inference(row, checkpoint)
            rpu_ml = target_to_rpu(ml_target)
            print(f"  ML poly: {rpu_ml['n_segs']} segs", flush=True)
        except Exception as e:
            print(f"  WARNING: ML inference failed: {e}", flush=True)

    # 4. Render frames
    frames_dict = {}
    if not skip_render:
        l1_max = float(row.get('l1_max_pq', 0.5)) if row is not None else None
        l1_avg = float(row.get('l1_avg_pq', 0.3)) if row is not None else None

        modes = [('gold', None), ('ml', lut_path), ('spline', None), ('st2094-40', None)]
        for mode, lut in modes:
            if mode == 'ml' and lut is None:
                print(f"  Skipping ML render (no polynomial)", flush=True)
                continue
            print(f"  Rendering {mode}...", flush=True)
            img = render_frame(video_path, pts, mode, out_nits, lut, l1_max, l1_avg)
            frames_dict[mode] = img
            if img is None:
                print(f"  WARNING: {mode} render returned no data", flush=True)

    # 5. Compute curves
    xs, curves = compute_curves(row, ml_target, out_nits) if row is not None else (
        np.linspace(0, 0.5, 256), {'identity': np.linspace(0, 0.5, 256)}
    )

    # Print curve stats
    if 'gold' in curves and 'ml' in curves:
        gold_rms = float(np.sqrt(np.mean((curves['gold'] - xs)**2)))
        ml_rms   = float(np.sqrt(np.mean((curves['ml']   - xs)**2)))
        err_rms  = float(np.sqrt(np.mean((curves['ml'] - curves['gold'])**2)))
        id_rms   = gold_rms
        print(f"  Gold vs identity: RMS={id_rms:.5f}", flush=True)
        print(f"  ML vs identity:   RMS={ml_rms:.5f}", flush=True)
        print(f"  ML vs gold:       RMS={err_rms:.5f}  "
              f"({'BETTER' if err_rms < id_rms else 'worse'} than identity)", flush=True)

    # 6. Save figure
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if row is not None:
        frames_path, curves_path = make_figure(
            video_path, frame_idx, out_nits, xs, curves, frames_dict, row
        )
        return frames_path, curves_path

    return None, None


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='DV polynomial comparison tool')
    ap.add_argument('--video',       required=True, help='Input video file')
    ap.add_argument('--frame',       type=int,   default=None, help='Frame index')
    ap.add_argument('--pts',         type=float, default=None, help='PTS in seconds (alternative to --frame)')
    ap.add_argument('--nits',        type=float, default=143,  help='Target display nits (default 143 = SDR)')
    ap.add_argument('--checkpoint',  default=str(CHECKPOINT),  help='Model checkpoint path')
    ap.add_argument('--no-render',   action='store_true',      help='Skip dv_render calls (curves only)')
    ap.add_argument('--interactive', action='store_true',      help='Loop asking for frame numbers')
    args = ap.parse_args()

    # Pre-load model and dataset
    load_model(Path(args.checkpoint))
    load_dataset()

    if args.interactive:
        print(f"\nInteractive mode — video: {Path(args.video).name}  nits={args.nits}", flush=True)
        print("Enter frame number (or 'q' to quit):", flush=True)
        while True:
            try:
                inp = input('\nframe> ').strip()
                if inp.lower() in ('q', 'quit', 'exit'):
                    break
                if not inp.isdigit():
                    print("  Enter a frame number or 'q'")
                    continue
                compare(args.video, int(inp), args.nits,
                        Path(args.checkpoint), args.no_render)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        frame = args.frame
        if frame is None and args.pts is not None:
            # Convert PTS to approximate frame
            frame = int(args.pts * 24)
        if frame is None:
            ap.error('Provide --frame or --pts')
        compare(args.video, frame, args.nits, Path(args.checkpoint), args.no_render)
