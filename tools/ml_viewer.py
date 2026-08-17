"""
ml_viewer.py — DV Polynomial Comparison Viewer (Streamlit)

Two modes:
  DV Val Set  — compare Gold RPU / ML model / libplacebo spline on val set
  HDR10       — load any HDR10 video, compare libplacebo spline vs ML model

Run: streamlit run tools/ml_viewer.py
"""
import sys, os, subprocess, warnings, glob, re
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))
os.environ['PATH'] = (r'C:\Code\libplacebo\build\src;C:\msys64\ucrt64\bin;'
                      + os.environ.get('PATH', ''))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from dv_coef_model import (load_data, row_to_target, target_to_rpu,
                            eval_rpu, write_rpu_lut, _sanitise_rpu)
from dv_mlp_model import DVPolyMLP

# ── Config ──────────────────────────────────────────────────────────────────
DV_RENDER    = Path('c:/Code/libplacebo/build/tools/dv_render.exe')
BASELINE     = Path('c:/Code/libplacebo/build/tools/libplacebo_baseline_eval.exe')
DAEMON_BIN   = Path('c:/Code/libplacebo/build/tools/pl_extract_features_daemon.exe')
TRAIN_CSV    = Path('F:/DTMModelData/train/train_dataset.csv')
VAL_CSV      = Path('F:/DTMModelData/val/val_dataset.csv')
CHECKPOINT   = Path('F:/DTMModelData/ckpt11_best.pt')
DATASET_ROOT = Path('G:/Dataset')
TMP_DIR      = Path('F:/DTMModelData/compare')
TMP_DIR.mkdir(parents=True, exist_ok=True)
RENDER_W, RENDER_H = 1280, 720

VAL_TITLES   = {'andor', 'euphoria', 'prehistoric', 'our', 'wondla'}
DISPLAY_NITS = {'SDR 143 nits': 143.0, 'HDR 1030 nits': 1030.0,
                'HDR 1669 nits': 1669.0}
CURVE_STYLE  = {
    'gold':              dict(color='#FFD700', lw=2.5, ls='-',  label='Gold RPU'),
    'ml':                dict(color='#00BFFF', lw=2.5, ls='-',  label='ML model'),
    'libplacebo_spline': dict(color='#FF6347', lw=1.8, ls='--', label='libplacebo spline'),
    'identity':          dict(color='#888888', lw=1.2, ls=':',  label='Identity'),
}
MODE_LABEL = {'gold': 'Gold RPU (DV)', 'ml': 'ML model (DV)',
              'spline': 'libplacebo spline', 'st2094-40': 'ST2094-40',
              'ml-lut': 'ML model (HDR10)'}

# ── Helpers ──────────────────────────────────────────────────────────────────
def nits_to_pq(nits):
    L = max(float(nits), 1e-6) / 10000.0
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lm1 = L ** m1
    return ((c1 + c2 * Lm1) / (1 + c3 * Lm1)) ** m2

def write_ml_lut(ml_target, path, n_pts=512, maxscl=None, out_nits=None):
    """
    Write ML polynomial as x y LUT for --mode ml-lut (HDR10 mode).

    The ML model was trained on DV content — its curve output level reflects
    the DV colorist's compressed targets. For HDR10, we normalise the curve so
    that output at the content peak (maxscl) maps to the display peak
    (nits_to_pq(out_nits)), preserving curve SHAPE while correcting the gain.
    """
    from scipy.interpolate import UnivariateSpline
    # xs_norm spans [0,1] where 1.0 = scene peak (maxscl in absolute ICtCp).
    # ml_tone_map: output = yn * input_max where input_max = maxscl.
    # So ys[-1] must equal target_pq/maxscl so output@peak = target_pq.
    xs_norm = np.linspace(0.0, 1.0, n_pts)
    try:
        # Evaluate polynomial at xs_abs = [0, maxscl] so ys[-1] = poly(maxscl).
        # This keeps xs_norm and xs_abs in the same coordinate system:
        # xs_norm[i] * maxscl = xs_abs[i] — the tone map's xn = x/maxscl.
        abs_max = maxscl if (maxscl is not None and maxscl > 0.05) else 1.0
        xs_abs  = np.linspace(0.0, abs_max, 1024)
        ys_poly = eval_rpu(target_to_rpu(ml_target), xs_abs)
        ys_poly = np.maximum.accumulate(np.clip(ys_poly, 0.0, 1.5))

        # Smooth monotone spline on normalised [0,1] control points
        spline = UnivariateSpline(xs_abs / abs_max, ys_poly, k=3,
                                  s=len(xs_abs) * 0.0001, ext=3)
        ys = np.maximum.accumulate(np.clip(spline(xs_norm), 0.0, 1.5))
    except Exception:
        ys = xs_norm.copy()

    # Normalise: ys[-1] = poly(maxscl); scale so ys[-1] = target_pq/maxscl = target_yn.
    # output@peak = ys[-1] * maxscl = target_pq ✓
    if maxscl is not None and out_nits is not None and maxscl > 0.05:
        target_pq = nits_to_pq(out_nits)
        target_yn = target_pq / maxscl
        peak_out  = float(ys[-1])
        if peak_out > 0.01:
            ys = np.clip(ys * (target_yn / peak_out), 0.0, 1.0)

    xs = xs_norm  # LUT written with normalised xs [0,1]

    with open(path, 'w') as f:
        for x, y in zip(xs, ys):
            f.write(f'{x:.6f} {y:.6f}\n')

def frame_to_pts(video_path, frame_idx):
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=8)
        fps_str = r.stdout.strip().split(',')[0]  # strip trailing comma
        fps = (float(fps_str.split('/')[0]) / float(fps_str.split('/')[1])
               if '/' in fps_str else float(fps_str))
        return frame_idx / fps
    except Exception:
        return frame_idx / 24.0

def get_frame_count(video_path):
    """Estimate total frames from duration × fps — no full packet scan."""
    try:
        r_dur = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=8)
        r_fps = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=8)
        dur = float(r_dur.stdout.strip())
        fps_s = r_fps.stdout.strip().split(',')[0]  # strip trailing comma
        fps = (float(fps_s.split('/')[0]) / float(fps_s.split('/')[1])
               if '/' in fps_s else float(fps_s))
        return max(1, int(dur * fps))
    except Exception:
        return 100000

def render_frame(video_path, pts, mode, out_nits,
                 lut_path=None, l1_max=None, l1_avg=None,
                 top_bar=0.0, bot_bar=0.0):
    if not DV_RENDER.exists():
        return None
    cmd = [str(DV_RENDER), '--input', video_path,
           '--pts', f'{pts:.6f}', '--mode', mode,
           '--width', str(RENDER_W), '--height', str(RENDER_H),
           '--out-nits', str(out_nits)]
    if lut_path:           cmd += ['--lut', lut_path]
    if l1_max is not None: cmd += ['--l1-max', f'{l1_max:.6f}']
    if l1_avg is not None: cmd += ['--l1-avg', f'{l1_avg:.6f}']
    # Zero out bar rows so LUT cannot lift them above true black
    if top_bar > 0.01:     cmd += ['--top-bar-norm', f'{top_bar:.6f}']
    if bot_bar > 0.01:     cmd += ['--bot-bar-norm', f'{bot_bar:.6f}']
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=25)
        if len(result.stdout) != RENDER_W * RENDER_H * 3:
            return None
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape(RENDER_H, RENDER_W, 3)
    except Exception:
        return None

def make_curve_figure(xs, curves, maxscl, title=''):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1d24')
    for name, ys in curves.items():
        s = CURVE_STYLE.get(name, dict(color='white', lw=1.5, ls='-', label=name))
        ax.plot(xs, ys, **s)
    ax.axvline(maxscl, color='#666', lw=0.8, ls=':', alpha=0.7)
    ax.text(maxscl + 0.01, 0.03, f'{maxscl:.3f}', color='#888', fontsize=6)
    ax.set_xlabel('Input PQ (ICtCp-I)', color='#ccc', fontsize=8)
    ax.set_ylabel('Output PQ', color='#ccc', fontsize=8)
    ax.set_xlim(0, 1.0);  ax.set_ylim(-0.02, 1.02)
    ax.tick_params(colors='#aaa', labelsize=7)
    ax.spines[:].set_color('#555')
    ax.grid(True, alpha=0.2, color='#555')
    ax.legend(loc='upper left', fontsize=7,
              facecolor='#1a1d24', edgecolor='#555', labelcolor='#ddd')
    if title:
        ax.set_title(title, color='#ccc', fontsize=7)
    fig.tight_layout(pad=0.5)
    return fig

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading model...')
def load_model():
    ckpt = torch.load(str(CHECKPOINT), map_location='cpu', weights_only=False)
    feat_cols = ckpt['feat_cols']
    model = DVPolyMLP(pixel_dim=len(feat_cols), dropout=0.3, use_tier_embed=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, feat_cols, ckpt.get('epoch'), ckpt.get('val_loss')

@st.cache_resource(show_spinner='Loading dataset...')
def load_full_dataset():
    dfs = [load_data(str(p)) for p in [TRAIN_CSV, VAL_CSV] if p.exists()]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

@st.cache_data(show_spinner='Scanning video files...')
def build_mkv_index():
    idx = {}
    for mkv in glob.glob(str(DATASET_ROOT / '**' / '*.mkv'), recursive=True):
        idx[Path(mkv).stem] = mkv
    return idx

def episode_to_mkv(episode_col, mkv_index):
    parts = episode_col.split('_')
    for i, p in enumerate(parts):
        if p and p[0].isupper():
            stem = '_'.join(parts[i:])
            if stem in mkv_index:
                return mkv_index[stem]
    return None

@st.cache_data(show_spinner='Building episode list...')
def get_val_episodes():
    df = load_full_dataset()
    mkv_idx = build_mkv_index()
    episodes, seen = [], set()
    for _, row in df[df['title'].isin(VAL_TITLES)].iterrows():
        ep = row['episode']
        if ep in seen: continue
        seen.add(ep)
        video = episode_to_mkv(ep, mkv_idx)
        parts = ep.split('_')
        fname = '_'.join(p for p in parts if p and p[0].isupper())
        m = re.search(r'S\d+E\d+', fname)
        label = f"{row['title']}  {m.group(0) if m else fname[:20]}"
        episodes.append({'label': label, 'title': row['title'],
                         'episode': ep, 'video': video})
    return sorted(episodes, key=lambda x: (x['title'], x['label']))

def find_scene(episode, frame_idx, df):
    ep_df = df[df['episode'] == episode]
    if ep_df.empty: return None
    mask = (ep_df['start_frame'] <= frame_idx) & (ep_df['end_frame'] >= frame_idx)
    if mask.sum() > 0: return ep_df[mask].iloc[0]
    return ep_df.iloc[(ep_df['rep_frame'] - frame_idx).abs().argmin()]

def run_dv_inference(row, model, feat_cols):
    feat_vals = [float(row.get(c, 0.0)) for c in feat_cols]
    feats = torch.tensor([feat_vals], dtype=torch.float32)
    with torch.no_grad():
        pred_42, _ = model(feats, torch.zeros(1, dtype=torch.long))
    t = pred_42[0].numpy()
    lut_path = str(TMP_DIR / '_ml_poly.rpu_poly')
    write_rpu_lut(t, lut_path)
    return t, lut_path

def run_hdr10_inference(features_dict, model, feat_cols,
                         maxscl=None, out_nits=None):
    """Run inference using features extracted from HDR10 frame via daemon."""
    feat_vals = []
    for col in feat_cols:
        if col in features_dict:
            feat_vals.append(float(features_dict[col]))
        elif col == 'l1_max_pq':
            feat_vals.append(float(features_dict.get('maxscl', 0.5)))  # proxy
        elif col == 'l1_avg_pq':
            feat_vals.append(float(features_dict.get('average_maxrgb', 0.3)))  # proxy
        else:
            feat_vals.append(0.0)
    feats = torch.tensor([feat_vals], dtype=torch.float32)
    with torch.no_grad():
        pred_42, _ = model(feats, torch.zeros(1, dtype=torch.long))
    t = pred_42[0].numpy()
    lut_path = str(TMP_DIR / '_ml_hdr10.lut')
    write_ml_lut(t, lut_path, maxscl=maxscl, out_nits=out_nits)
    return t, lut_path

def compute_dv_curves(row, ml_target, out_nits, n_pts=256):
    maxscl = float(row.get('maxscl', 0.5))
    l1_avg = float(row.get('l1_avg_pq', 0.3))
    xs = np.linspace(0.0, 1.0, n_pts)
    curves = {'identity': xs.copy()}
    t = row_to_target(row)
    if t is not None:
        ys = eval_rpu(target_to_rpu(t), xs)
        if not np.any(np.isnan(ys)): curves['gold'] = ys
    if ml_target is not None:
        try:
            ys_ml = eval_rpu(target_to_rpu(ml_target), xs)
            if not np.any(np.isnan(ys_ml)): curves['ml'] = ys_ml
        except Exception: pass
    if BASELINE.exists():
        inp = f"scene_id,maxscl,l1_avg_pq,target_nits\n0,{maxscl:.6f},{l1_avg:.6f},{out_nits}\n"
        try:
            r = subprocess.run([str(BASELINE)], input=inp, capture_output=True,
                               text=True, timeout=5)
            lines = r.stdout.strip().split('\n')
            if len(lines) >= 2:
                raw = np.array([float(x) for x in lines[1].split(',')[1:]])
                lut_xs = np.linspace(0, maxscl, len(raw))
                full_xs = np.concatenate([lut_xs, [1.0]])
                full_ys = np.concatenate([raw, [raw[-1]]])
                curves['libplacebo_spline'] = np.interp(xs, full_xs, full_ys)
        except Exception: pass
    return xs, curves, maxscl

@st.cache_resource(show_spinner='Starting feature extractor daemon...')
def get_daemon():
    """Start the feature extractor daemon once and keep it alive for the session."""
    sys.path.insert(0, str(Path(__file__).parent))
    from libplacebo_daemon_client import SustainedFeatureExtractor
    extractor = SustainedFeatureExtractor()
    extractor.start()
    return extractor

@st.cache_data(show_spinner=False)
def detect_bar_norms(video_path, n_samples=5, black_thresh=0.03):
    """
    Detect letterbox bars by sampling actual pixel rows via ffmpeg rawvideo.

    Downscales to 256×72, measures consecutive black rows from top/bottom.
    Each row in 72-row frame = 30 source rows.
    bar_norm = detected_rows × 30 / 2160  (same units as training L5 metadata).

    Uses median across n_samples frames for robustness against scene-level variance.
    """
    import re
    # Get duration first
    try:
        rd = subprocess.run(['ffprobe','-v','error','-show_entries',
                             'format=duration','-of','csv=p=0', video_path],
                            capture_output=True, text=True, timeout=8)
        duration = float(rd.stdout.strip())
    except Exception:
        duration = 3600.0

    pts_list = [duration * i / (n_samples + 1) for i in range(1, n_samples + 1)]

    top_bars, bot_bars = [], []
    for pts in pts_list:
        cmd = ['ffmpeg','-hide_banner','-v','error',
               '-ss', f'{pts:.1f}', '-i', video_path,
               '-vframes','1', '-vf','scale=256:72',
               '-pix_fmt','gray', '-f','rawvideo', '-']
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        if len(r.stdout) < 256 * 72:
            continue
        frame = np.frombuffer(r.stdout, dtype=np.uint8).reshape(72, 256).astype(float) / 255.0
        row_means = frame.mean(axis=1)
        # Count consecutive black rows from top
        top = 0
        for m in row_means:
            if m < black_thresh: top += 1
            else: break
        # Count consecutive black rows from bottom
        bot = 0
        for m in reversed(row_means):
            if m < black_thresh: bot += 1
            else: break
        top_bars.append(top)
        bot_bars.append(bot)

    if not top_bars:
        return (0.0, 0.0)

    med_top = sorted(top_bars)[len(top_bars) // 2]
    med_bot = sorted(bot_bars)[len(bot_bars) // 2]
    return (med_top * 30 / 2160, med_bot * 30 / 2160)

def extract_hdr10_features(video_path, pts):
    """Extract ML features from a single frame at pts using the persistent daemon."""
    try:
        extractor = get_daemon()
        # Decodes ONE frame at the given PTS — seeks then decodes, ~0.15s
        features = extractor.extract_frame(video_path, pts, target_nits=100.0)
        if features is not None:
            # Detect black bars via pixel scan — same units as training L5 metadata
            # (cached per video path — only scans once)
            top_bar, bot_bar = detect_bar_norms(video_path)
            features['top_bar_norm']    = top_bar
            features['bottom_bar_norm'] = bot_bar
            if top_bar > 0.02:
                import streamlit as _st
                _st.caption(f'Letterbox: top={top_bar:.4f} ({top_bar*2160:.0f}px)  '
                            f'bot={bot_bar:.4f} ({bot_bar*2160:.0f}px)')
        return features
    except Exception as e:
        st.error(f'Feature extraction failed: {e}')
        return None

# ── App ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='DV Curve Viewer', layout='wide', page_icon='🎬')
st.title('DV Polynomial Comparison Viewer')

model, feat_cols, ep_num, val_loss = load_model()
df = load_full_dataset()
episodes = get_val_episodes()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f'Model ep={ep_num}  val_loss={val_loss:.5f}')
    viewer_mode = st.radio('Mode', ['DV Val Set', 'HDR10 Any Video'],
                            horizontal=True)
    st.divider()

    if viewer_mode == 'DV Val Set':
        nits_label = st.selectbox('Target display', list(DISPLAY_NITS.keys()))
        out_nits   = DISPLAY_NITS[nits_label]
    else:
        out_nits = float(st.slider('Target nits', 50, 2000, 143, step=10,
                                    help='Peak luminance of target display (nits)'))

    if viewer_mode == 'DV Val Set':
        ep_by_title = {}
        for e in episodes:
            ep_by_title.setdefault(e['title'], []).append(e)
        title_sel  = st.selectbox('Title', sorted(ep_by_title.keys()))
        title_eps  = ep_by_title[title_sel]
        ep_sel_idx = st.selectbox('Episode', range(len(title_eps)),
                                   format_func=lambda i: title_eps[i]['label'])
        sel_ep     = title_eps[ep_sel_idx]
        video_path = sel_ep.get('video')
        if video_path:
            st.caption(Path(video_path).name[-50:])
        else:
            st.error('Video not found in G:/Dataset')
        ep_df     = df[df['episode'] == sel_ep['episode']]
        max_frame = int(ep_df['end_frame'].max()) if not ep_df.empty else 50000
        st.caption(f'{len(ep_df)} scenes, 0–{max_frame:,}')
        frame_idx = int(st.number_input('Frame', 0, max_frame,
                                         min(1000, max_frame), step=1))
        frame_idx = int(st.slider('', 0, max_frame, frame_idx, step=24,
                                   label_visibility='collapsed'))
        render_btn = st.button('Render frames', type='primary',
                                use_container_width=True)
        modes = st.multiselect('Modes', ['gold', 'ml', 'spline', 'st2094-40'],
                                default=['gold', 'ml', 'spline'])

    else:  # HDR10 mode
        hdr10_path = st.text_input('Video path (HDR10 mkv/mp4)',
                                    placeholder='G:/path/to/video.mkv')
        if hdr10_path and Path(hdr10_path).exists():
            h_max = get_frame_count(hdr10_path)
            st.caption(f'~{h_max:,} frames')
            h_frame = int(st.number_input('Frame', 0, h_max,
                                           min(1000, h_max), step=1))
            h_frame = int(st.slider('', 0, h_max, h_frame, step=24,
                                     label_visibility='collapsed'))
        else:
            h_frame = 1000
            if hdr10_path:
                st.error('File not found')
        render_btn = st.button('Extract + Render', type='primary',
                                use_container_width=True)

# ── DV Val Set panel ───────────────────────────────────────────────────────────
if viewer_mode == 'DV Val Set':
    if not video_path:
        st.warning('Video not found — curve-only mode')
    row = find_scene(sel_ep['episode'], frame_idx, df)
    if row is None:
        st.error('Scene not found'); st.stop()
    pts = frame_to_pts(video_path, frame_idx) if video_path else 0.0

    c = st.columns(6)
    c[0].metric('Title',     row.get('title', '?'))
    c[1].metric('Frames',    f"{row.get('start_frame','?')}-{row.get('end_frame','?')}")
    c[2].metric('maxscl',    f"{float(row.get('maxscl',0)):.4f}")
    c[3].metric('l1_max_pq', f"{float(row.get('l1_max_pq',0)):.4f}")
    c[4].metric('l1_avg_pq', f"{float(row.get('l1_avg_pq',0)):.4f}")
    c[5].metric('PTS',       f"{pts:.2f}s")

    ml_target, lut_path = None, None
    try:
        ml_target, lut_path = run_dv_inference(row, model, feat_cols)
    except Exception as e:
        st.warning(f'ML inference failed: {e}')

    xs, curves, maxscl = compute_dv_curves(row, ml_target, out_nits)

    # Curve + stats on one compact row
    curve_col, stat_col = st.columns([3, 1])
    with curve_col:
        fig = make_curve_figure(xs, curves, maxscl)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with stat_col:
        if 'gold' in curves:
            id_rms   = float(np.sqrt(np.mean((curves['gold'] - xs)**2)))
            gold_dev = float(np.mean(curves['gold'] - xs))
            st.metric('Gold dev', f'{gold_dev:+.4f}',
                      help='+ boost / - compress')
            st.metric('Gold vs id', f'{id_rms:.5f}')
        if 'ml' in curves and 'gold' in curves:
            ml_err  = float(np.sqrt(np.mean((curves['ml'] - curves['gold'])**2)))
            id_rms2 = float(np.sqrt(np.mean((curves['gold'] - xs)**2)))
            st.metric('ML vs gold', f'{ml_err:.5f}',
                      delta='✓ beats id' if ml_err < id_rms2 else '✗ worse',
                      delta_color='normal' if ml_err < id_rms2 else 'inverse')

    # Full-width rendered frames below
    rkey = (video_path, frame_idx, out_nits, tuple(modes))
    if render_btn and video_path:
        l1_max = float(row.get('l1_max_pq', 0.5))
        l1_avg = float(row.get('l1_avg_pq', 0.3))
        rendered, prog = {}, st.progress(0, 'Rendering...')
        for i, mode in enumerate(modes):
            prog.progress(i / len(modes), f'Rendering {mode}...')
            lut = lut_path if mode == 'ml' else None
            if mode in ('spline', 'st2094-40'):
                r_max = float(row.get('maxscl', 0.5))
                r_avg = float(row.get('average_maxrgb', 0.3))
            else:
                r_max, r_avg = l1_max, l1_avg
            rendered[mode] = render_frame(video_path, pts, mode,
                                           out_nits, lut, r_max, r_avg)
        prog.empty()
        st.session_state['dv_rendered'] = rendered
        st.session_state['dv_key'] = rkey
    else:
        rendered = st.session_state.get('dv_rendered', {})

    if rendered:
        fcols = st.columns(len(rendered))
        for col, (mode, img) in zip(fcols, rendered.items()):
            with col:
                st.caption(f'**{MODE_LABEL.get(mode, mode)}**')
                if img is not None:
                    st.image(img, use_container_width=True)
                else:
                    st.error(f'{mode} failed')
    else:
        st.info('Click **Render frames**')

# ── HDR10 Any Video panel ──────────────────────────────────────────────────────
else:
    if not hdr10_path or not Path(hdr10_path).exists():
        st.info('Enter an HDR10 video path in the sidebar to begin.')
        st.stop()

    pts = frame_to_pts(hdr10_path, h_frame)
    st.caption(f'Frame {h_frame}  →  PTS {pts:.2f}s  |  {Path(hdr10_path).name}')

    if render_btn:
        with st.spinner('Extracting features from frame...'):
            features = extract_hdr10_features(hdr10_path, pts)

        maxscl_h = features.get('maxscl', 0.5) if features else 0.5

        ml_target_h, lut_path_h = None, None
        if features is not None:
            try:
                ml_target_h, lut_path_h = run_hdr10_inference(
                    features, model, feat_cols,
                    maxscl=maxscl_h, out_nits=out_nits)
                st.success(f'Features extracted  |  '
                           f"maxscl={maxscl_h:.4f}")
            except Exception as e:
                st.error(f'Inference failed: {e}')
        l1_max_h = features.get('maxscl', 0.5) if features else 0.5
        l1_avg_h = features.get('average_maxrgb', 0.3) if features else 0.3

        rendered_h = {}
        prog = st.progress(0, 'Rendering...')

        # Spline — libplacebo default on HDR10
        prog.progress(0.3, 'Rendering libplacebo spline...')
        rendered_h['spline'] = render_frame(hdr10_path, pts, 'spline',
                                             out_nits, None, l1_max_h, l1_avg_h)

        # ML model via LUT tone map function
        if lut_path_h:
            prog.progress(0.7, 'Rendering ML model...')
            rendered_h['ml-lut'] = render_frame(hdr10_path, pts, 'ml-lut',
                                                  out_nits, lut_path_h,
                                                  l1_max_h, l1_avg_h,
                                                  top_bar=top_bar, bot_bar=top_bar)
        prog.empty()
        st.session_state['hdr10_rendered'] = rendered_h
        st.session_state['hdr10_ml_target'] = ml_target_h
        st.session_state['hdr10_maxscl']    = maxscl_h
        st.session_state['hdr10_features']  = features

    # Invalidate cache when video/frame/nits changes
    hdr10_cache_key = (hdr10_path, h_frame, out_nits)
    if st.session_state.get('hdr10_cache_key') != hdr10_cache_key:
        st.session_state.pop('hdr10_rendered', None)
        st.session_state.pop('hdr10_ml_target', None)
        st.session_state.pop('hdr10_maxscl', None)
        st.session_state.pop('hdr10_features', None)
        st.session_state['hdr10_cache_key'] = hdr10_cache_key

    rendered_h   = st.session_state.get('hdr10_rendered', {})
    ml_target_h  = st.session_state.get('hdr10_ml_target')
    maxscl_h     = st.session_state.get('hdr10_maxscl', 0.5)
    features     = st.session_state.get('hdr10_features')

    # Build curves
    xs_h = np.linspace(0, 1, 256)
    curves_h = {'identity': xs_h.copy()}
    if ml_target_h is not None:
        try:
            from scipy.interpolate import UnivariateSpline
            abs_max  = maxscl_h if maxscl_h > 0.05 else 1.0
            # Evaluate full [0,1] polynomial domain
            xs_abs   = np.linspace(0.0, 1.0, 1024)
            ys_poly  = np.maximum.accumulate(
                           np.clip(eval_rpu(target_to_rpu(ml_target_h), xs_abs), 0.0, 1.5))
            sp       = UnivariateSpline(xs_abs, ys_poly,
                                        k=3, s=1024*0.0001, ext=3)
            ys_fit   = np.maximum.accumulate(np.clip(sp(xs_h), 0.0, 1.5))
            # Affine normalise: scale so output at maxscl = target_pq
            # (same as what the LUT renderer does — fair comparison with libplacebo)
            if out_nits and maxscl_h > 0.05:
                maxscl_idx = min(int(maxscl_h * len(xs_h)), len(xs_h)-1)
                peak_val   = float(ys_fit[maxscl_idx])
                f0_val     = float(ys_fit[0])
                denom      = peak_val - f0_val
                tpq        = nits_to_pq(out_nits)
                if denom > 0.001:
                    ys_ml = np.clip((ys_fit - f0_val) / denom * tpq, 0.0, tpq)
                else:
                    ys_ml = xs_h * tpq
            else:
                ys_ml = ys_fit
            if not np.any(np.isnan(ys_ml)): curves_h['ml'] = ys_ml
        except Exception: pass
    if features and BASELINE.exists():
        inp = (f"scene_id,maxscl,l1_avg_pq,target_nits\n"
               f"0,{maxscl_h:.6f},"
               f"{features.get('average_maxrgb', 0.3):.6f},{out_nits}\n")
        try:
            r = subprocess.run([str(BASELINE)], input=inp,
                               capture_output=True, text=True, timeout=5)
            lines = r.stdout.strip().split('\n')
            if len(lines) >= 2:
                raw = np.array([float(x) for x in lines[1].split(',')[1:]])
                lut_xs = np.linspace(0, maxscl_h, len(raw))
                curves_h['libplacebo_spline'] = np.interp(
                    xs_h,
                    np.concatenate([lut_xs, [1.0]]),
                    np.concatenate([raw, [raw[-1]]]))
        except Exception: pass

    # Curve + metadata compact row
    curve_col, stat_col = st.columns([3, 1])
    with curve_col:
        if len(curves_h) > 1:
            fig = make_curve_figure(xs_h, curves_h, maxscl_h,
                                     title='HDR10 inferred curves')
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.info('Click **Extract + Render** to see curves')
    with stat_col:
        if features:
            st.metric('maxscl',    f"{features.get('maxscl',0):.4f}")
            st.metric('avg_maxrgb',f"{features.get('average_maxrgb',0):.4f}")
            st.metric('bright_px', f"{features.get('fraction_bright_pixels',0):.4f}")

    # Full-width rendered frames below
    if rendered_h:
        fcols = st.columns(len(rendered_h))
        for col, (mode, img) in zip(fcols, rendered_h.items()):
            with col:
                st.caption(f'**{MODE_LABEL.get(mode, mode)}**')
                if img is not None:
                    st.image(img, use_container_width=True)
                else:
                    st.error(f'{mode} failed')
    else:
        st.info('Enter a video path and click **Extract + Render**')
