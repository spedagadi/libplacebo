"""
dv_mlp_model.py
===============
MLP replacement for the GBR in dv_coef_model.py.

Key upgrade over GBR: trains on CURVE ERROR, not coefficient error.
Gradients flow from 256-point curve MSE back through a differentiable
piecewise polynomial evaluator into the 42 predicted coefficients.
This forces the model to learn coefficients that are jointly correct
across the full curve — eliminating the pivot-boundary kinks GBR produced.

Architecture:
    [27 pixel features] + [8-dim display tier embedding]  -> 35-dim input
    -> Shared encoder (35 -> 128 -> 128)
    -> 42 raw weights (num_segs, pivots, coefs)
    -> DifferentiablePiecewisePoly  ->  [256-pt curve]
    -> CurveLoss (MSE + monotonicity penalty) vs ground truth 256-pt curve

Usage:
    python ml/dv_mlp_model.py --dataset dv_dataset_full.csv --l1 l1_data.csv
"""

import argparse
import math
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

# Re-use all data-loading and target logic from dv_coef_model
from dv_coef_model import (
    FEATURE_COLS, FEATURE_COLS_5X5, BASE_FEATURE_COLS, SAT_FEATURE_COLS, DERIVED_SAT_COLS,
    FEATURE_SET, TARGET_DIM, MAX_SEGS, MAX_PIVOTS,
    COEF_SCALE, INPUT_MAX,
    row_to_target, target_to_rpu, eval_rpu, _sanitise_rpu,
    load_data, add_derived_sat,
)

# ---------------------------------------------------------------------------
# Differentiable piecewise polynomial evaluator + curve-space loss
# ---------------------------------------------------------------------------
N_CURVE_PTS  = 256   # evaluation resolution for loss and ground-truth curves
GATE_SCALE   = 200.0 # sigmoid sharpness for soft segment masking
MONO_WEIGHT  = 50.0  # penalty weight for non-monotone diffs


# ---------------------------------------------------------------------------
# Training augmentation — Blu-ray grain / compression domain shift
# ---------------------------------------------------------------------------
# Feature layout (indices 0-28 for our 27+2 feature vector):
#   [0]    maxscl
#   [1]    average_maxrgb
#   [2]    fraction_bright_pixels   ← grain dilutes bright pixel count
#   [3-8]  p25..p99 percentiles
#   [9-17] zone_mean_3x3            ← grain effect: small (mean averages it out)
#   [18-26] zone_max_3x3            ← grain effect: largest (max catches spikes)
#   [27]   top_bar_norm             ← DO NOT augment (structural metadata)
#   [28]   bottom_bar_norm          ← DO NOT augment (structural metadata)
PIXEL_FEAT_END = 27   # indices 0-26 are pixel features; 27-28 are bar offsets


def augment_bluray_grain(pixel_features: "torch.Tensor",
                          p: float = 0.5) -> "torch.Tensor":
    """
    Simulate Blu-ray film grain / compression domain shift during training.

    Only applied when grad is enabled (training mode); bypassed at validation/inference.

    Three targeted perturbations:
      1. zone_max_3x3 [18-26]: +Gaussian noise (grain inflates local max values)
      2. fraction_bright_pixels [2]: random deflation up to 5%
         (grain breaks up unified highlight regions into alternating bright/dark pixels)
      3. zone_mean_3x3 [9-17]: tiny noise (much smaller effect than max)

    Bar offset features [27-28] are deliberately left untouched.
    """
    import torch
    if not torch.is_grad_enabled() or torch.rand(1).item() > p:
        return pixel_features

    aug = pixel_features.clone()

    # zone_max_3x3 — grain spike contamination (±2% noise)
    aug[:, 18:27] = torch.clamp(
        aug[:, 18:27] + torch.randn_like(aug[:, 18:27]) * 0.02, 0.0, 1.0
    )

    # fraction_bright_pixels — highlight fragmentation (deflate 0-5%)
    deflation = 1.0 - torch.rand(aug.shape[0], 1, device=aug.device) * 0.05
    aug[:, 2:3] = aug[:, 2:3] * deflation

    # zone_mean_3x3 — minimal grain effect (±0.5% noise)
    aug[:, 9:18] = torch.clamp(
        aug[:, 9:18] + torch.randn_like(aug[:, 9:18]) * 0.005, 0.0, 1.0
    )

    return aug


class DifferentiablePiecewisePoly(nn.Module):
    """
    Maps a [B, 42] predicted coefficient vector to a [B, N_CURVE_PTS] curve.

    Target vector layout (from dv_coef_model.py):
        [0]      num_segs  (not used here — we always evaluate all MAX_SEGS)
        [1..9]   pivots    (9 values; last = 1.0 by cumsum-softmax in model)
        [10..41] 8 × 4 per segment: order(unused), c0f, c1f, c2f

    Segment masking uses a partition-of-unity soft gate:
        gate_i(x) = sigmoid(scale*(x - p_i)) * sigmoid(scale*(p_{i+1} - x))
    Gates are then L1-normalised across segments so they sum to 1.0 at every x.
    This lets gradients flow across boundaries without double-counting.
    """

    def __init__(self, n_pts=N_CURVE_PTS, gate_scale=GATE_SCALE):
        super().__init__()
        self.n_pts = n_pts
        self.gate_scale = gate_scale
        # Static evaluation grid [1, n_pts] — moves to GPU automatically
        self.register_buffer('x_eval', torch.linspace(0.0, 1.0, n_pts).unsqueeze(0))

    def forward(self, pred):
        """
        Args:
            pred : [B, 42]  raw MLP output
        Returns:
            curve : [B, N_CURVE_PTS]  evaluated piecewise polynomial
        """
        B = pred.shape[0]
        # pivots: [B, MAX_PIVOTS=9]  — already in [0,1], monotone from model
        pivots = pred[:, 1:1 + MAX_PIVOTS]          # [B, 9]

        # coefficients: layout is [order, c0, c1, c2] × 8 segs starting at index 10
        # order is integer metadata — ignore for differentiable eval, always use quadratic
        c0 = pred[:, 10 + 1::4][:, :MAX_SEGS]      # [B, 8]  every 4th starting at 11
        c1 = pred[:, 10 + 2::4][:, :MAX_SEGS]      # [B, 8]
        c2 = pred[:, 10 + 3::4][:, :MAX_SEGS]      # [B, 8]

        x = self.x_eval.expand(B, -1)               # [B, n_pts]

        # Evaluate all 8 segments everywhere: [B, 8, n_pts]
        x_e = x.unsqueeze(1)                         # [B, 1, n_pts]
        seg_y = (c0.unsqueeze(2)
                 + c1.unsqueeze(2) * x_e
                 + c2.unsqueeze(2) * x_e ** 2)       # [B, 8, n_pts]

        # Soft partition-of-unity gates per segment
        p_lo = pivots[:, :MAX_SEGS].unsqueeze(2)     # [B, 8, 1]
        p_hi = pivots[:, 1:].unsqueeze(2)             # [B, 8, 1]
        gate = (torch.sigmoid(self.gate_scale * (x_e - p_lo))
                * torch.sigmoid(self.gate_scale * (p_hi - x_e)))  # [B, 8, n_pts]

        # Normalise so gates sum to 1.0 at every x (true partition of unity)
        gate = gate / (gate.sum(dim=1, keepdim=True) + 1e-8)      # [B, 8, n_pts]

        # Weighted sum of segment outputs
        curve = (seg_y * gate).sum(dim=1)             # [B, n_pts]
        return curve


class CurveLoss(nn.Module):
    """
    Multi-task loss:
      1. Curve MSE:         differentiable poly evaluator → MSE vs 256-pt gold curve
      2. Monotonicity:      penalise non-monotone curve steps
      3. Trim MAE:          [slope, offset, power, ms_weight] normalised around identity=2048

    Trim values are normalised to [-1, 1] around 2048 so gradients are on the same
    scale as the curve loss (which is already in [0, 1]).
    """

    TRIM_WEIGHT = 0.05  # default; pass trim_weight=0.0 to disable trim head

    def __init__(self, mono_weight=MONO_WEIGHT, trim_weight=None):
        super().__init__()
        self.evaluator   = DifferentiablePiecewisePoly()
        self.mse         = nn.MSELoss()
        self.mae         = nn.L1Loss()
        self.mono_weight = mono_weight
        self.trim_weight = self.TRIM_WEIGHT if trim_weight is None else trim_weight

    @staticmethod
    def curve_weights(gold_devs):
        """
        Per-sample loss weights based on gold curve type.
        Focuses gradient on under-represented / hard curve types.
          boost        (gold_dev >  0.02): 5.0  — model almost never predicts boost
          near_identity(|gold_dev| < 0.01): 6.0  — model over-compresses here
          mild_compress(gold_dev  > -0.05): 1.5  — slight over-compression
          strong_compress (else)          : 1.0  — already learned perfectly
        """
        w = torch.ones_like(gold_devs)
        w = torch.where(gold_devs  >  0.02, torch.full_like(w, 5.0), w)
        w = torch.where(gold_devs.abs() < 0.01, torch.full_like(w, 6.0), w)
        w = torch.where((gold_devs > -0.05) & (gold_devs <= -0.01),
                        torch.full_like(w, 1.5), w)
        return w

    def forward(self, pred_42, pred_trim, gold_256, gold_trim, gold_devs=None):
        """
        Args:
            pred_42   : [B, 42]         MLP poly output
            pred_trim : [B, 4]          predicted [slope, offset, power, ms_weight]
            gold_256  : [B, 256]        ground truth curve
            gold_trim : [B, 4]          ground truth trim values (raw, centred on 2048)
            gold_devs : [B]             optional mean(gold - identity) for weighted loss
        Returns:
            total, loss_curve, loss_mono, loss_trim
        """
        pred_curve = self.evaluator(pred_42)

        if gold_devs is not None:
            w = self.curve_weights(gold_devs).view(-1, 1).to(pred_curve.device)
            loss_curve = (w * (pred_curve - gold_256).pow(2)).mean()
        else:
            loss_curve = self.mse(pred_curve, gold_256)

        diffs     = pred_curve[:, 1:] - pred_curve[:, :-1]
        loss_mono = torch.mean(torch.clamp(-diffs, min=0.0) ** 2) * self.mono_weight

        # Normalise trim to [-1, 1]: deviation from identity 2048, scaled by 2048
        pred_trim_n = (pred_trim - TRIM_IDENTITY) / TRIM_IDENTITY
        gold_trim_n = (gold_trim - TRIM_IDENTITY) / TRIM_IDENTITY
        loss_trim   = self.mae(pred_trim_n, gold_trim_n) * self.trim_weight

        total = loss_curve + loss_mono + loss_trim
        return total, loss_curve, loss_mono, loss_trim


def precompute_curve(row, n_pts=N_CURVE_PTS):
    """
    Evaluate the gold RPU polynomial to a dense curve for use as training target.
    Returns float32 array [n_pts] or None if row is invalid.
    """
    t = row_to_target(row)
    if t is None:
        return None
    xs = np.linspace(0.0, 1.0, n_pts)
    try:
        rpu = target_to_rpu(t)
        ys  = eval_rpu(rpu, xs)
        if np.any(np.isnan(ys)) or np.any(np.isinf(ys)):
            return None
        return ys.astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Display tier mapping — derived from RPU L2 analysis across all 17 titles
# ---------------------------------------------------------------------------
DISPLAY_NITS   = [143, 1030, 1669]
NITS_TO_IDX    = {143: 0, 1030: 1, 1669: 2}
DISPLAY_PQ     = [2081, 2851, 3079]   # matching target_max_pq values in trim CSVs
NUM_TIERS      = len(DISPLAY_NITS)

# Black bar features — L5 active area offsets normalised by frame height (2160)
# top_bar_norm and bottom_bar_norm inform the model which SAT zone rows are unreliable
BAR_FEATURE_COLS = ["top_bar_norm", "bottom_bar_norm"]   # pixel_dim += 2 (27 → 29)

# Trim parameters predicted per tier: slope, offset, power
# ms_weight excluded — it is a per-title colorist constant (e.g. mindhunter=512, most=2048)
# not predictable from per-scene pixel features; including it contaminates the trim loss.
# chroma_weight and sat_gain are also consistently 2048 (identity) — excluded.
TRIM_PARAMS    = ['trim_slope', 'trim_offset', 'trim_power']
TRIM_DIM       = len(TRIM_PARAMS)       # 3 values per tier
TRIM_IDENTITY  = 2048.0                 # neutral value for all trim params
TRIM_DIR       = Path("F:/DTMModelData/trims")


def expand_tiers(df):
    """
    Replicate each scene row once per display tier, adding a 'target_nits' column.
    Transforms N scenes -> N*3 rows so the model trains on all tier targets.
    """
    frames = []
    for nits in DISPLAY_NITS:
        copy = df.copy()
        copy['target_nits'] = nits
        frames.append(copy)
    return pd.concat(frames, ignore_index=True)


def nits_to_tier(nits_value):
    """Map raw nits float to nearest discrete tier index."""
    nits = int(round(float(nits_value)))
    nearest = min(DISPLAY_NITS, key=lambda n: abs(n - nits))
    return NITS_TO_IDX[nearest]


def load_trim_lookup(title_key, episode_stem):
    """
    Load trim CSV for one episode, return dict: {tier_idx -> [slope, offset, power, ms_weight]}
    keyed by frame index.  Returns None if file not found (identity trims will be used).

    trim CSV columns: frame, target_max_pq, trim_slope, trim_offset,
                      trim_power, trim_chroma_weight, trim_saturation_gain, ms_weight
    """
    pattern = f"{title_key}_{episode_stem}*.trim.csv"
    matches = list(TRIM_DIR.glob(pattern))
    if not matches:
        return None

    df = pd.read_csv(matches[0])
    # Build frame -> tier -> [4 values] lookup
    lookup = {}   # frame_idx -> {tier_idx: np.array([4])}
    pq_to_tier = {pq: idx for idx, pq in enumerate(DISPLAY_PQ)}

    for _, row in df.iterrows():
        frame = int(row['frame'])
        pq    = int(row['target_max_pq'])
        if pq not in pq_to_tier:
            # Snap to nearest known PQ
            pq = min(DISPLAY_PQ, key=lambda p: abs(p - pq))
        tier = pq_to_tier[pq]
        vals = np.array([
            row.get('trim_slope',   TRIM_IDENTITY),
            row.get('trim_offset',  TRIM_IDENTITY),
            row.get('trim_power',   TRIM_IDENTITY),
            row.get('ms_weight',    TRIM_IDENTITY),
        ], dtype=np.float32)
        if frame not in lookup:
            lookup[frame] = {}
        lookup[frame][tier] = vals

    return lookup


def get_trim_for_frame(trim_lookup, frame_idx, tier_idx):
    """Return [4] trim values for a frame/tier, falling back to identity."""
    identity = np.full(TRIM_DIM, TRIM_IDENTITY, dtype=np.float32)
    if trim_lookup is None:
        return identity
    frame_trims = trim_lookup.get(frame_idx, {})
    return frame_trims.get(tier_idx, identity)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DVCoefDataset(Dataset):
    """
    One sample = one training row with valid RPU coefficients.

    Returns 5-tuple:
        features   : [27]           normalised ICtCp-I pixel statistics
        tier_idx   : []             LongTensor — display tier index 0/1/2
        target_42  : [42]           RPU polynomial coefficient vector (for eval/inject)
        gold_curve : [N_CURVE_PTS]  pre-evaluated 256-pt gold curve (curve loss target)
        gold_trims : [TRIM_DIM=4]   [slope, offset, power, ms_weight] for requested tier
    """

    def __init__(self, df, feat_cols, tier_col='display_tier'):
        self.feat_cols = feat_cols
        rows, targets, curves, trims = [], [], [], []

        n_total = len(df)
        n_skip  = 0
        print(f"  Building dataset from {n_total:,} rows ...", flush=True)
        t0 = time.time()
        for i, (_, row) in enumerate(df.iterrows()):
            t   = row_to_target(row)
            crv = precompute_curve(row)
            if t is None or crv is None:
                n_skip += 1
                continue

            tier    = nits_to_tier(row.get('target_nits', 143))
            tier_pq = DISPLAY_PQ[tier]

            # Read trim directly from embedded dataset columns (trim_XXXX_param).
            # Falls back to identity (2048) for any missing column.
            trim_vals = np.array([
                float(row.get(f'trim_{tier_pq}_slope',  TRIM_IDENTITY)),
                float(row.get(f'trim_{tier_pq}_offset', TRIM_IDENTITY)),
                float(row.get(f'trim_{tier_pq}_power',  TRIM_IDENTITY)),
            ], dtype=np.float32)

            rows.append(row)
            targets.append(t)
            curves.append(crv)
            trims.append(trim_vals)

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta  = (n_total - i - 1) / rate
                print(f"    {i+1:>6,}/{n_total:,}  valid={len(rows):,}  "
                      f"skip={n_skip}  {rate:.0f} rows/s  ETA {eta:.0f}s", flush=True)

        if not rows:
            raise ValueError("No valid rows with RPU coefficients found in dataset.")
        print(f"  Dataset built: {len(rows):,} valid  {n_skip} skipped  "
              f"({time.time()-t0:.1f}s)", flush=True)

        self.df_valid = pd.DataFrame(rows).reset_index(drop=True)
        self.targets  = np.vstack(targets).astype(np.float32)   # [N, 42]
        self.curves   = np.vstack(curves).astype(np.float32)    # [N, N_CURVE_PTS]
        self.trims    = np.vstack(trims).astype(np.float32)     # [N, TRIM_DIM]

        # gold_dev: mean(gold_curve - identity) per scene
        # Used for weighted loss: boost/near_identity scenes get higher weight
        xs_norm = np.linspace(0.0, 1.0, self.curves.shape[1], dtype=np.float32)
        self.gold_devs = (self.curves - xs_norm[None, :]).mean(axis=1)  # [N]

        # Core pixel feature columns + bar offset features appended last
        bar_avail = [c for c in BAR_FEATURE_COLS if c in self.df_valid.columns]
        avail = [c for c in feat_cols if c in self.df_valid.columns]
        missing_cols = [c for c in feat_cols if c not in self.df_valid.columns]
        if missing_cols:
            print(f"  WARNING: {len(missing_cols)} feature cols missing from dataset: {missing_cols}", flush=True)
        all_feat_cols = avail + [c for c in bar_avail if c not in avail]
        self.X = self.df_valid[all_feat_cols].values.astype(np.float32)
        self.feat_cols_used = all_feat_cols
        print(f"  Feature cols resolved: {len(all_feat_cols)}  "
              f"(pixel={len(avail)}  bar={len(bar_avail)})", flush=True)

        if tier_col in self.df_valid.columns:
            self.tiers = self.df_valid[tier_col].values.astype(np.int64)
        else:
            self.tiers = np.array(
                [nits_to_tier(r.get('target_nits', 143)) for _, r in self.df_valid.iterrows()],
                dtype=np.int64
            )

        self.groups = self.df_valid['scene_id'].values if 'scene_id' in self.df_valid.columns \
                      else np.arange(len(self.df_valid))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.tiers[idx], dtype=torch.long),
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.curves[idx]),
            torch.from_numpy(self.trims[idx]),
            torch.tensor(self.gold_devs[idx], dtype=torch.float32),  # for weighted loss
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DVPolyMLP(nn.Module):
    """
    Predicts the 42-dim RPU polynomial coefficient vector from:
      - 27 ICtCp-I pixel features
      - display tier categorical embedding (3 classes -> 8-dim)

    Target vector layout (same as dv_coef_model.py):
      [0]      num_segs
      [1..9]   pivots (normalised 0-1)
      [10..41] seg0..seg7: order, c0f, c1f, c2f  (4 values each)

    Pivot predictions are passed through a cumulative-sum + softmax to
    enforce monotonicity before concatenating with the rest of the target.
    """

    def __init__(
        self,
        pixel_dim=27,
        num_tiers=NUM_TIERS,
        embed_dim=8,
        hidden_dim=128,
        target_dim=TARGET_DIM,
        dropout=0.3,
        use_tier_embed=True,
    ):
        super().__init__()

        self.use_tier_embed = use_tier_embed
        if use_tier_embed:
            self.embed = nn.Embedding(num_tiers, embed_dim)
            in_dim = pixel_dim + embed_dim
        else:
            self.embed = None
            in_dim = pixel_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # num_segs head — predicts scalar in [1, MAX_SEGS]
        self.seg_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Pivot head — MAX_PIVOTS logits -> softmax -> cumsum -> [0,1] monotone
        self.pivot_head = nn.Linear(hidden_dim, MAX_PIVOTS)

        # Coefficient head — (num_segs * 4) values: order + c0f + c1f + c2f per seg
        self.coef_head = nn.Linear(hidden_dim, MAX_SEGS * 4)

        # Trim head — predicts [slope, offset, power, ms_weight] for the REQUESTED tier.
        # Tier context comes from the embedding — predicting per-requested-tier (not all 3)
        # avoids masking and is consistent with the forward pass design.
        self.trim_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, TRIM_DIM),   # [B, 4] — unconstrained, identity = 2048
        )

    def forward(self, pixel_features, tier_ids):
        """
        Args:
            pixel_features : [B, 27]  float32, values in [0, 1]
            tier_ids       : [B]      int64,   values in {0, 1, 2}
        Returns:
            pred_42  : [B, 42]         polynomial coefficients
            pred_trim: [B, TRIM_DIM=4] [slope, offset, power, ms_weight] for requested tier
        """
        x_pix = torch.clamp(pixel_features, 0.0, 1.0)
        if self.use_tier_embed:
            x_emb = self.embed(tier_ids)
            x = torch.cat([x_pix, x_emb], dim=1)
        else:
            x = x_pix

        h = self.encoder(x)

        num_segs   = self.seg_head(h).squeeze(1) * MAX_SEGS
        piv_probs  = torch.softmax(self.pivot_head(h), dim=1)
        pivots     = torch.cumsum(piv_probs, dim=1)
        coefs      = self.coef_head(h)

        pred_42 = torch.cat([
            num_segs.unsqueeze(1),
            pivots,
            coefs,
        ], dim=1)

        pred_trim = self.trim_head(h)   # [B, 4] raw — denormalise with * 2048 + 2048 at output

        return pred_42, pred_trim


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def train(df, feat_cols=None, epochs=100, batch_size=64, lr=3e-4, device=None,
          save_path=None, save_every=10, val_df=None, dropout=0.3, no_trim=False,
          no_tier=False, weighted_loss=False):
    """
    Train DVPolyMLP. Returns (model, feat_cols_used).
    Interface mirrors dv_coef_model.train() for drop-in use.

    Args:
        save_path   : path prefix for checkpoint files (e.g. 'F:/DTMModelData/ckpt').
                      Saves <save_path>_best.pt on new best val loss and
                      <save_path>_ep{N}.pt every save_every epochs.
        save_every  : checkpoint interval in epochs (default 10).
    """
    if feat_cols is None:
        feat_cols = FEATURE_COLS  # full 27-feature set

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 72, flush=True)
    print(f"DVPolyMLP training", flush=True)
    print(f"  device={device}  epochs={epochs}  batch={batch_size}  lr={lr}", flush=True)
    if save_path:
        print(f"  checkpoints: {save_path}_best.pt  (every {save_every} ep)", flush=True)
    print("=" * 72, flush=True)

    if no_tier:
        # Polynomial is tier-independent — one row per scene, no tier embedding
        print(f"  Tier embedding DISABLED — one row per scene (polynomial is tier-independent)", flush=True)
        if 'target_nits' not in df.columns:
            df = df.copy()
            df['target_nits'] = 143   # SDR tier used only for trim column lookup; ignored for curve
        df = df.copy()
        df['display_tier'] = 0
    else:
        # Expand scenes to one row per display tier if target_nits not already set
        if 'target_nits' not in df.columns:
            n_orig = len(df)
            df = expand_tiers(df)
            print(f"  Tier expansion: {n_orig:,} scenes × {NUM_TIERS} tiers = {len(df):,} rows", flush=True)
        if 'display_tier' not in df.columns:
            df = df.copy()
            df['display_tier'] = df['target_nits'].apply(nits_to_tier)

    print(f"\n[1/4] Building dataset ({len(df):,} rows) ...", flush=True)
    ds = DVCoefDataset(df, feat_cols)
    pixel_dim = len(ds.feat_cols_used)
    print(f"  Feat cols used: {ds.feat_cols_used}", flush=True)

    # Tier distribution
    tier_counts = np.bincount(ds.tiers, minlength=NUM_TIERS)
    for i, (nits, cnt) in enumerate(zip(DISPLAY_NITS, tier_counts)):
        print(f"  Tier {i} ({nits:4d} nits): {cnt:>6,} samples ({100*cnt/len(ds):.1f}%)", flush=True)

    if val_df is not None:
        # Apply same tier logic to val set
        if no_tier:
            if 'target_nits' not in val_df.columns:
                val_df = val_df.copy()
                val_df['target_nits'] = 143
            val_df = val_df.copy()
            val_df['display_tier'] = 0
        elif 'target_nits' not in val_df.columns:
            val_df = expand_tiers(val_df)
        # Use dedicated held-out val set (cross-title generalisation)
        print(f"\n[2/4] Building dedicated val dataset ({len(val_df):,} rows) ...", flush=True)
        val_ds = DVCoefDataset(val_df, feat_cols)
        tr_ds  = ds
        te_ds  = val_ds
        print(f"  Val titles: {sorted(val_df['title'].unique()) if 'title' in val_df.columns else 'N/A'}", flush=True)
    else:
        # Fallback: scene-grouped 80/20 split from train data
        print(f"\n[2/4] Splitting train/val (scene-grouped 80/20, no dedicated val set) ...", flush=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(np.arange(len(ds)), groups=ds.groups))
        tr_ds = torch.utils.data.Subset(ds, tr_idx)
        te_ds = torch.utils.data.Subset(ds, te_idx)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  Train: {len(tr_ds):,}  Val: {len(te_ds):,}  "
          f"Batches/epoch: {len(tr_loader)}", flush=True)

    print(f"\n[3/4] Building model ...", flush=True)
    model    = DVPolyMLP(pixel_dim=pixel_dim, dropout=dropout,
                         use_tier_embed=(not no_tier)).to(device)
    trim_w   = 0.0 if no_trim else 0.05
    loss_fn  = CurveLoss(trim_weight=trim_w).to(device)
    if no_trim:
        print(f"  Trim head DISABLED — curve-only training", flush=True)
    if weighted_loss:
        print(f"  Weighted loss ENABLED: boost=5x  near_identity=6x  mild=1.5x", flush=True)
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  pixel_dim={pixel_dim}  params={n_params:,}", flush=True)

    print(f"\n[4/4] Training for {epochs} epochs ...", flush=True)
    print("  Losses: curve=curve-MSE  mono=monotonicity-penalty  trim=trim-MAE(normalised)", flush=True)
    print("  Trim MAE in raw units (identity=2048); grad=pre-clip gradient norm", flush=True)

    best_val, best_state = float('inf'), None
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()

        # ---- Train ----
        model.train()
        tr_loss = tr_curve = tr_mono = tr_trim = 0.0
        tr_grad_norm = 0.0
        n_tr_batches = 0
        for feats, tiers, _, gold_curves, gold_trims, gold_devs_batch in tr_loader:
            feats, tiers   = feats.to(device), tiers.to(device)
            gold_curves    = gold_curves.to(device)
            gold_trims     = gold_trims.to(device)

            feats = augment_bluray_grain(feats, p=0.5)

            opt.zero_grad()
            pred_42, pred_trim = model(feats, tiers)
            w = gold_devs_batch.to(device) if weighted_loss else None
            loss, lc, lm, lt   = loss_fn(pred_42, pred_trim, gold_curves, gold_trims, w)
            loss.backward()

            # Capture grad norm before clipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step()

            n = len(feats)
            tr_loss      += loss.item() * n
            tr_curve     += lc.item()   * n
            tr_mono      += lm.item()   * n
            tr_trim      += lt.item()   * n
            tr_grad_norm += grad_norm
            n_tr_batches += 1

        N_tr = len(tr_ds)
        tr_loss  /= N_tr
        tr_curve /= N_tr
        tr_mono  /= N_tr
        tr_trim  /= N_tr
        tr_grad_norm /= n_tr_batches

        # ---- Validate ----
        model.eval()
        val_loss = val_curve = val_mono = val_trim = 0.0
        # Per-param trim MAE in raw units (sum, divide by N_val after)
        trim_abs_err = torch.zeros(TRIM_DIM)   # [slope, offset, power, ms_weight]
        N_val = len(te_ds)

        with torch.no_grad():
            for feats, tiers, _, gold_curves, gold_trims, gold_devs_batch in te_loader:
                feats, tiers   = feats.to(device), tiers.to(device)
                gold_curves    = gold_curves.to(device)
                gold_trims     = gold_trims.to(device)

                p42, pt = model(feats, tiers)
                # Always evaluate val loss without weighting for comparability
                l, lc, lm, lt = loss_fn(p42, pt, gold_curves, gold_trims, None)

                n = len(feats)
                val_loss  += l.item()  * n
                val_curve += lc.item() * n
                val_mono  += lm.item() * n
                val_trim  += lt.item() * n

                # pred_trim (pt) is in raw trim units — CurveLoss normalises internally.
                # Direct absolute error vs gold_trims (also raw units, identity=2048).
                trim_abs_err += torch.abs(pt - gold_trims).sum(dim=0).cpu()

        val_loss  /= N_val
        val_curve /= N_val
        val_mono  /= N_val
        val_trim  /= N_val
        trim_mae_raw = (trim_abs_err / N_val).tolist()   # [slope, offset, power, ms_weight]

        sched.step()
        ep_secs = time.time() - ep_t0
        lr_now  = sched.get_last_lr()[0]
        is_best = val_loss < best_val

        if is_best:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                torch.save({'epoch': epoch, 'model_state': best_state,
                            'val_loss': best_val, 'feat_cols': ds.feat_cols_used},
                           f"{save_path}_best.pt")

        best_mark = '  ** BEST **' if is_best else ''
        print(f"\nEp {epoch:3d}/{epochs}  ({ep_secs:.1f}s)  "
              f"lr={lr_now:.3e}  grad={tr_grad_norm:.3f}{best_mark}", flush=True)
        if no_trim:
            print(f"  TRAIN  curve={tr_curve:.5f}  mono={tr_mono:.5f}", flush=True)
            print(f"  VAL    curve={val_curve:.5f}  mono={val_mono:.5f}", flush=True)
        else:
            print(f"  TRAIN  total={tr_loss:.5f}  "
                  f"curve={tr_curve:.5f}  mono={tr_mono:.5f}  trim={tr_trim:.5f}", flush=True)
            print(f"  VAL    total={val_loss:.5f}  "
                  f"curve={val_curve:.5f}  mono={val_mono:.5f}  trim={val_trim:.5f}", flush=True)
            print(f"  TRIM   slope={trim_mae_raw[0]:.1f}  offset={trim_mae_raw[1]:.1f}  "
                  f"power={trim_mae_raw[2]:.1f}  "
                  f"(raw MAE, identity=2048)", flush=True)

        if save_path and epoch % save_every == 0:
            ckpt = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'epoch': epoch, 'model_state': ckpt,
                        'val_loss': val_loss, 'feat_cols': ds.feat_cols_used},
                       f"{save_path}_ep{epoch:04d}.pt")
            print(f"  -> checkpoint: {save_path}_ep{epoch:04d}.pt", flush=True)

    total_mins = (time.time() - train_start) / 60
    print(f"\nTraining complete: {total_mins:.1f} min  best_val={best_val:.5f}", flush=True)

    # Restore best checkpoint
    model.load_state_dict(best_state)

    # Evaluate in curve space MAE
    print("\nEvaluating held-out MAE (curve space) ...", flush=True)
    model.eval()
    xs = np.linspace(0, 1, 128)
    mae_list = []
    with torch.no_grad():
        for feats, tiers, targets, _, _, _ in te_loader:
            feats, tiers = feats.to(device), tiers.to(device)
            pred_42, _   = model(feats, tiers)
            preds        = pred_42.cpu().numpy()
            for i in range(len(preds)):
                t_gold = targets[i].numpy()
                t_pred = preds[i]
                try:
                    rpu_gold = target_to_rpu(t_gold)
                    rpu_pred = target_to_rpu(t_pred)
                    ys_gold  = eval_rpu(rpu_gold, xs)
                    ys_pred  = eval_rpu(rpu_pred, xs)
                    mae_list.append(np.mean(np.abs(ys_gold - ys_pred)))
                except Exception:
                    pass

    mae = np.mean(mae_list) if mae_list else float('nan')
    print(f"Held-out MAE (curve space): {mae:.5f}  ({mae*1500:.1f} nits approx)", flush=True)
    print(f"Best val loss: {best_val:.5f}  Evaluated: {len(mae_list)} scenes", flush=True)

    return model, ds.feat_cols_used


def predict(model, feat_cols, row, device=None):
    """Predict RPU target vector for one row. Mirrors dv_coef_model.predict()."""
    if device is None:
        device = next(model.parameters()).device

    feats = torch.tensor(
        [[float(row.get(f, 0.0)) for f in feat_cols]], dtype=torch.float32
    ).to(device)

    tier_val = row.get('target_nits', 143)
    tier = torch.tensor([nits_to_tier(tier_val)], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        pred_42, _ = model(feats, tier)
        pred = pred_42.cpu().numpy()[0]
    return pred


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, path):
        self._file = open(path, 'w', buffering=1, encoding='utf-8')
        self._stdout = sys.stdout
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',     required=True)
    ap.add_argument('--val-dataset', default=None,
                    help='Second CSV to merge (e.g. F:/DTMModelData/val/val_dataset.csv). '
                         'Combined with --dataset when --val-titles is given; otherwise used as val set directly.')
    ap.add_argument('--val-titles',  default=None,
                    help='Comma-separated title keys to hold out as val (e.g. mindhunter,euphoria,prehistoric,our,wondla). '
                         'Requires --val-dataset so both CSVs are available. Splits the combined data by title.')
    ap.add_argument('--l1',          default=None)
    ap.add_argument('--epochs',      type=int,   default=100)
    ap.add_argument('--batch-size',  type=int,   default=64)
    ap.add_argument('--lr',          type=float, default=3e-4)
    ap.add_argument('--save',        default=None,
                    help='Checkpoint path prefix (e.g. F:/DTMModelData/ckpt). '
                         'Saves _best.pt on new best and _epNNNN.pt every --save-every epochs.')
    ap.add_argument('--save-every',  type=int, default=10,
                    help='Save a periodic checkpoint every N epochs (default 10)')
    ap.add_argument('--dropout',     type=float, default=0.3,
                    help='Encoder dropout rate (default 0.3)')
    ap.add_argument('--no-trim',    action='store_true',
                    help='Disable trim head entirely — curve-only training. '
                         'Use when trim will be predicted by a separate model.')
    ap.add_argument('--weighted-loss', action='store_true',
                    help='Weight curve MSE by curve type: boost=5x, near_identity=6x, '
                         'mild_compress=1.5x. Focuses gradient on under-represented '
                         'hard cases. Val loss always unweighted for comparability.')
    ap.add_argument('--no-tier',    action='store_true',
                    help='Disable tier expansion and tier embedding. '
                         'Correct for curve-only training: the RPU polynomial is '
                         'tier-independent (one curve per scene, not per display tier). '
                         'Reduces dataset 3x and removes irrelevant conditioning.')
    ap.add_argument('--use-5x5',    action='store_true',
                    help='Add 5x5 zone features (pixel_dim 29 -> 79). '
                         'Richer spatial resolution; may improve content-type discrimination.')
    ap.add_argument('--split-episodes', default=None,
                    help='Episode-level partial split for one title. '
                         'Format: TITLE:VAL_EP1,VAL_EP2 (episode substrings). '
                         'Example: --split-episodes our:E07,E08  moves our_planet E01-E06 '
                         'to training and keeps E07,E08 in val. '
                         'Requires --val-titles to include TITLE.')
    ap.add_argument('--log',         default=None,
                    help='Path to save a copy of all stdout. Written to terminal and file simultaneously.')
    args = ap.parse_args()

    tee = None
    if args.log:
        tee = _Tee(args.log)
        sys.stdout = tee
        print(f"Logging to: {args.log}", flush=True)

    try:
        print(f"Loading dataset: {args.dataset}", flush=True)
        df = load_data(args.dataset, args.l1)
        print(f"  {len(df):,} rows  titles: {sorted(df['title'].unique())}", flush=True)

        val_df = None

        if args.val_titles:
            # Title-based split: combine both CSVs, partition by title
            if not args.val_dataset:
                raise ValueError("--val-titles requires --val-dataset to supply the second CSV")
            print(f"Loading second CSV:  {args.val_dataset}", flush=True)
            df2 = load_data(args.val_dataset)
            print(f"  {len(df2):,} rows  titles: {sorted(df2['title'].unique())}", flush=True)

            all_data   = pd.concat([df, df2], ignore_index=True)
            val_keys   = set(t.strip() for t in args.val_titles.split(','))
            df         = all_data[~all_data['title'].isin(val_keys)].reset_index(drop=True)
            val_df     = all_data[ all_data['title'].isin(val_keys)].reset_index(drop=True)

            print(f"\nTitle-based split  (val-titles={sorted(val_keys)})", flush=True)
            print(f"  Train: {len(df):,} scenes  titles: {sorted(df['title'].unique())}", flush=True)
            print(f"  Val:   {len(val_df):,} scenes  titles: {sorted(val_df['title'].unique())}", flush=True)

            # Episode-level partial split for one title (e.g. our:E07,E08)
            if args.split_episodes:
                split_title, val_eps_str = args.split_episodes.split(':', 1)
                val_ep_patterns = [p.strip() for p in val_eps_str.split(',')]
                # Rows where title matches AND episode contains any val pattern → stay in val
                # Rows where title matches AND episode does NOT match → move to train
                if 'episode' not in df.columns:
                    raise ValueError("--split-episodes requires an 'episode' column in the dataset")
                split_mask = val_df['title'] == split_title
                ep_col = val_df.loc[split_mask, 'episode']
                in_val_eps = ep_col.apply(
                    lambda e: any(pat in str(e) for pat in val_ep_patterns)
                )
                # Rows of the split title that go to train
                move_to_train = val_df[split_mask & ~in_val_eps]
                # Keep only matched episodes in val
                val_df = val_df[~split_mask | in_val_eps].reset_index(drop=True)
                df     = pd.concat([df, move_to_train], ignore_index=True)
                print(f"\n  Episode split for '{split_title}': "
                      f"val_eps={val_ep_patterns}", flush=True)
                print(f"  Moved {len(move_to_train):,} '{split_title}' scenes to train", flush=True)
                print(f"  Val  '{split_title}' scenes remaining: "
                      f"{(val_df['title']==split_title).sum():,}", flush=True)

            # Validate — episode split means title can appear in both, that's intentional
            print(f"  Final train: {len(df):,}  val: {len(val_df):,}", flush=True)

        elif args.val_dataset:
            print(f"Loading val dataset: {args.val_dataset}", flush=True)
            val_df = load_data(args.val_dataset)
            if 'target_nits' not in val_df.columns:
                val_df = expand_tiers(val_df)
            print(f"  {len(val_df):,} val rows ({len(val_df)//NUM_TIERS:,} scenes x {NUM_TIERS} tiers)", flush=True)

        feat_cols = FEATURE_COLS_5X5 if args.use_5x5 else FEATURE_COLS
        if args.use_5x5:
            print(f"Using 5x5 features: pixel_dim=79  (3x3+5x5 zones)", flush=True)

        model, feats = train(df, feat_cols=feat_cols,
                             epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                             save_path=args.save, save_every=args.save_every, val_df=val_df,
                             dropout=args.dropout, no_trim=args.no_trim,
                             no_tier=args.no_tier, weighted_loss=args.weighted_loss)

        # Save final model
        out = args.dataset.replace('.csv', '_mlp.pt')
        torch.save({'model_state': model.state_dict(), 'feat_cols': feats}, out)
        print(f"Saved final model: {out}", flush=True)

    finally:
        if tee:
            sys.stdout = tee._stdout
            tee.close()
