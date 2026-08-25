"""
dv_mlp_model.py
===============
Monotone Control Points (MCP) tone-mapping model for Dynamic Tone Mapping.

KEY ARCHITECTURE CHANGES vs. Naka-Rushton (Run 12):

  NR:  Linear(128 -> 2)  → Naka-Rushton(Xn/(Xn+sigman)) → 256-pt curve
  MCP: Linear(128 -> 7)  → softplus → cumsum → 8 y-knots → NCS → 256-pt curve

Why MCP beats NR:
  - NR cannot represent neutral-neutral-boost (30% of data) — MSE 0.003-0.67 on 8/9 cells
  - MCP K=7 achieves MSE < 3e-5 on ALL 9 curve-shape cells
  - Natural cubic spline (C2) gives 7-9x lower d2 discontinuity vs PCHIP → no banding
  - f(0)=0 guaranteed (cumsum starts at 0); monotonicity guaranteed (softplus diffs)
  - Zero mono violations observed across all 9 cell types in analytical fit study

Architecture:
    [pixel features] → encoder: Linear -> LayerNorm -> SiLU -> Dropout x 2
    -> MCP head: Linear(128->7) -> softplus -> cumsum -> scale -> 8 y-knots
    -> NaturalCubicSpline(x-knots uniform in [0,1]) -> [256-pt curve]
    -> BoundedDTMLoss (envelope hinge + cell-weighted MSE) vs gold curve

Usage:
    python ml/dv_mlp_model.py --dataset train.csv --val-dataset val.csv \\
        --val-titles andor,euphoria,prehistoric,our,wondla \\
        --split-episodes our:E07,E08 \\
        --epochs 100 --dropout 0.3 --use-5x5 --no-trim --no-tier --mcp \\
        --save F:/DTMModelData/ckpt_mcp_run13 \\
        --log F:/DTMModelData/train_log_mcp_run13.txt
"""

import argparse
import math
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar   # kept for legacy CurveLoss path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

# Re-use all data-loading and target logic from dv_coef_model
from dv_coef_model import (
    FEATURE_COLS, FEATURE_COLS_5X5, BASE_FEATURE_COLS, SAT_FEATURE_COLS, DERIVED_SAT_COLS,
    SPLINE_KNOT_COLS, SPLINE_KNOT12_COLS, K12_INDICES,
    FEATURE_COLS_PRUNED, DERIVED_FEAT_COLS,
    FEATURE_SET, TARGET_DIM, MAX_SEGS, MAX_PIVOTS,
    COEF_SCALE, INPUT_MAX,
    row_to_target, target_to_rpu, eval_rpu, _sanitise_rpu,
    load_data, add_derived_sat,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CURVE_PTS = 256           # evaluation resolution for loss and gold curves


def steffen_interp(x_knots: torch.Tensor,
                   y_knots: torch.Tensor,
                   x_eval:  torch.Tensor) -> torch.Tensor:
    """
    Steffen (1990) monotone cubic Hermite interpolation.

    Replaces the Natural Cubic Spline (NCS) for tone-mapping curves.
    NCS is C2 but can oscillate between knots even for monotone data.
    Steffen is C1 and *guaranteed monotone* whenever y_knots are non-decreasing,
    eliminating dips, overshoots and kinks that NCS can introduce.

    Args:
        x_knots : [n_knots]          fixed knot x-positions (uniform [0,1])
        y_knots : [B, n_knots]       predicted knot y-values (non-decreasing)
        x_eval  : [n_eval]           evaluation points (uniform [0,1])
    Returns:
        curve   : [B, n_eval]        monotone C1 interpolated curve
    """
    n     = x_knots.shape[0]
    h     = x_knots[1:] - x_knots[:-1]                       # [n-1] (uniform: all equal)

    # Secant slopes [B, n-1]
    s = (y_knots[:, 1:] - y_knots[:, :-1]) / h.unsqueeze(0)

    # Interior derivative estimates: weighted average of adjacent secant slopes [B, n-2]
    p = (s[:, :-1] * h[1:] + s[:, 1:] * h[:-1]) / (h[:-1] + h[1:])

    # Steffen monotonicity clamp: m = min(|p|, 2|s[i-1]|, 2|s[i]|) * sign(p)
    # For tone-mapping curves s >= 0, so sign(p) >= 0 — the abs() handles zero-slope.
    m_int = torch.minimum(
        torch.minimum(torch.abs(p), 2.0 * s[:, :-1]),
        2.0 * s[:, 1:]
    ) * torch.sign(p + 1e-30)                                 # [B, n-2]

    # Endpoint derivatives: match the adjacent secant slope
    m = torch.cat([s[:, :1], m_int, s[:, -1:]], dim=1)       # [B, n]

    # Locate each eval point's segment (uniform knots → closed-form)
    seg   = torch.clamp((x_eval * (n - 1)).long(), 0, n - 2) # [n_eval]
    x0    = x_knots[seg]                                      # [n_eval]
    h_seg = h[seg]                                            # [n_eval]
    t     = (x_eval - x0) / h_seg                            # [n_eval] in [0,1]

    t2, t3 = t * t, t * t * t
    h00 =  2*t3 - 3*t2 + 1   # [n_eval]
    h10 =    t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 =    t3 -   t2

    y0 = y_knots[:, seg]      # [B, n_eval]
    y1 = y_knots[:, seg + 1]
    m0 = m[:, seg]
    m1 = m[:, seg + 1]

    return h00 * y0 + (h10 * h_seg) * m0 + h01 * y1 + (h11 * h_seg) * m1

# ── MCP (Monotone Control Points) constants ──
# K free raw outputs → K positive diffs via softplus → cumsum → K+1 y-knots
# x-knots fixed at linspace(0, 1, K+1); y[0]=0 (black anchor)
# Interpolated with natural cubic spline (C2, no banding)
MCP_K_SHAPE   = 7      # shape params: 7 diffs → normalized cumsum in [0,1]
MCP_K_SCALE   = 1      # scale param: y_max = softplus(raw) + 0.05
MCP_K         = MCP_K_SHAPE + MCP_K_SCALE   # 8 total raw outputs
MCP_N_KNOTS   = MCP_K_SHAPE + 1   # 8 knots: y[0]=0 + 7 shape-defined values

# ── BoundedDTMLoss hyperparameters ──
ENVELOPE_INSIDE  = 1.0      # standard weight for curve learning inside safe pocket
ENVELOPE_OUTSIDE = 10.0     # severe penalty for escaping the envelope

# ── Cell-weighted loss defaults ──
# Rare/under-represented cells get higher weight to force exploration
CELL_WEIGHTS = {
    # Identity anchor — strong weight to prevent over-compression
    "neutral-crush-crush":   4.0,
    # Mid-tone expansion cells: raised to 10× (was 4-6×) to force model to learn
    # the characteristic knee-with-plateau S-curve shape for expansion content.
    # These cells have skin tones and highlights that the model was systematically
    # undercutting due to ~500:1 effective training ratio vs neutral-crush-crush.
    "neutral-neutral-boost": 10.0,
    "boost-boost-boost":     10.0,
    "boost-boost-neutral":   10.0,
    "boost-boost-crush":     10.0,
    # Moderate weight for other cells
    "boost-neutral-crush":   4.0,
    "neutral-neutral-crush": 2.0,
}
DEFAULT_CELL_WEIGHT = 1.0

# ── Cell weight label column in dataset ──
CELL_LABEL_COL = "cell"

# Cell index mapping for auxiliary classification head
CELL_CLASSES = [
    "neutral-crush-crush", "neutral-neutral-boost", "boost-neutral-crush",
    "boost-crush-crush",   "neutral-neutral-crush", "crush-crush-crush",
    "boost-boost-crush",   "boost-boost-boost",     "boost-boost-neutral",
]
CELL_TO_IDX = {c: i for i, c in enumerate(CELL_CLASSES)}
N_CELL_CLASSES = len(CELL_CLASSES)
CELL_AUX_ALPHA = 0.2   # weight of auxiliary cell loss relative to curve loss

# ── Display tier constants (used by DVPolyMLP constructor defaults) ──
DISPLAY_NITS   = [143, 1030, 1669]
NITS_TO_IDX    = {143: 0, 1030: 1, 1669: 2}
DISPLAY_PQ     = [2081, 2851, 3079]
NUM_TIERS      = len(DISPLAY_NITS)

# ── Feature bar columns ──
BAR_FEATURE_COLS = ["top_bar_norm", "bottom_bar_norm"]

# ── Trim parameters ──
TRIM_PARAMS    = ['trim_slope', 'trim_offset', 'trim_power']
TRIM_DIM       = len(TRIM_PARAMS)
TRIM_IDENTITY  = 2048.0
TRIM_DIR       = Path("F:/DTMModelData/trims")


# ---------------------------------------------------------------------------
# Monotone Control Points curve evaluator (differentiable, C2 output)
# ---------------------------------------------------------------------------
class MonotoneControlPoints(nn.Module):
    """
    Evaluates a monotone curve at N_CURVE_PTS points via natural cubic spline.

    Parameterisation (MCP_K=7 free params → MCP_N_KNOTS=8 knots):
        raw    : [B, MCP_K]   raw logits from model head
        diffs  : softplus(raw)              all positive  [B, MCP_K]
        y[0]   : 0.0                        black anchor  (fixed)
        y[1:8] : cumsum(diffs)              monotone      [B, MCP_K]
        y      : [y[0], y[1:8]]             8 knot values [B, MCP_N_KNOTS]

    x-knots are fixed at linspace(0, 1, MCP_N_KNOTS).
    Interpolation: natural cubic spline (C2 — no kinks, no banding).

    Guarantees:
      - f(0) = 0  (black pinned to black, no letterbox lifting)
      - Strictly monotone  (softplus diffs > 0 → cumsum strictly increasing)
      - C2 continuous output  (natural cubic spline → smooth second derivative)
      - Covers all 9 observed curve-shape cells  (MSE < 3e-5 across all cells)
    """

    def __init__(self, n_pts=N_CURVE_PTS, k=MCP_K_SHAPE):
        super().__init__()
        self.n_pts   = n_pts
        self.k       = k
        self.n_knots = k + 1   # MCP_N_KNOTS = MCP_K_SHAPE + 1 = 8

        # Fixed x-knot positions (uniform)
        x_knots = torch.linspace(0.0, 1.0, self.n_knots)   # [n_knots]
        x_eval  = torch.linspace(0.0, 1.0, n_pts)           # [n_pts]

        self.register_buffer('x_knots', x_knots)
        self.register_buffer('x_eval',  x_eval)

        # Pre-compute natural cubic spline basis: A such that y_dense = A @ y_knots
        # We build the NCS coefficient matrix analytically once at init.
        self.register_buffer('_ncs_A', self._build_ncs_matrix(x_knots, x_eval))

    @staticmethod
    def _build_ncs_matrix(x_knots, x_eval):
        """
        Build [n_pts, n_knots] matrix A so that:
            y_dense = A @ y_knots
        for a natural cubic spline through (x_knots, y_knots).

        Uses the standard tridiagonal system for natural spline second derivatives,
        then evaluates the cubic Hermite basis at each x_eval point.
        """
        n = len(x_knots)
        m = len(x_eval)
        xk = x_knots.numpy()
        xe = x_eval.numpy()
        h  = np.diff(xk)                    # (n-1,) interval widths

        # Build tridiagonal system for second derivatives M (natural BC: M[0]=M[-1]=0)
        # 6*(y[i+1]-y[i])/h[i] - 6*(y[i]-y[i-1])/h[i-1]  = rhs[i-1..n-3]
        # Solve: [diag, off] @ M[1:-1] = rhs @ y_knots
        # → express M as a linear function of y_knots
        size = n - 2
        diag = 2.0 * (h[:-1] + h[1:])
        off  = h[1:-1]

        # tridiagonal matrix T (size x size)
        T = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

        # RHS matrix R (size x n) so that rhs = R @ y_knots
        R = np.zeros((size, n))
        for i in range(size):
            R[i, i]   =  6.0 / h[i]
            R[i, i+1] = -6.0 / h[i] - 6.0 / h[i+1]
            R[i, i+2] =  6.0 / h[i+1]

        # M[1:-1] = T^{-1} R @ y_knots
        T_inv_R = np.linalg.solve(T, R)      # (size x n)

        # Full M (n x n linear map of y_knots); M[0]=M[-1]=0
        M_mat = np.zeros((n, n))
        M_mat[1:-1, :] = T_inv_R

        # Evaluate spline at each x_eval point
        # For x in segment [xk[i], xk[i+1]]:
        #   y = a*y[i] + b*y[i+1] + c*M[i] + d*M[i+1]
        # where a,b,c,d are standard cubic Hermite weights
        A = np.zeros((m, n))
        for j, x in enumerate(xe):
            # Find segment
            i = min(np.searchsorted(xk, x, side='right') - 1, n - 2)
            hi = h[i]
            t  = (x - xk[i]) / hi          # local parameter in [0,1]
            s  = 1.0 - t

            # Cubic basis
            a = s
            b = t
            c = (s**3 - s) * hi**2 / 6.0
            d = (t**3 - t) * hi**2 / 6.0

            A[j, :] += a * (np.arange(n) == i).astype(float)
            A[j, :] += b * (np.arange(n) == i+1).astype(float)
            A[j, :] += c * M_mat[i,   :]
            A[j, :] += d * M_mat[i+1, :]

        return torch.from_numpy(A.astype(np.float32))   # [n_pts, n_knots]

    def forward(self, raw):
        """
        Args:
            raw   : [B, MCP_K]  raw logits from model head
                    raw[:, :MCP_K_SHAPE] → shape (7 diffs, normalized to [0,1])
                    raw[:, MCP_K_SHAPE:] → scale (y_max = softplus + 0.05)
        Returns:
            curve : [B, N_CURVE_PTS]  monotone C2 curve, y[0]=0, bounded
        """
        B = raw.shape[0]

        shape_raw = raw[:, :MCP_K_SHAPE]                            # [B, 7]
        scale_raw = raw[:, MCP_K_SHAPE:]                            # [B, 1]

        # Shape: normalized cumsum in [0, 1]
        diffs = nn.functional.softplus(shape_raw)                   # [B, 7], all > 0
        cumsum = torch.cumsum(diffs, dim=1)                         # [B, 7], monotone
        cumsum_norm = cumsum / (cumsum[:, -1:] + 1e-8)              # [B, 7], last ≈ 1

        # Scale: y_max ∈ [0.05, 1.5]  (tone mapping curves are physically bounded)
        y_max = torch.sigmoid(scale_raw) * 1.45 + 0.05              # [B, 1]

        y_pos   = cumsum_norm * y_max                               # [B, 7], in [0, y_max]
        zeros   = torch.zeros(B, 1, device=raw.device, dtype=raw.dtype)
        y_knots = torch.cat([zeros, y_pos], dim=1)                  # [B, 8], y[0]=0

        # Steffen monotone interpolation — C1, no oscillations between knots
        curve = steffen_interp(self.x_knots, y_knots, self.x_eval)  # [B, n_pts]
        return curve


# ---------------------------------------------------------------------------
# DeltaMLP — Predicts correction delta added to spline baseline (Run 31+)
# ---------------------------------------------------------------------------
class DeltaMLP(nn.Module):
    """
    Predicts a per-point correction delta that is ADDED to the spline baseline.

        final_curve = clamp(spline_baseline + delta, 0, 1)  [monotone projected]

    Why this is better than predicting the absolute curve:
    - Model starts at spline (zero-init head → delta=0 → output=spline)
    - Learns ONLY the colorist's creative deviation, not the technical baseline
    - Domain shift from DV to HDR10 is minimised: spline handles the levels,
      model handles the style correction
    - Uncertain/unseen scenes: delta→0 → output=spline (safe fallback)

    Architecture: same encoder as DVPolyMLP, different head.
    """

    def __init__(self, pixel_dim, hidden_dim=128, dropout=0.3,
                 delta_scale=0.35, use_tier_embed=False, has_trim_head=False):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.delta_scale  = delta_scale

        self.encoder = nn.Sequential(
            nn.Linear(pixel_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(hidden_dim, N_CURVE_PTS)

        # Zero-init: model starts at zero correction → output = spline on epoch 1
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, x, tiers=None, spline_baseline=None):
        """
        Args:
            x:               [B, pixel_dim] normalised features
            spline_baseline: [B, 256] maxscl-based spline curve, or None
        Returns:
            (output, None, None) — compatible with DVPolyMLP interface
        """
        h = self.encoder(x)
        delta = torch.tanh(self.delta_head(h)) * self.delta_scale   # [B, 256] ∈ [-scale, +scale]

        if spline_baseline is not None:
            out = torch.clamp(spline_baseline + delta, 0.0, 1.0)
            out, _ = torch.cummax(out, dim=1)   # monotone projection
        else:
            out = delta

        return out, None, None

    def mcp_eval(self, x):
        """Compatibility stub — DeltaMLP doesn't use MCP; returns x unchanged."""
        return x


# ---------------------------------------------------------------------------
# ResidualMCP — Residual correction on top of libplacebo spline baseline
# ---------------------------------------------------------------------------
class ResidualMCP(nn.Module):
    """
    Predicts signed corrections on top of the libplacebo spline knot values.

    Parameterisation:
        raw         : [B, 8]   raw logits from model head
        corrections : tanh(raw) * SCALE   bounded to ±SCALE PQ
        corrections[:, 0] = 0            black anchor always preserved
        y_knots     : clamp(spline_k + corrections, 0, 1)
        y_knots     : cummax(y_knots, dim=1)  monotone projection
        curve       : NCS(x_knots, y_knots) at 256 pts

    Inductive bias: zero corrections → output = spline baseline.
    For neutral-crush-crush, corrections ≈ 0 (easy).
    For neutral-neutral-boost, large positive corrections (consistent signal).
    """
    CORRECTION_SCALE = 0.20  # max ±0.20 PQ ≈ ±300 nits of correction

    def __init__(self, n_pts=N_CURVE_PTS):
        super().__init__()
        self.n_pts   = n_pts
        self.n_knots = MCP_N_KNOTS  # 8

        x_knots = torch.linspace(0.0, 1.0, self.n_knots)
        x_eval  = torch.linspace(0.0, 1.0, n_pts)
        self.register_buffer('x_knots', x_knots)
        self.register_buffer('x_eval',  x_eval)
        # Reuse the NCS matrix builder from MonotoneControlPoints
        self.register_buffer('_ncs_A',
            MonotoneControlPoints._build_ncs_matrix(x_knots, x_eval))

    def forward(self, raw, spline_k):
        """
        Args:
            raw      : [B, 8]   raw logits (MCP head output)
            spline_k : [B, 8]   spline sampled at knot positions (from input features)
        Returns:
            curve    : [B, N_CURVE_PTS]
        """
        corrections = torch.tanh(raw) * self.CORRECTION_SCALE  # [B, 8], bounded
        # Preserve black anchor: never shift y[0]
        zero_col = torch.zeros_like(corrections[:, :1])
        corrections = torch.cat([zero_col, corrections[:, 1:]], dim=1)

        y_knots = spline_k + corrections                        # [B, 8]
        y_knots = torch.clamp(y_knots, 0.0, 1.0)

        # Monotone projection: each knot ≥ previous (cumulative max)
        y_knots, _ = torch.cummax(y_knots, dim=1)

        curve = steffen_interp(self.x_knots, y_knots, self.x_eval)  # [B, n_pts]
        return curve


# ---------------------------------------------------------------------------
# ResidualL1Loss — MSE(correction, target) + non-uniform L1 sparsity
# ---------------------------------------------------------------------------
class ResidualL1Loss(nn.Module):
    """
    Asymmetric L1 loss for ResidualL1MCP training.

    L = MSE(correction, target_correction)
      + (beta_pos * relu(correction) + beta_neg * relu(-correction)).mean()

    beta_pos[i]: penalty for POSITIVE corrections at knot i (expanding beyond spline)
    beta_neg[i]: penalty for NEGATIVE corrections at knot i (compressing below spline)

    At highlight knots (k8-k11): beta_pos >> beta_neg
      → large positive corrections (expansion at highlights) are expensive
      → model only expands highlights when MSE benefit is strong (true expansion content)
      → compression content naturally falls back to spline (corrections stay ≤ 0)

    Both are data-derived from std(gold - spline) per knot, then beta_pos is scaled
    by highlight_pos_mult at highlight knots.
    """
    def __init__(self, beta_pos: np.ndarray, beta_neg: np.ndarray):
        super().__init__()
        self.register_buffer('beta_pos', torch.tensor(beta_pos, dtype=torch.float32))
        self.register_buffer('beta_neg', torch.tensor(beta_neg, dtype=torch.float32))

    def forward(self, corrections, target_corrections):
        """
        corrections       : [B, K]  model predictions
        target_corrections: [B, K]  gold_at_knots - spline_at_knots
        """
        mse = ((corrections - target_corrections) ** 2).mean()
        l1  = (torch.relu( corrections) * self.beta_pos +
               torch.relu(-corrections) * self.beta_neg).mean()
        return mse + l1


# ---------------------------------------------------------------------------
# ResidualL1MCP — K=12 residual corrections with zero-init and organic fallback
# ---------------------------------------------------------------------------
class ResidualL1MCP(nn.Module):
    """
    Predicts unconstrained corrections on top of K=12 spline knot values.
    Zero-initialized correction head → model starts at spline, deviates only when needed.

    final_knots = cummax(clamp(spline_q + correction, 0, 1))
    correction[0] = 0  (black anchor preserved)
    """
    def __init__(self, n_pts=N_CURVE_PTS, k=12):
        super().__init__()
        self.n_pts   = n_pts
        self.n_knots = k

        x_knots = torch.linspace(0.0, 1.0, k)
        x_eval  = torch.linspace(0.0, 1.0, n_pts)
        self.register_buffer('x_knots', x_knots)
        self.register_buffer('x_eval',  x_eval)
        self.register_buffer('_ncs_A',
            MonotoneControlPoints._build_ncs_matrix(x_knots, x_eval))

    def forward(self, raw_correction, spline_q):
        """
        raw_correction : [B, K]  unconstrained output from correction head
        spline_q       : [B, K]  spline sampled at K knot positions (from features)
        Returns: curve [B, N_CURVE_PTS]
        """
        # Reconstruct correction out-of-place to avoid autograd inplace errors.
        # Ceiling rule: at x >= 0.635 (k7-k11), spline is already at/near 0.544.
        # Any positive correction there pushes above the 143-nit display ceiling.
        # Clamp all highlight corrections to ≤ 0 for all content types.
        TARGET_PQ = 0.5444   # nits_to_pq(143) ≈ 0.5444

        # Only hard-clamp k11 (x=1.0): mathematically correct since display ceiling
        # is 143 nits = TARGET_PQ and both gold/spline end there.
        # k0..k10 remain free — expansion learning preserved.
        k11_clamped = raw_correction[:, -1:].clamp(max=0.0)  # k11 ≤ 0
        correction = torch.cat([
            torch.zeros_like(raw_correction[:, :1]),  # k0 = 0 (black anchor)
            raw_correction[:, 1:-1],                  # k1..k10 free
            k11_clamped,                              # k11 ≤ 0 (ceiling anchor)
        ], dim=1)

        y_knots = spline_q + correction
        y_knots = torch.clamp(y_knots, 0.0, 1.0)
        y_knots, _ = torch.cummax(y_knots, dim=1)  # monotone projection

        curve = y_knots @ self._ncs_A.T
        curve = torch.clamp(curve, min=0.0)
        return curve, correction  # return correction for loss computation


# ---------------------------------------------------------------------------
# BoundedDTMLoss — Envelope Hinge + Cell-Weighted MSE
# ---------------------------------------------------------------------------
class BoundedDTMLoss(nn.Module):
    """
    Computes loss for the Naka-Rushton model with bounded envelope constraint.

    Envelope boundary per evaluation point:
        lower_bound = min(gold, spline_baseline)
        upper_bound = max(gold, spline_baseline)

    Loss composition:
        1. MSE(gold, predicted) — standard learning loss (inside envelope)
        2. Hinge penalty — if predicted escapes envelope, severe penalty

    Cell weighting:
        Rare cells (boost-boost-boost, etc.) get higher weight to force exploration.
    """

    def __init__(self, lambda_inside=ENVELOPE_INSIDE, lambda_outside=ENVELOPE_OUTSIDE):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lambda_inside = lambda_inside
        self.lambda_outside = lambda_outside

    def forward(self, predicted_curves, dolby_targets, spline_baselines=None,
                cell_labels=None, cell_weights=None, curve_weights=None):
        """
        Args:
            predicted_curves : [B, 256]  MCP curve output
            dolby_targets    : [B, 256]  ground truth gold curve
            spline_baselines : [B, 256]  libplacebo spline baseline (or None → bounds = gold)
            cell_labels      : List[str] per-sample cell label (for weighted loss)
            cell_weights     : Optional Tensor[B] — pre-computed per-sample weights
            curve_weights    : Optional Tensor[B, 256] — per-point content-aware weights
                               (from compute_scene_curve_weights; sum-to-1 per row)
        Returns:
            total_loss : scalar
        """
        B = predicted_curves.shape[0]

        # 1. Envelope boundaries
        if spline_baselines is not None:
            lower_bound = torch.min(dolby_targets, spline_baselines)
            upper_bound = torch.max(dolby_targets, spline_baselines)
        else:
            lower_bound = dolby_targets
            upper_bound = dolby_targets

        # 2. MSE vs gold — content-weighted + near-black shadow boost
        # Shadow boost: 1 + 9×exp(-x/0.05) → 10× penalty at x=0, tapers to ~1× by x=0.2 PQ
        # Fixes: MSE blind spot where tiny absolute errors near black cause perceptual black crush
        xs_curve = torch.linspace(0, 1, predicted_curves.shape[1],
                                   device=predicted_curves.device)           # [256]
        shadow_boost = 1.0 + 9.0 * torch.exp(-xs_curve / 0.05)              # [256]

        sq_err = self.mse(predicted_curves, dolby_targets) * shadow_boost    # [B, 256]
        if curve_weights is not None:
            base_mse = (sq_err * curve_weights).sum(dim=1)
        else:
            base_mse = sq_err.mean(dim=1)

        # 3. Perceptually-adaptive hinge penalty
        # Base: scale λ by 1/(lower+ε) — brick wall near black, tapers at midtones.
        # Mid-tone Gaussian bump: extra penalty centred at x=0.20 PQ (skin tone zone).
        # Prevents model from systematically undercutting the corridor where skin tones live.
        # Gaussian: 1 + 8×exp(-(x-0.20)²/0.008) → peaks at x=0.20 (9× boost), ±0.09 PQ FWHM
        xs_curve_h = torch.linspace(0, 1, predicted_curves.shape[1],
                                     device=predicted_curves.device)           # [256]
        midtone_boost = 1.0 + 8.0 * torch.exp(-(xs_curve_h - 0.20)**2 / 0.008)  # [256]
        # Clamp lower_bound to >=0 before division: gold curves can be slightly negative
        # (RPU polynomial floating-point artefacts). Without clamp, the denominator goes
        # negative → perceptual_lambda < 0 → negative loss → reward hacking.
        _safe_denom = torch.clamp(lower_bound, min=0.0) + 1e-2                 # always > 0
        perceptual_lambda = (self.lambda_outside / _safe_denom) * midtone_boost
        under_shoot = torch.clamp(lower_bound - predicted_curves, min=0.0)
        over_shoot  = torch.clamp(predicted_curves - upper_bound, min=0.0)
        # Both penalties use _safe_denom to prevent sign flip when gold < -1e-2
        lower_penalty = (perceptual_lambda * under_shoot).mean(dim=1)
        upper_penalty = ((self.lambda_outside / _safe_denom) * over_shoot).mean(dim=1)
        boundary_violation = lower_penalty + upper_penalty

        # 4. Composite loss
        total_sample = (self.lambda_inside * base_mse) + boundary_violation

        # 5. Cell weighting
        if cell_weights is not None:
            total_sample = total_sample * cell_weights

        elif cell_labels is not None:
            w = torch.ones_like(total_sample)
            for label, weight in cell_weights_map.items():
                mask = torch.tensor([l == label for l in cell_labels],
                                    device=total_sample.device)
                w = torch.where(mask, torch.full_like(w, weight), w)
            total_sample = total_sample * w

        return total_sample.mean()

# Aliased for use in cell weighting within forward
cell_weights_map = CELL_WEIGHTS


# ---------------------------------------------------------------------------
# Content-aware curve weights
# ---------------------------------------------------------------------------
def compute_scene_curve_weights(df, n_pts=N_CURVE_PTS):
    """
    Per-scene curve weights: uniform within the scene's active HDR range
    [0, l1_max_pq/4095], zero above it (no content reaches there).

    Rationale: curve error above l1_max_pq is irrelevant — the tone curve
    is identity there and both gold & spline agree.  Focusing loss on the
    active range avoids wasting gradient budget on the high-peak tail.

    Returns float32 [N, n_pts], each row sums to 1.
    """
    xs = np.linspace(0, 1, n_pts, dtype=np.float32)           # [n_pts]
    raw = df['l1_max_pq'].values.astype(np.float32)
    # l1_max_pq may be raw (0-4095) or already normalised (0-1) depending on caller.
    # load_data() normalises on read; raw CSVs do not.  Detect by magnitude.
    l1_max = raw / 4095.0 if raw.max() > 2.0 else raw          # [N] in [0,1]

    # Soft sigmoid taper at l1_max_pq:
    #   weight ≈ 1.0 below l1_max_pq  (full gradient — content lives here)
    #   weight tapers to ALPHA above   (partial gradient — preserves expansion signal)
    # Hard zero would silence boost-curve learning above the scene peak;
    # the soft floor keeps gradient flowing so neutral-neutral-boost / boost-boost-boost
    # cells can still learn to lift highlights above identity.
    ALPHA = 0.15     # minimum weight above peak  (15% of full)
    WIDTH = 0.05     # sigmoid transition width in PQ units (~75 nits)
    # sigmoid: 1 → 0 centered at l1_max_pq with width WIDTH
    z = (xs[np.newaxis, :] - l1_max[:, np.newaxis]) / WIDTH    # [N, n_pts]
    w = 1.0 / (1.0 + np.exp(z))                                # sigmoid, 1 below / 0 above
    weights = (w * (1.0 - ALPHA) + ALPHA).astype(np.float32)   # [N, n_pts], in [ALPHA, 1]

    row_sums = weights.sum(axis=1, keepdims=True)
    return (weights / np.maximum(row_sums, 1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# RPU helpers for spline baseline computation
# ---------------------------------------------------------------------------
def target_to_rpu_nr(gold_dev):
    """
    Convert a gold curve deviation to a simple 2-pivot RPU for spline baseline.
    Creates a piecewise linear curve with one pivot at the average deviation point.

    At inference, we cannot compute this (no gold curve). For now, we use it
    only during training to define the envelope bounds.
    """
    # For the envelope: we need a deterministic spline-like curve
    # Use a simple identity + small compression as the "spline baseline"
    # This is a conservative baseline that always compresses slightly
    pivot_val = float(gold_dev) if isinstance(gold_dev, (int, float, np.floating)) else float(gold_dev.mean())
    return {
        'pivots': [0.0, 0.5, 1.0],
        'seg_orders': [1, 1],
        'seg_c0': [0.0, pivot_val * 0.1],
        'seg_c1': [1.0, 1.0],
        'seg_c2': [0.0, 0.0],
    }


_SPLINE_COLS = [f"spline_y_{i}" for i in range(N_CURVE_PTS)]


def load_spline_baseline(row) -> "np.ndarray | None":
    """
    Load pre-computed libplacebo pl_tone_map_spline curve from dataset row.

    The spline columns (spline_y_0..spline_y_255) are added to Stage 1 CSVs
    by tools/gen_spline_baselines.py, which runs libplacebo_baseline_eval.exe
    using l1_max_pq / l1_avg_pq from RPU metadata (NOT GPU-derived maxscl).

    Returns [N_CURVE_PTS] float32 array, or None if columns are missing.
    """
    if _SPLINE_COLS[0] not in row.index:
        return None
    try:
        vals = [float(row[c]) for c in _SPLINE_COLS]
        arr = np.array(vals, dtype=np.float32)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return None
        return arr
    except Exception:
        return None


def fit_mcp_params(gold_curve):
    """
    Fit MCP parameters to a gold curve analytically.

    Parameterisation:
        raw[:MCP_K_SHAPE] = shape logits (7 values)
        raw[MCP_K_SHAPE:]  = scale logit  (1 value)

    Analytical inversion:
        1. Sample gold at MCP_N_KNOTS uniform x-knots → y_knots (monotone)
        2. y_max = y_knots[-1]  (scale)
        3. normalized_y = y_knots[1:] / y_max  (shape, in [0,1])
        4. diffs = diff(normalized_y) with [0] prepended (diff of normalized cumsum)
        5. Invert softplus for shape diffs and scale

    Returns [MCP_K] float32 array: [shape_raw..., scale_raw].
    """
    xs = np.linspace(0.0, 1.0, N_CURVE_PTS)
    x_knots = np.linspace(0.0, 1.0, MCP_N_KNOTS)

    # Sample gold at knot positions, enforce monotone
    y_knots = np.interp(x_knots, xs, gold_curve)
    y_knots[0] = 0.0
    for i in range(1, len(y_knots)):
        y_knots[i] = max(y_knots[i], y_knots[i-1] + 1e-6)

    y_max = max(y_knots[-1], 0.06)

    # Shape: normalized cumsum in [0, 1]
    y_norm = y_knots[1:] / y_max               # [MCP_K_SHAPE], in [0, 1]
    # diffs of normalized cumsum (all positive)
    norm_diffs = np.diff(np.concatenate([[0.0], y_norm]))   # [MCP_K_SHAPE]
    norm_diffs = np.maximum(norm_diffs, 1e-6)
    shape_raw = np.log(np.expm1(np.clip(norm_diffs, 1e-6, 80.0)))  # softplus inverse

    # Scale: invert sigmoid(scale_raw) * 1.45 + 0.05 = y_max
    # sigmoid(x) = (y_max - 0.05) / 1.45  → x = logit(...)
    s = np.clip((y_max - 0.05) / 1.45, 1e-6, 1 - 1e-6)
    scale_raw = np.log(s / (1.0 - s))

    return np.concatenate([shape_raw, [scale_raw]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Training augmentation — Blu-ray grain / compression domain shift
# ---------------------------------------------------------------------------
PIXEL_FEAT_END = 27   # indices 0-26 are pixel features; 27-28 are bar offsets


def augment_bluray_grain(pixel_features: "torch.Tensor",
                          p: float = 0.5) -> "torch.Tensor":
    """
    Simulate Blu-ray film grain / compression domain shift during training.

    Only applied when grad is enabled (training mode); bypassed at validation/inference.
    """
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


# ---------------------------------------------------------------------------
# Dataset — MCP output
# ---------------------------------------------------------------------------
class DVNRDataset(Dataset):
    """
    One sample = one training row.

    Returns 7-tuple:
        features      : [pixel_dim]   normalised ICtCp-I pixel statistics
        cell_label    : str           cell name (e.g. "neutral-neutral-boost")
        mcp_params    : [MCP_K]       ground truth MCP raw logits fitted to gold curve
        gold_curve    : [N_CURVE_PTS] pre-evaluated 256-pt gold curve
        spline_base   : [N_CURVE_PTS] spline baseline for envelope bounds
        gold_dev      : float         mean(gold - identity) for monitoring
        scene_id      : int           for grouping (split-episodes support)
    """

    def __init__(self, df, feat_cols, tier_col='display_tier',
                 feat_mean=None, feat_std=None, content_loss=False):
        self.feat_cols = feat_cols
        self.df_valid = df.reset_index(drop=True)

        n_total = len(df)
        n_skip  = 0
        has_spline = _SPLINE_COLS[0] in df.columns
        if not has_spline:
            print(f"  WARNING: spline columns missing — envelope loss will use gold as fallback.", flush=True)
            print(f"           Run: python3 tools/gen_spline_baselines.py  to pre-compute.", flush=True)
        print(f"  Building MCP dataset from {n_total:,} rows  spline={'yes' if has_spline else 'NO (fallback)'}...", flush=True)
        t0 = time.time()

        curves   = []
        splines  = []
        mcp_ps   = []
        dev_ids  = []

        for i, (_, row) in enumerate(df.iterrows()):
            crv = precompute_curve(row)
            if crv is None:
                n_skip += 1
                continue

            curves.append(crv)
            dev_ids.append(float((crv - np.linspace(0.0, 1.0, N_CURVE_PTS)).mean()))

            spl = load_spline_baseline(row)
            splines.append(spl if spl is not None else crv)

            mcp_ps.append(fit_mcp_params(crv))

            if (i + 1) % 20000 == 0:
                elapsed = time.time() - t0
                print(f"    {i+1:>6,}/{n_total:,}  valid={len(curves):,}  "
                      f"skip={n_skip}  {(i+1)/elapsed:.0f} rows/s", flush=True)

        if not curves:
            raise ValueError("No valid rows with RPU coefficients found in dataset.")

        self.curves     = np.vstack(curves).astype(np.float32)     # [N, 256]
        self.splines    = np.vstack(splines).astype(np.float32)    # [N, 256]
        self.mcp_params = np.vstack(mcp_ps).astype(np.float32)     # [N, MCP_K]
        self.gold_devs  = np.array(dev_ids, dtype=np.float32)      # [N]

        bar_avail = [c for c in BAR_FEATURE_COLS if c in self.df_valid.columns]
        avail     = [c for c in feat_cols if c in self.df_valid.columns]
        missing   = [c for c in feat_cols if c not in self.df_valid.columns]
        if missing:
            print(f"  WARNING: {len(missing)} feature cols missing: {missing}", flush=True)
        all_feat_cols = avail + [c for c in bar_avail if c not in avail]
        X_raw = self.df_valid[all_feat_cols].values.astype(np.float32)

        # Input normalisation: standardize to mean=0, std=1 per feature.
        # Applied at init; stats saved for checkpoint/inference reuse.
        if feat_mean is None:
            self.feat_mean = X_raw.mean(axis=0)
            self.feat_std  = X_raw.std(axis=0) + 1e-8
        else:
            self.feat_mean = feat_mean
            self.feat_std  = feat_std
        self.X = (X_raw - self.feat_mean) / self.feat_std
        self.feat_cols_used = all_feat_cols

        self.tiers = self.df_valid[tier_col].values.astype(np.int64) \
                     if tier_col in self.df_valid.columns \
                     else np.zeros(len(df), dtype=np.int64)

        self.cell_labels = self.df_valid[CELL_LABEL_COL].values \
                           if CELL_LABEL_COL in self.df_valid.columns \
                           else np.array(["neutral-crush-crush"] * len(df))

        self.groups = self.df_valid['scene_id'].values \
                      if 'scene_id' in self.df_valid.columns \
                      else np.arange(len(df))

        # Raw (un-normalised) spline_q values for ResidualL1 correction computation
        if all(c in self.df_valid.columns for c in SPLINE_KNOT12_COLS):
            self.spline_q_raw = self.df_valid[SPLINE_KNOT12_COLS].values.astype(np.float32)
        else:
            self.spline_q_raw = None

        # Content-aware curve weights: uniform within [0, l1_max_pq], zero above
        if content_loss and 'l1_max_pq' in self.df_valid.columns:
            print(f"  Computing content-aware curve weights (active-range masking)...", flush=True)
            self.curve_weights = compute_scene_curve_weights(self.df_valid)
            print(f"  curve_weights shape: {self.curve_weights.shape}  "
                  f"mean active pts: {(self.curve_weights > 0).sum(axis=1).mean():.1f}/256", flush=True)
        else:
            self.curve_weights = np.full(
                (len(self.X), N_CURVE_PTS), 1.0 / N_CURVE_PTS, dtype=np.float32
            )

        print(f"  Dataset built: {len(curves):,} valid  {n_skip} skipped  "
              f"({time.time()-t0:.1f}s)", flush=True)
        print(f"  Feat cols: {len(all_feat_cols)}  MCP params: {MCP_K}  "
              f"Knots: {MCP_N_KNOTS} at uniform x  Interp: NCS (C2)", flush=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        spline_q = (torch.from_numpy(self.spline_q_raw[idx])
                    if self.spline_q_raw is not None
                    else torch.zeros(12, dtype=torch.float32))
        return (
            torch.from_numpy(self.X[idx]),
            str(self.cell_labels[idx] if isinstance(self.cell_labels[idx], str)
                else self.cell_labels[idx].decode('utf-8') if hasattr(self.cell_labels[idx], 'decode')
                else str(self.cell_labels[idx])),
            torch.from_numpy(self.mcp_params[idx]),                  # [MCP_K]
            torch.from_numpy(self.curves[idx]),                      # [256]
            torch.from_numpy(self.splines[idx]),                     # [256]
            torch.tensor(self.gold_devs[idx], dtype=torch.float32),
            torch.tensor(int(self.groups[idx]), dtype=torch.long),
            spline_q,                                                 # [12] raw spline_q
            torch.from_numpy(self.curve_weights[idx]),               # [256] content weights
        )


# ---------------------------------------------------------------------------
# Model — MCP output (MCP_K=7 params → 8 knots → 256-pt C2 curve)
# ---------------------------------------------------------------------------
class DVPolyMLP(nn.Module):
    """
    Predicts MCP_K raw logits from pixel features.
    These are decoded by MonotoneControlPoints into a 256-pt monotone C2 curve.

    Architecture:
      [pixel features] → encoder: Linear -> LayerNorm -> SiLU -> Dropout x 2
      -> MCP head: Linear(hidden_dim -> MCP_K)
      -> MonotoneControlPoints → [256-pt curve]
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
        has_trim_head=False,
    ):
        super().__init__()

        self.use_tier_embed = use_tier_embed
        self.has_trim_head  = has_trim_head

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

        # MCP head — predicts MCP_K raw logits → decoded by MonotoneControlPoints
        # When residual_mcp=True, decoded by ResidualMCP (corrections on spline)
        self._hidden_dim      = hidden_dim
        self.mcp_head         = nn.Linear(hidden_dim, MCP_K)
        self.mcp_eval         = MonotoneControlPoints(k=MCP_K_SHAPE)   # reads current global
        self.residual_mcp_dec = None   # set by set_residual_mode()
        self.cell_head        = None   # set by enable_cell_aux()

        if has_trim_head:
            self.trim_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.SiLU(),
                nn.Linear(32, TRIM_DIM),
            )

    def forward(self, pixel_features, tier_ids):
        """
        Args:
            pixel_features : [B, pixel_dim]  float32
            tier_ids       : [B]              int64
        Returns:
            mcp_raw   : [B, MCP_K]    raw logits (for supervision loss)
            pred_trim : [B, TRIM_DIM] or None
        """
        x_pix = torch.clamp(pixel_features, 0.0, 1.0)

        if self.use_tier_embed:
            x = torch.cat([x_pix, self.embed(tier_ids)], dim=1)
        else:
            x = x_pix

        h = self.encoder(x)
        mcp_raw = self.mcp_head(h)     # [B, MCP_K]

        pred_trim = self.trim_head(h) if self.has_trim_head else None
        cell_logits = self.cell_head(h) if self.cell_head is not None else None
        return mcp_raw, pred_trim, cell_logits

    def enable_cell_aux(self, n_cells: int = 9):
        """
        Add auxiliary cell-type classification head.
        Forces encoder to learn cell-discriminative features alongside curve prediction.
        Loss: BoundedDTMLoss + alpha * CrossEntropy(cell_pred, cell_label)
        """
        self.cell_head = nn.Sequential(
            nn.Linear(self._hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, n_cells),
        )
        print(f"  Cell aux head enabled ({n_cells} classes, hidden={self._hidden_dim})", flush=True)

    def set_residual_mode(self, spline_k_indices: list):
        """
        Enable residual MCP decoding. Call before training with --residual-mcp.
        spline_k_indices: positions of spline_k0..k7 in the feature vector.
        """
        self.residual_mcp_dec  = ResidualMCP()
        self._spline_k_indices = spline_k_indices
        print(f"  ResidualMCP enabled  spline_k at feat indices {spline_k_indices}", flush=True)

    def mcp_eval_residual(self, mcp_raw, pixel_features):
        """Decode MCP raw logits as residual corrections on spline_k features."""
        spline_k = pixel_features[:, self._spline_k_indices]  # [B, 8]
        return self.residual_mcp_dec(mcp_raw, spline_k)

    def predict_curve(self, pixel_features, tier_ids):
        """Convenience: run encoder + MCP head → 256-pt curve."""
        mcp_raw, _ = self.forward(pixel_features, tier_ids)
        if self.residual_mcp_dec is not None:
            return self.mcp_eval_residual(mcp_raw, pixel_features)
        return self.mcp_eval(mcp_raw)  # [B, 256]


# ---------------------------------------------------------------------------
# Train / eval — Two-phase NR training
# ---------------------------------------------------------------------------
def train(df, feat_cols=None, epochs=100, batch_size=64, lr=3e-4, device=None,
          save_path=None, save_every=10, val_df=None, dropout=0.3, no_trim=False,
          no_tier=False, nr=False, envelope=False, phase=1, resume_from=None,
          residual_mcp=False, freq_weights=False,
          cell_aux=False, cell_alpha=0.2,
          residual_l1=False, beta_scale=0.02, highlight_pos_mult=1.0,
          content_loss=False, smooth_gamma=0.0,
          use_delta=False):
    """
    Train DVPolyMLP with Naka-Rushton output.

    Args:
        nr            : Use Naka-Rushton (2-param) instead of piecewise poly (42-param)
        envelope      : Use BoundedDTMLoss (envelope hinge) instead of CurveLoss
        phase         : 1 = train backbone + NR head; 2 = freeze backbone+NR, train trim
        resume_from   : Path to Phase 1 checkpoint for Phase 2
        save_path     : Checkpoint path prefix
        save_every    : Save interval in epochs
    """
    if feat_cols is None:
        feat_cols = FEATURE_COLS if not nr else FEATURE_COLS_5X5

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 72, flush=True)
    print(f"{'Naka-Rushton' if nr else 'Piecewise-Poly'} training", flush=True)
    print(f"  device={device}  epochs={epochs}  batch={batch_size}  lr={lr}", flush=True)
    if nr:
        print(f"  NR curve: Y = Xn/(Xn+sigman)  sigma in (1e-5,inf)  n in (1,4)", flush=True)
        print(f"  Envelope: {'ENABLED' if envelope else 'OFF'}  "
              f"lambda_in={ENVELOPE_INSIDE}  lambda_out={ENVELOPE_OUTSIDE}", flush=True)
    if envelope:
        print(f"  Cell weights: {CELL_WEIGHTS}", flush=True)
    if phase == 2:
        print(f"  Phase 2: freeze backbone+NR, train trim head only", flush=True)
    if resume_from:
        print(f"  Resume from: {resume_from}", flush=True)
    if save_path:
        print(f"  checkpoints: {save_path}_best.pt  (every {save_every} ep)", flush=True)
    print("=" * 72, flush=True)

    # ── Tier handling ──
    if no_tier:
        print(f"  Tier embedding DISABLED — one row per scene", flush=True)
        df = df.copy()
        df['display_tier'] = 0
    else:
        if 'target_nits' not in df.columns:
            df = expand_tiers(df)
        if 'display_tier' not in df.columns:
            df['display_tier'] = df['target_nits'].apply(nits_to_tier)

    if val_df is not None:
        if no_tier:
            val_df = val_df.copy()
            val_df['display_tier'] = 0
        elif 'target_nits' not in val_df.columns:
            val_df = expand_tiers(val_df)

    # ── Build datasets ──
    print(f"\n[1/{3 if nr else 4}/4] Building dataset ({len(df):,} rows) ...", flush=True)
    ds = DVNRDataset(df, feat_cols, content_loss=content_loss) if nr else DVCoefDataset(df, feat_cols)
    pixel_dim = len(ds.feat_cols_used)
    if nr:
        print(f"  Input normalisation: mean=[{ds.feat_mean[:3]}...]  "
              f"std=[{ds.feat_std[:3]}...]", flush=True)

    if val_df is not None:
        print(f"\n[2/{3 if nr else 4}/4] Building val dataset ({len(val_df):,} rows) ...", flush=True)
        # Pass train stats to val so both use same normalisation
        val_ds = (DVNRDataset(val_df, feat_cols,
                              feat_mean=ds.feat_mean if nr else None,
                              feat_std=ds.feat_std  if nr else None,
                              content_loss=content_loss)
                  if nr else DVCoefDataset(val_df, feat_cols))
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(np.arange(len(ds)), groups=ds.groups))
        tr_ds = torch.utils.data.Subset(ds, tr_idx)
        te_ds = torch.utils.data.Subset(ds, te_idx)
        val_ds = None

    # Windows requires spawn for multiprocessing; persistent_workers causes silent
    # crashes with spawn + CUDA. Use num_workers=0 (safe on all platforms).
    # Speed gain from larger batch_size already covers most of the DataLoader overhead.
    _pm = torch.cuda.is_available()
    tr_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=_pm)
    if val_ds is not None:
        te_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=_pm)
    else:
        te_loader = DataLoader(torch.utils.data.Subset(ds, te_idx),
                               batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=_pm)

    print(f"  Train: {len(ds):,}  Val: {len(te_loader.dataset):,}  "
          f"Batches/epoch: {len(tr_loader)}", flush=True)

    # ── Build model ──
    print(f"\n[3/{3 if nr else 4}/4] Building model ...", flush=True)

    if use_delta:
        model = DeltaMLP(pixel_dim=pixel_dim, dropout=dropout).to(device)
        print(f"  DeltaMLP: pixel_dim={pixel_dim}  params={sum(p.numel() for p in model.parameters()):,}  "
              f"delta_scale={model.delta_scale}  zero-initialized (starts at spline)", flush=True)
    else:
        model = DVPolyMLP(
            pixel_dim=pixel_dim,
            dropout=dropout,
            use_tier_embed=(not no_tier),
            has_trim_head=(not no_trim),
        ).to(device)

    # ── ResidualL1 mode: replace standard MCP + BoundedDTMLoss ──────────────
    if residual_l1:
        # Force K=12 globals
        _sys = __import__('sys')
        _mod = _sys.modules[__name__]
        _mod.MCP_K_SHAPE = 11
        _mod.MCP_K       = 12
        _mod.MCP_N_KNOTS = 12

        # Spline_q indices must be in feat_cols
        spline_q_indices = [feat_cols.index(f"spline_q{i}") for i in range(12)]

        # Replace mcp_head with K=12 output + zero-init (model starts at spline)
        model.mcp_head = nn.Linear(model._hidden_dim, 12).to(device)
        nn.init.zeros_(model.mcp_head.weight)
        nn.init.zeros_(model.mcp_head.bias)

        # Attach ResidualL1MCP decoder
        model.residual_l1_dec = ResidualL1MCP(k=12).to(device)

        # Compute data-driven beta_vector from training data
        gold_k12   = ds.curves[:, K12_INDICES]                              # [N, 12]
        spline_k12 = ds.df_valid[SPLINE_KNOT12_COLS].values.astype(np.float32)  # [N, 12] raw

        target_corrections = gold_k12 - spline_k12
        std_per_knot = target_corrections.std(axis=0) + 1e-8   # [12]
        # highlight_pos_mult comes from train() parameter


        beta_neg = (beta_scale / std_per_knot).astype(np.float32)
        beta_pos = beta_neg.copy()
        beta_neg[0] = beta_neg[0] * 5    # black anchor: strong sparsity (symmetric)
        beta_pos[0] = beta_pos[0] * 5
        beta_pos[8:] = beta_pos[8:] * highlight_pos_mult  # extra cost for highlight expansion

        print(f"  ResidualL1 beta (L1 sparsity per knot, highlight_pos_mult={highlight_pos_mult}):", flush=True)
        for i, (s, bp, bn) in enumerate(zip(std_per_knot, beta_pos, beta_neg)):
            mark = " << HIGHLIGHT" if i >= 8 and highlight_pos_mult > 1 else ""
            print(f"    k{i:2d} (x={K12_INDICES[i]/255:.3f}): std={s:.4f}  "
                  f"beta_pos={bp:.4f}  beta_neg={bn:.4f}{mark}", flush=True)

        nr = True   # ensure MCP training path is used

    # Enable auxiliary cell classification head
    if cell_aux:
        model.enable_cell_aux(N_CELL_CLASSES)
        model.cell_head = model.cell_head.to(device)

    # Enable residual MCP if requested — requires spline_k features in feat_cols
    if residual_mcp:
        spline_k_indices = [feat_cols.index(f"spline_k{i}") for i in range(8)]
        model.set_residual_mode(spline_k_indices)
        model.residual_mcp_dec = model.residual_mcp_dec.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    mode_str = "residual-MCP" if residual_mcp else ("MCP" if nr else "disabled")
    print(f"  pixel_dim={pixel_dim}  params={n_params:,}  "
          f"MCP={mode_str}  trim_head={'enabled' if not no_trim else 'disabled'}",
          flush=True)

    # ── Loss function ──
    if residual_l1 and nr:
        rl1_loss_fn = ResidualL1Loss(beta_pos, beta_neg).to(device)
        spline_q_feat_indices = [feat_cols.index(f"spline_q{i}") for i in range(12)]
        print(f"  Loss: ResidualL1Loss  beta_scale={beta_scale}", flush=True)
        loss_fn = None   # BoundedDTMLoss not used in residual_l1 mode
    elif nr:
        loss_fn = BoundedDTMLoss(
            lambda_inside=ENVELOPE_INSIDE,
            lambda_outside=ENVELOPE_OUTSIDE,
        ).to(device)

        # Frequency-proportional cell weights: weight ∝ sqrt(count/total)
        # Common cells dominate (matching real distribution), rare still get gradient.
        if freq_weights and hasattr(ds, 'cell_labels'):
            from collections import Counter
            cell_counts = Counter(ds.cell_labels)
            total_c = sum(cell_counts.values())
            raw = {c: (cnt/total_c)**0.5 for c, cnt in cell_counts.items()}
            max_w = max(raw.values())
            freq_cw = {c: round(w/max_w, 4) for c, w in raw.items()}
            import sys as _sys
            _sys.modules[__name__].cell_weights_map = freq_cw
            print(f"  --freq-weights: {freq_cw}", flush=True)

        print(f"  Loss: BoundedDTMLoss  (envelope hinge + cell-weighted MSE)", flush=True)
    else:
        trim_w = 0.0 if no_trim else 0.05
        loss_fn = CurveLoss(trim_weight=trim_w).to(device)
        print(f"  Loss: CurveLoss (MSE + mono + trim)", flush=True)

    # ── Optimizer and scheduler ──
    if phase == 1:
        params_to_opt = list(model.parameters())
        print(f"  Optimizing ALL parameters (Phase 1)", flush=True)
    else:
        freeze_all_except(model, 'trim_head')
        params_to_opt = list(model.trim_head.parameters())
        print(f"  Optimizing trim_head only (Phase 2, backbone+MCP frozen)", flush=True)

    opt = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # ── Resume from Phase 1 checkpoint ──
    if resume_from and Path(resume_from).exists():
        print(f"\n  Loading Phase 1 checkpoint: {resume_from}", flush=True)
        ckpt = torch.load(resume_from, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        print(f"    Loaded epoch {ckpt.get('epoch', '?')}  val_loss={ckpt.get('val_loss', '?')}", flush=True)

    # ── Training loop ──
    print(f"\n[{4 if nr else 4}/4] Training for {epochs} epochs ...", flush=True)

    best_val, best_state = float('inf'), None
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()

        # ---- Train ----
        model.train()
        tr_loss = 0.0
        tr_grad_norm = 0.0
        n_tr_batches = 0

        for batch in tr_loader:
            if nr:
                feats, cell_labels, mcp_params, gold_curves, spline_baselines, gold_devs, _, spline_q_raw_batch, curve_weights_batch = batch
                feats, gold_curves = feats.to(device), gold_curves.to(device)
                spline_baselines = spline_baselines.to(device) if envelope else None
                curve_w = curve_weights_batch.to(device) if content_loss else None
                tiers_batch = torch.zeros(len(feats), dtype=torch.long, device=device)
                cell_labels_list = list(cell_labels)
            else:
                feats, tiers_batch, targets_42, gold_curves, gold_trims, gold_devs_batch, _ = batch
                feats, tiers_batch = feats.to(device), tiers_batch.to(device)
                gold_curves = gold_curves.to(device)
                cell_labels_list = None

            feats = augment_bluray_grain(feats, p=0.5)
            opt.zero_grad()

            if nr:
                pred_trim = None    # default — only assigned for DVPolyMLP path
                cell_logits = None
                if use_delta:
                    # DeltaMLP: pass spline as baseline, model returns spline+delta
                    pred_curve, _, _ = model(feats, tiers_batch,
                                              spline_baseline=spline_baselines)
                    loss = loss_fn(pred_curve, gold_curves, spline_baselines,
                                   cell_labels_list, curve_weights=curve_w)
                else:
                    mcp_raw, pred_trim, cell_logits = model(feats, tiers_batch)

                    if residual_l1:
                        spline_q_batch = spline_q_raw_batch.to(device)
                        pred_curve, corrections = model.residual_l1_dec(mcp_raw, spline_q_batch)
                        gold_at_k12 = gold_curves[:, K12_INDICES]
                        target_corr = gold_at_k12 - spline_q_batch
                        loss = rl1_loss_fn(corrections, target_corr)
                    elif residual_mcp:
                        pred_curve = model.mcp_eval_residual(mcp_raw, feats)
                        if envelope:
                            loss = loss_fn(pred_curve, gold_curves, spline_baselines, cell_labels_list)
                        else:
                            loss = nn.MSELoss()(pred_curve, gold_curves)
                    else:
                        pred_curve = model.mcp_eval(mcp_raw)
                        if envelope:
                            loss = loss_fn(pred_curve, gold_curves, spline_baselines,
                                           cell_labels_list, curve_weights=curve_w)
                        else:
                            loss = nn.MSELoss()(pred_curve, gold_curves)

                # MCP monotonicity is structurally guaranteed; small penalty for safety
                diffs = pred_curve[:, 1:] - pred_curve[:, :-1]
                loss += torch.mean(torch.clamp(-diffs, min=0.0) ** 2) * 10.0

                # Smoothness regulariser: penalise curvature (second differences)
                if smooth_gamma > 0.0:
                    d2 = pred_curve[:, 2:] - 2.0 * pred_curve[:, 1:-1] + pred_curve[:, :-2]
                    loss += smooth_gamma * (d2 ** 2).mean()

                if pred_trim is not None and not no_trim:
                    loss += nn.L1Loss()(pred_trim, torch.zeros_like(pred_trim)) * 0.05

                # Auxiliary cell classification loss
                if cell_aux and cell_logits is not None:
                    cell_targets = torch.tensor(
                        [CELL_TO_IDX.get(lbl, 0) for lbl in cell_labels_list],
                        dtype=torch.long, device=device)
                    cell_loss = nn.CrossEntropyLoss()(cell_logits, cell_targets)
                    loss = loss + cell_alpha * cell_loss
            else:
                pred_42, pred_trim = model(feats, tiers_batch)
                loss, _, _, _ = loss_fn(pred_42, pred_trim, gold_curves,
                                        batch[4].to(device), None)

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(params_to_opt, 1.0).item()
            opt.step()

            tr_loss      += loss.item() * len(feats)
            tr_grad_norm += grad_norm
            n_tr_batches += 1

        N_tr = len(ds)
        tr_loss /= N_tr
        tr_grad_norm /= n_tr_batches

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        N_val = len(te_loader.dataset)

        with torch.no_grad():
            for batch in te_loader:
                if nr:
                    feats, _, _, gold_curves, spline_baselines, _, _, spline_q_raw_batch, curve_weights_batch = batch
                    feats, gold_curves = feats.to(device), gold_curves.to(device)
                    spline_baselines = spline_baselines.to(device) if envelope else None
                    curve_w = curve_weights_batch.to(device) if content_loss else None
                    tiers_batch = torch.zeros(len(feats), dtype=torch.long, device=device)

                    if use_delta:
                        pred_curve, _, _ = model(feats, tiers_batch,
                                                  spline_baseline=spline_baselines)
                        l = loss_fn(pred_curve, gold_curves, spline_baselines, None,
                                    curve_weights=curve_w)
                    else:
                        mcp_raw, _, cell_logits = model(feats, tiers_batch)
                        if residual_l1:
                            spline_q_b = spline_q_raw_batch.to(device)
                            pred_curve, corrections = model.residual_l1_dec(mcp_raw, spline_q_b)
                            gold_at_k12 = gold_curves[:, K12_INDICES]
                            target_corr = gold_at_k12 - spline_q_b
                            l = rl1_loss_fn(corrections, target_corr)
                        elif residual_mcp:
                            pred_curve = model.mcp_eval_residual(mcp_raw, feats)
                            l = loss_fn(pred_curve, gold_curves, spline_baselines, None,
                                        curve_weights=curve_w) \
                                if envelope else nn.MSELoss()(pred_curve, gold_curves)
                        else:
                            pred_curve = model.mcp_eval(mcp_raw)
                            l = loss_fn(pred_curve, gold_curves, spline_baselines, None,
                                        curve_weights=curve_w) \
                                if envelope else nn.MSELoss()(pred_curve, gold_curves)
                else:
                    feats, tiers_b, targets_42, gold_curves, gold_trims, _, _ = batch
                    feats, tiers_b = feats.to(device), tiers_b.to(device)
                    gold_curves = gold_curves.to(device)
                    p42, pt, _ = model(feats, tiers_b)
                    l, _, _, _ = loss_fn(p42, pt, gold_curves, gold_trims, None)

                val_loss += l.item() * len(feats)

        val_loss /= N_val
        sched.step()

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                torch.save({'epoch': epoch, 'model_state': best_state,
                            'val_loss': best_val, 'feat_cols': ds.feat_cols_used,
                            'mcp': nr, 'envelope': envelope, 'phase': phase,
                            'mcp_k': MCP_K,
                            'feat_mean': ds.feat_mean if hasattr(ds, 'feat_mean') else None,
                            'feat_std':  ds.feat_std  if hasattr(ds, 'feat_std')  else None},
                           f"{save_path}_best.pt")

        best_mark = '  ** BEST **' if is_best else ''
        print(f"\nEp {epoch:3d}/{epochs}  ({time.time()-ep_t0:.1f}s)  "
              f"lr={sched.get_last_lr()[0]:.3e}  grad={tr_grad_norm:.3f}{best_mark}", flush=True)
        print(f"  TRAIN  total={tr_loss:.5f}  VAL  total={val_loss:.5f}", flush=True)

        # Periodic save
        if save_path and epoch % save_every == 0:
            ckpt = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'epoch': epoch, 'model_state': ckpt,
                        'val_loss': val_loss, 'feat_cols': ds.feat_cols_used,
                        'nr': nr, 'envelope': envelope, 'phase': phase},
                       f"{save_path}_ep{epoch:04d}.pt")

    total_mins = (time.time() - train_start) / 60
    print(f"\nTraining complete: {total_mins:.1f} min  best_val={best_val:.5f}", flush=True)

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate MCP curve MAE
    if nr:
        print("\nEvaluating MCP curve MAE (curve space) ...", flush=True)
        mae_list = []
        model.eval()
        with torch.no_grad():
            for batch in te_loader:
                feats, _, _, gold_curves, _, _, _, _sq, _cw = batch
                feats = feats.to(device)
                tiers_b = torch.zeros(len(feats), dtype=torch.long, device=device)
                if use_delta:
                    # DeltaMLP needs spline baseline — load from batch (position 4)
                    _splines_b = _  # position 4 in batch is spline_baselines (unused above)
                    # Re-unpack for clarity:
                    (feats2, _, _, gold_curves2, splines_eval, _, _, _sq2, _cw2) = \
                        (feats, *[None]*8)  # already unpacked above
                    # Pass None spline → returns raw delta (for MAE vs gold delta)
                    pred_curve, _, _ = model(feats, tiers_b, spline_baseline=None)
                    pred_curve = pred_curve.cpu().numpy()
                else:
                    mcp_raw, _, _cell = model(feats, tiers_b)
                    if residual_l1:
                        spline_q_b = _sq.to(device)
                        pred_curve, _ = model.residual_l1_dec(mcp_raw, spline_q_b)
                        pred_curve = pred_curve.cpu().numpy()
                    elif residual_mcp:
                        pred_curve = model.mcp_eval_residual(mcp_raw, feats).cpu().numpy()
                    else:
                        pred_curve = model.mcp_eval(mcp_raw).cpu().numpy()
                gold = gold_curves.cpu().numpy()
                for i in range(len(pred_curve)):
                    mae_list.append(np.mean(np.abs(gold[i] - pred_curve[i])))
        mae = np.mean(mae_list) if mae_list else float('nan')
        print(f"MCP MAE (curve space): {mae:.5f}  ({mae*1500:.1f} nits approx)  "
              f"evaluated: {len(mae_list)} scenes", flush=True)

    return model, ds.feat_cols_used


def freeze_all_except(model, keep_name):
    """Freeze all parameters except those in the named submodule."""
    for name, param in model.named_parameters():
        if keep_name not in name:
            param.requires_grad = False
    print(f"  Frozen: {sum(1 for p in model.parameters() if not p.requires_grad)} params", flush=True)
    print(f"  Active: {sum(1 for p in model.parameters() if p.requires_grad)} params", flush=True)


# ── Legacy helpers (kept for compatibility when nr=False) ──────────────────

def precompute_curve(row, n_pts=N_CURVE_PTS):
    """Evaluate the gold RPU polynomial to a dense curve for use as training target."""
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
# ── Display tier constants (duplicated from module-level for legacy compat) ──
# Defined at top of file — references below are for legacy DVCoefDataset backward compat


def expand_tiers(df):
    """Replicate each scene row once per display tier."""
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
    """Load trim CSV for one episode → dict: {tier_idx -> [slope, offset, power]}."""
    pattern = f"{title_key}_{episode_stem}*.trim.csv"
    matches = list(TRIM_DIR.glob(pattern))
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    lookup = {}
    pq_to_tier = {pq: idx for idx, pq in enumerate(DISPLAY_PQ)}
    for _, row in df.iterrows():
        frame = int(row['frame'])
        pq = int(row['target_max_pq'])
        if pq not in pq_to_tier:
            pq = min(DISPLAY_PQ, key=lambda p: abs(p - pq))
        tier = pq_to_tier[pq]
        vals = np.array([
            row.get('trim_slope',   TRIM_IDENTITY),
            row.get('trim_offset',  TRIM_IDENTITY),
            row.get('trim_power',   TRIM_IDENTITY),
        ], dtype=np.float32)
        if frame not in lookup:
            lookup[frame] = {}
        lookup[frame][tier] = vals
    return lookup


def get_trim_for_frame(trim_lookup, frame_idx, tier_idx):
    """Return trim values for a frame/tier, falling back to identity."""
    identity = np.full(TRIM_DIM, TRIM_IDENTITY, dtype=np.float32)
    if trim_lookup is None:
        return identity
    frame_trims = trim_lookup.get(frame_idx, {})
    return frame_trims.get(tier_idx, identity)


# Legacy DVCoefDataset — kept for backward compatibility when nr=False
class DVCoefDataset(Dataset):
    """Legacy dataset for piecewise-poly training (kept for compatibility)."""

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
            trim_vals = np.array([
                float(row.get(f'trim_{tier_pq}_slope',   TRIM_IDENTITY)),
                float(row.get(f'trim_{tier_pq}_offset',  TRIM_IDENTITY)),
                float(row.get(f'trim_{tier_pq}_power',   TRIM_IDENTITY)),
            ], dtype=np.float32)
            rows.append(row)
            targets.append(t)
            curves.append(crv)
            trims.append(trim_vals)
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"    {i+1:>6,}/{n_total:,}  valid={len(rows):,}  "
                      f"skip={n_skip}  {rate:.0f} rows/s", flush=True)
        if not rows:
            raise ValueError("No valid rows found in dataset.")
        self.df_valid = pd.DataFrame(rows).reset_index(drop=True)
        self.targets  = np.vstack(targets).astype(np.float32)
        self.curves   = np.vstack(curves).astype(np.float32)
        self.trims    = np.vstack(trims).astype(np.float32)
        xs_norm = np.linspace(0.0, 1.0, self.curves.shape[1], dtype=np.float32)
        self.gold_devs = (self.curves - xs_norm[None, :]).mean(axis=1)
        bar_avail = [c for c in BAR_FEATURE_COLS if c in self.df_valid.columns]
        avail = [c for c in feat_cols if c in self.df_valid.columns]
        all_feat_cols = avail + [c for c in bar_avail if c not in avail]
        self.X = self.df_valid[all_feat_cols].values.astype(np.float32)
        self.feat_cols_used = all_feat_cols
        if tier_col in self.df_valid.columns:
            self.tiers = self.df_valid[tier_col].values.astype(np.int64)
        else:
            self.tiers = np.array([nits_to_tier(r.get('target_nits', 143))
                                   for _, r in self.df_valid.iterrows()], dtype=np.int64)
        self.groups = self.df_valid['scene_id'].values if 'scene_id' in self.df_valid.columns \
                      else np.arange(len(self.df_valid))
        print(f"  Dataset built: {len(rows):,} valid  {n_skip} skipped  ({time.time()-t0:.1f}s)", flush=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.tiers[idx], dtype=torch.long),
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.curves[idx]),
            torch.from_numpy(self.trims[idx]),
            torch.tensor(self.gold_devs[idx], dtype=torch.float32),
            torch.tensor(self.groups[idx] if hasattr(self.groups[idx], '__int__') else int(self.groups[idx]),
                         dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict(model, feat_cols, row, device=None, nr=False):
    """Predict curve output for one row."""
    if device is None:
        device = next(model.parameters()).device

    feats = torch.tensor(
        [[float(row.get(f, 0.0)) for f in feat_cols]], dtype=torch.float32
    ).to(device)

    tier_val = row.get('target_nits', 143)
    tier = torch.tensor([nits_to_tier(tier_val)], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        mcp_raw, _, _ = model(feats, tier)
        if nr:
            curve = model.mcp_eval(mcp_raw).cpu().numpy()[0]
            return curve  # [256] MCP curve via NCS
        else:
            return mcp_raw.cpu().numpy()[0]  # [42] legacy output


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
    ap.add_argument('--val-dataset', default=None)
    ap.add_argument('--val-titles',  default=None,
                    help='Comma-separated title keys for held-out val set')
    ap.add_argument('--l1',          default=None)
    ap.add_argument('--epochs',      type=int,   default=100)
    ap.add_argument('--batch-size',  type=int,   default=64)
    ap.add_argument('--lr',          type=float, default=3e-4)
    ap.add_argument('--save',        default=None)
    ap.add_argument('--save-every',  type=int, default=10)
    ap.add_argument('--dropout',     type=float, default=0.3)
    ap.add_argument('--no-trim',     action='store_true',
                    help='Disable trim head — curve-only training')
    ap.add_argument('--no-bar',       action='store_true',
                    help='Exclude bar features (top_bar_norm/bottom_bar_norm) from feature vector')
    ap.add_argument('--spline-feats',  action='store_true',
                    help='Add spline_k0..k7 (libplacebo spline at MCP knot positions) to features')
    ap.add_argument('--residual-mcp',  action='store_true',
                    help='Use ResidualMCP: predict corrections on spline knots (requires --spline-feats)')
    ap.add_argument('--pruned-feats',  action='store_true',
                    help='Use FEATURE_COLS_PRUNED (40 features) from permutation importance analysis')
    ap.add_argument('--derived-feats', action='store_true',
                    help='Append DERIVED_FEAT_COLS (8 inference-compatible derived features)')
    ap.add_argument('--freq-weights',  action='store_true',
                    help='Use frequency-proportional cell weights: common cells weighted higher')
    ap.add_argument('--residual-l1',  action='store_true',
                    help='Use ResidualL1MCP + ResidualL1Loss (organic spline fallback, K=12)')
    ap.add_argument('--beta-scale',       type=float, default=0.02,
                    help='Global L1 sparsity scale for residual corrections (default 0.02)')
    ap.add_argument('--highlight-pos-mult', type=float, default=1.0,
                    help='Extra multiplier on beta_pos at highlight knots k8-k11 (default 1.0, try 4-8)')
    ap.add_argument('--cell-aux',     action='store_true',
                    help='Add auxiliary cell classification head (forces cell-discriminative encoder)')
    ap.add_argument('--cell-alpha',   type=float, default=0.2,
                    help='Weight of auxiliary cell CE loss (default 0.2)')
    ap.add_argument('--mcp-k',        type=int, default=8,
                    help='Number of MCP output params (default 8 = 7 shape + 1 scale)')
    ap.add_argument('--lambda-out',   type=float, default=10.0,
                    help='Envelope hinge penalty for escaping gold/spline bounds (default 10)')
    ap.add_argument('--no-tier',     action='store_true',
                    help='Disable tier expansion/embedding (correct for curve training)')
    ap.add_argument('--use-5x5',     action='store_true',
                    help='Add 5x5 zone features (pixel_dim 29 -> 79)')
    ap.add_argument('--split-episodes', default=None,
                    help='Episode-level partial split: TITLE:VAL_EP1,VAL_EP2')
    ap.add_argument('--log',         default=None)

    # ── MCP mode ──
    ap.add_argument('--mcp',         action='store_true',
                    help='Use Monotone Control Points (MCP_K=7 params, NCS interpolation). '
                         'Covers all 9 curve cells, C2 smooth output, no banding.')
    ap.add_argument('--nr',          action='store_true',
                    help='Alias for --mcp (deprecated name, kept for backward compatibility)')
    ap.add_argument('--envelope',    action='store_true',
                    help='Use BoundedDTMLoss with envelope hinge. '
                         'Penalizes curve escaping [min(gold,spline), max(gold,spline)]')
    ap.add_argument('--delta', action='store_true',
                    help='DeltaMLP: predict correction delta added to spline baseline. '
                         'Model starts at zero correction (output=spline) and learns '
                         'colorist style on top. Fixes domain shift for HDR10 inference.')
    ap.add_argument('--use-maxscl', action='store_true',
                    help='V2 feature set: replace l1_max_pq/l1_avg_pq with maxscl/average_maxrgb '
                         'and use spline_km_* (maxscl-based knots). Makes training consistent '
                         'with HDR10 inference. Retrain from scratch — incompatible with Run14 weights.')
    ap.add_argument('--content-loss', action='store_true',
                    help='Content-aware curve weighting: loss zeroed above l1_max_pq per scene '
                         '(no gradient wasted on unreachable curve region)')
    ap.add_argument('--smooth-gamma', type=float, default=0.0,
                    help='Smoothness regulariser weight: penalises second differences of '
                         'predicted curve (curvature). Try 1e-3 to 1e-2. Default 0 (off).')

    # ── Two-phase training ──
    ap.add_argument('--phase',       type=int,   default=1, choices=[1, 2],
                    help='Training phase: 1=train all, 2=freeze backbone+NR, train trim only')
    ap.add_argument('--resume-from', default=None,
                    help='Path to Phase 1 checkpoint for Phase 2 resume')

    args = ap.parse_args()

    tee = None
    if args.log:
        tee = _Tee(args.log)
        sys.stdout = tee
        print(f"Logging to: {args.log}", flush=True)

    try:
        print(f"Loading dataset: {args.dataset}", flush=True)
        df = load_data(args.dataset, args.l1)
        print(f"  {len(df):,} rows  titles: {sorted(df['title_key'].unique())}", flush=True)

        val_df = None

        if args.val_titles:
            if not args.val_dataset:
                raise ValueError("--val-titles requires --val-dataset")
            print(f"Loading second CSV:  {args.val_dataset}", flush=True)
            df2 = load_data(args.val_dataset)
            print(f"  {len(df2):,} rows  titles: {sorted(df2['title'].unique())}", flush=True)

            all_data   = pd.concat([df, df2], ignore_index=True)
            val_keys   = set(t.strip() for t in args.val_titles.split(','))
            df         = all_data[~all_data['title_key'].isin(val_keys)].reset_index(drop=True)
            val_df     = all_data[ all_data['title_key'].isin(val_keys)].reset_index(drop=True)

            print(f"\nTitle-based split  (val-titles={sorted(val_keys)})", flush=True)
            print(f"  Train: {len(df):,}  Val: {len(val_df):,}", flush=True)

            if args.split_episodes:
                split_title, val_eps_str = args.split_episodes.split(':', 1)
                val_ep_patterns = [p.strip() for p in val_eps_str.split(',')]
                if 'episode' not in df.columns:
                    raise ValueError("--split-episodes requires 'episode' column")
                split_mask = val_df['title_key'] == split_title
                ep_col = val_df.loc[split_mask, 'episode']
                in_val_eps = ep_col.apply(lambda e: any(pat in str(e) for pat in val_ep_patterns))
                move_to_train = val_df[split_mask & ~in_val_eps]
                val_df = val_df[~split_mask | in_val_eps].reset_index(drop=True)
                df     = pd.concat([df, move_to_train], ignore_index=True)
                print(f"  Moved {len(move_to_train):,} '{split_title}' scenes to train")

        elif args.val_dataset:
            print(f"Loading val dataset: {args.val_dataset}", flush=True)
            val_df = load_data(args.val_dataset)
            if 'target_nits' not in val_df.columns:
                val_df = expand_tiers(val_df)

        if args.use_maxscl:
            # V2 feature set: drop l1_max_pq/l1_avg_pq (RPU metadata, not available for HDR10).
            # Use maxscl/average_maxrgb as consistent scene peak — same signal in training and inference.
            # Automatically includes spline_km_* (maxscl-based knots) instead of spline_k_* (l1_max_pq-based).
            from dv_coef_model import (BASE_FEATURE_COLS_V2, SPLINE_KNOT_MAXSCL_COLS,
                                       SAT_FEATURE_COLS_5X5 as _SAT5X5)
            base_v2 = BASE_FEATURE_COLS_V2 + SAT_FEATURE_COLS
            if args.use_5x5:
                base_v2 = base_v2 + _SAT5X5
            feat_cols = base_v2 + SPLINE_KNOT_MAXSCL_COLS
            print(f"  --use-maxscl: V2 features ({len(feat_cols)} total) — "
                  f"l1_max_pq/l1_avg_pq replaced by maxscl/average_maxrgb; "
                  f"spline_km_* (maxscl-based) included", flush=True)
        elif args.pruned_feats:
            feat_cols = FEATURE_COLS_PRUNED
            print(f"  --pruned-feats: using {len(feat_cols)}-feature pruned set", flush=True)
        else:
            feat_cols = FEATURE_COLS_5X5 if args.use_5x5 else FEATURE_COLS
        if not args.use_maxscl:
            if args.no_bar:
                import sys as _sys
                _mod = _sys.modules[__name__]
                _mod.BAR_FEATURE_COLS = []
                print("  --no-bar: bar features excluded from feature vector", flush=True)
            if getattr(args, 'derived_feats', False):
                feat_cols = feat_cols + DERIVED_FEAT_COLS
                print(f"  --derived-feats: added {len(DERIVED_FEAT_COLS)} derived features "
                      f"({len(feat_cols)} total)", flush=True)
            if args.spline_feats and not args.pruned_feats:
                feat_cols = feat_cols + SPLINE_KNOT_COLS
                print(f"  --spline-feats: added {len(SPLINE_KNOT_COLS)} spline knot features "
                      f"({len(feat_cols)} total)", flush=True)

        # Override MCP_K and ENVELOPE_OUTSIDE from CLI
        import sys as _sys
        _mod = _sys.modules[__name__]
        if args.mcp_k != 8:
            _mod.MCP_K_SHAPE = args.mcp_k - 1
            _mod.MCP_K       = args.mcp_k
            _mod.MCP_N_KNOTS = args.mcp_k
            print(f"  --mcp-k {args.mcp_k}: MCP_K_SHAPE={args.mcp_k-1}  knots={args.mcp_k}", flush=True)
        if args.lambda_out != 10.0:
            _mod.ENVELOPE_OUTSIDE = args.lambda_out
            print(f"  --lambda-out {args.lambda_out}: envelope hinge strength updated", flush=True)

        if args.residual_l1:
            # residual-l1 always needs spline_q features
            if not any('spline_q' in c for c in feat_cols):
                feat_cols = feat_cols + SPLINE_KNOT12_COLS
                print(f"  --residual-l1: auto-added spline_q0..11 ({len(feat_cols)} total feats)", flush=True)

        if args.residual_mcp and not args.spline_feats:
            print("WARNING: --residual-mcp requires --spline-feats. Adding automatically.", flush=True)
            feat_cols = feat_cols + SPLINE_KNOT_COLS

        model, feats = train(
            df, feat_cols=feat_cols,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            save_path=args.save, save_every=args.save_every, val_df=val_df,
            dropout=args.dropout, no_trim=args.no_trim,
            no_tier=args.no_tier, nr=(args.mcp or args.nr), envelope=args.envelope,
            phase=args.phase, resume_from=args.resume_from,
            residual_mcp=args.residual_mcp,
            freq_weights=args.freq_weights,
            cell_aux=args.cell_aux,
            cell_alpha=args.cell_alpha,
            residual_l1=args.residual_l1,
            beta_scale=args.beta_scale,
            highlight_pos_mult=args.highlight_pos_mult,
            content_loss=args.content_loss,
            smooth_gamma=args.smooth_gamma,
            use_delta=args.delta,
        )

        # Save final
        use_mcp = args.mcp or args.nr
        out = args.dataset.replace('.csv', '_mcp.pt') if use_mcp \
              else args.dataset.replace('.csv', '_mlp.pt')
        torch.save({'model_state': model.state_dict(), 'feat_cols': feats,
                    'mcp': use_mcp, 'envelope': args.envelope, 'mcp_k': MCP_K}, out)
        print(f"Saved final model: {out}", flush=True)

    finally:
        if tee:
            sys.stdout = tee._stdout
            tee.close()
