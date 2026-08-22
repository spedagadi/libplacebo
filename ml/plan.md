# ML Dynamic Tone-Mapping — Implementation Plan

**Last Updated**: 2026-08-18
**Branch**: build-v7.360.1
**Status**: 25-title master schema consolidated; curve-stratify master script ready

---

## Resuming Work in a New Session

### Quick status check (30 seconds)

```bash
# Current best checkpoint
ls F:/DTMModelData/ckpt11_best.pt

# Latest training log
tail -20 F:/DTMModelData/train_log11.txt

# Val/train dataset sanity
cd c:/Code/libplacebo && python3 tools/check_both_splits.py

# Master stratification manifest
ls F:/DTMModelData/balanced_manifest.csv

# Launch Streamlit inference viewer
streamlit run tools/ml_viewer.py
```

### Current best model

| File | Val MSE | Held-out MAE | Notes |
|---|---|---|---|
| `F:/DTMModelData/ckpt11_best.pt` | **0.00923** (ep 2) | **~96 nits** | Run 11, curve-only, no-tier, dropout=0.3, weighted-loss |

**Baselines on same val set (andor, euphoria, prehistoric, our E07-E08, wondla):**

| Method | MSE |
|---|---|
| libplacebo pl_tone_map_spline | 0.02411 |
| **Our model (ckpt11_best)** | **0.00923** |
| Identity (y=x, no tone map) | 0.00650 |

Model beats libplacebo by **61%**. Gap to identity floor: 42% remaining.

---

## Key Discoveries (Critical — Read Before Modifying Model)

### 1. Poly pivots are DELTA values, not absolute positions
`poly_pivots` stores cumulative deltas (e.g., `0 173 110 109...`).
Must accumulate first: `cumsum(deltas) / 1023`.
**Fixed in `dv_coef_model.py::row_to_target()`.**

### 2. Polynomial is tier-independent — ONE curve per scene
The RPU polynomial is the SAME for all 3 display tiers. Only L2 trims differ per tier.
**Use `--no-tier` for curve training** — removes 2/3 redundant data (282K→94K rows, 3× faster).
Tier embedding is irrelevant for curve prediction.

### 3. ms_weight is a per-title colorist constant — unlearnable from pixels
~512 for mindhunter/our_planet, 2048 (identity) for others. Not correlated with scene pixels.
**Excluded from TRIM_PARAMS.** Use libplacebo's gamut mapping for trim at inference.

### 4. Mastering display luminance creates domain shift
mindhunter_s01: **4000 nits** mastering display. All others: 1000 nits.
At inference on HDR10, mastering_max_lum is unavailable (DV-only metadata).
Mitigation: `l1_max_pq` and `l1_avg_pq` added as features (GPU histogram analogs at inference).

### 5. Curve type distribution — model wins/losses
Exhaustive analysis on 20,696 val scenes:

| Curve type | Val scenes | Model wins vs identity | Fix needed |
|---|---|---|---|
| strong_compress (dev < -0.05) | 11,888 | **100%** | ✅ None |
| mild_compress (-0.05 to -0.01) | 7,417 | 67% | More bright training data |
| near_identity (\|dev\| < 0.01) | 934 | 22% | Weighted loss |
| boost (dev > 0.02) | 457 | 2.2% | Bright nature doc training |

Dark content (andor 98%, euphoria 99% wins) is fully solved. Bright S-curve content is the remaining gap.

### 6. HDR10 inference LUT normalization — use [0, maxscl] domain
**Critical**: evaluate polynomial at `xs_abs = linspace(0, maxscl, 1024)` NOT `linspace(0, 1, 1024)`.
Using full [0,1] domain makes `ys[-1]` = polynomial at ICtCp=1.0 (10,000 nits!) causing massive overexposure.
Correct: `ys[-1]` = polynomial at maxscl → scale so `ys[-1] = target_pq/maxscl`.
**Fixed in `tools/ml_viewer.py::write_ml_lut()`.**

### 7. Bar detection — use ffmpeg pixel scan, not zone ratios or DAR metadata
L5 RPU bar offsets are hardcoded per-title in training (`tools/add_bar_features.py`).
At HDR10 inference: bars in 4K remuxes are real encoded pixels — ffprobe DAR=16:9 for all.
**Detection**: sample 5 frames, count consecutive near-black rows from top/bottom (threshold 0.03),
median across samples → `bar_norm = rows × 30 / 2160`.
Heat 2.39:1 detected as ≈0.128 ✓. Implemented in `tools/ml_viewer.py::detect_bar_norms()`.

---

## Current Architecture

### Feature Vector (81 dimensions) — `--use-5x5`

```
[0]     maxscl                       ICtCp-I peak luminance
[1]     average_maxrgb               ICtCp-I mean luminance
[2]     fraction_bright_pixels       fraction > 0.5 PQ
[3-8]   distrib_val_3..8             luma percentiles (p25-p99)
[9]     l1_max_pq                    RPU L1 scene peak / 4095  → inference: GPU histogram peak
[10]    l1_avg_pq                    RPU L1 scene avg / 4095   → inference: GPU histogram avg
[11-28] zone_mean/max_3x3_r0c0..r2c2  3×3 SAT zones (18)
[29-78] zone_mean/max_5x5_r0c0..r4c4  5×5 SAT zones (50)
[79]    top_bar_norm                 L5 top offset / 2160 → inference: ffmpeg pixel scan / 2160
[80]    bottom_bar_norm              L5 bottom offset / 2160
```

### Model: DVPolyMLP (`ml/dv_mlp_model.py`)

```
[81 features] → encoder: Linear(81→128) + LayerNorm + SiLU + Dropout(0.3) × 2
    → Poly head: num_segs + pivots (cumsum-softmax) + coefs → [42]
    → Trim head: DISABLED (--no-trim) — use libplacebo gamut mapping at inference
```

No tier embedding (`--no-tier`). Total params: ~37,165.

### Loss (current / legacy)

```
total = weighted_curve_MSE(eval_poly(pred_42), gold_256) + 50.0 × monotonicity_penalty
```

Weights: boost=5×, near_identity=6×, mild=1.5×, strong=1× (`--weighted-loss`).

### Naka-Rushton Model (Run 12+) — `--nr`

**Replace 42 unconstrained coefficients with 2 physically-bounded NR parameters.**

```
[27+ features] → encoder: Linear(→128) + LayerNorm + SiLU + Dropout(0.3) × 2
    → NR head: Linear(128→2) → [σ, n] via [softplus, sigmoid×3+1]
    → DifferentiableNakaRushton → [256-pt curve]
```

**Naka-Rushton equation:** `Y = Xⁿ / (Xⁿ + σⁿ)`

| Parameter | Range | Activation | Physical meaning |
|---|---|---|---|
| σ (sigma) | (1e-5, ∞) | softplus(raw) + 1e-5 | Semi-saturation / curve bend point |
| n (exponent) | (1, 4) | sigmoid(raw) × 3 + 1 | Contrast steepness |

**Guarantees (why this beats polynomials):**
- **Monotonicity:** f'(x) > 0 for all x > 0 — mathematically impossible to invert
- **Zero anchor:** f(0) = 0ⁿ / (0ⁿ + σⁿ) = 0 — black pinned to black, no letterbox lifting
- **C∞ continuity:** infinitely differentiable — no blocky posterization or step-jumps
- **No solarization:** no negative coefficients to flip the curve — physically valid at every step

**Loss: BoundedDTMLoss (`--envelope`)**

```
envelope_lower = min(gold_curve, spline_baseline)  per evaluation point
envelope_upper = max(gold_curve, spline_baseline)

base_loss = MSE(gold, predicted)                    inside envelope (λ=1.0)
hinge_loss = clamp(lower - predicted, min=0)        below envelope
         + clamp(predicted - upper, min=0)          above envelope (λ=10.0)

total = (base_loss × cell_weight) + (hinge_loss × 10.0)
```

Cell weights force exploration of rare cells:
| Cell | Weight | Rationale |
|---|---|---|
| boost-boost-boost | 6× | Global expansion — model has almost zero examples |
| boost-boost-neutral | 6× | Extreme expansion — critical for bright HDR content |
| neutral-crush-crush | 4× | Standard DV — anchor identity behavior |
| neutral-neutral-boost | 4× | Pure expansion — prevents over-compression |

**Two-Phase Training (`--phase 1` / `--phase 2`)**

```
Phase 1 (tone mapping):  Train backbone + NR head  (freeze color trim)
Phase 2 (color):         Freeze backbone + NR head → Train color trim head only
```

Phase 2 uses `--resume-from ckpt_nr_run12_best.pt` to load Phase 1 weights.

**Enabling NR mode:**
```bat
python3 ml\dv_mlp_model.py ^
  --dataset F:\DTMModelData\train\train_dataset.csv ^
  --val-dataset F:\DTMModelData\val\val_dataset.csv ^
  --val-titles andor,euphoria,prehistoric,our,wondla ^
  --split-episodes our:E07,E08 ^
  --epochs 100 --dropout 0.3 --use-5x5 --no-trim --no-tier --nr --envelope ^
  --save F:\DTMModelData\ckpt_nr_run12 ^
  --log F:\DTMModelData\train_log_nr_run12.txt
```

### Val Split (run 11)

| Split | Titles | Scenes |
|---|---|---|
| **Val** | andor, euphoria, prehistoric, our_planet(E07-E08), wondla | 17,690 |
| **Train** | house, last_of_us, mindhunter, stranger, monarch, rings, mandalorian, sandman, ted, born, for, witcher, **our_planet(E01-E06)** | 95,728 |

### Training Command (run 11 reference)

```bat
python3 ml\dv_mlp_model.py ^
  --dataset F:\DTMModelData\train\train_dataset.csv ^
  --val-dataset F:\DTMModelData\val\val_dataset.csv ^
  --val-titles andor,euphoria,prehistoric,our,wondla ^
  --split-episodes our:E07,E08 ^
  --epochs 100 --dropout 0.3 --use-5x5 --no-trim --no-tier --weighted-loss ^
  --save F:\DTMModelData\ckpt11 ^
  --log F:\DTMModelData\train_log11.txt
```

---

## HDR10 Inference Test Frames

These are the reference test frames used to validate the HDR10 inference path:

| Title | File | Frame | PTS | nits | Scene type | Key findings |
|---|---|---|---|---|---|---|
| **Heat (1995)** | `G:\Heat.1995.2160p.Remux.HDR10plus.HEVC.DTS-HD.MA.5.1-SYS.mkv` | — | **901.0s (15:01)** | 143 | Dark/dramatic night scene | ML shadow lift: much better than libplacebo; expansion case (target_yn=1.22) |
| **Exodus (2023)** | `D:\Jdownloader\Exodus.mp4` | 800 | 33.3s | 143 | Bright outdoor (Nile blood plague) | ML≈libplacebo for bright content; near-linear polynomial |
| **Supergirl (2026)** | `D:\Jdownloader\Supergirl_2026.mkv` | 40500 | 1689.2s | 50 | Bright CGI explosion | Reference frame for LUT normalization testing; 2.39:1 letterbox confirmed |
| Supergirl (dark) | same | 16639 | 694.7s | 143 | Dark alien creature / specular | Used to test banding fix (PCHIP → UnivariateSpline) |
| Supergirl (dark) | same | 16656 | 695.4s | 143 | Same scene | Used for bar detection validation |

**Key Supergirl test outcomes:**
- Frame 40500 at 50 nits: correct reference render is `F:\DTMModelData\compare\sg_f40500_render.png`
- Banding fixed by: UnivariateSpline on [0, maxscl] domain with correct `target_yn` normalization
- Bar detection: top=0.125, bot=0.139 (detected letterboxed 2.39:1 content)
- Bar features shift curve by ~0.021 more compression (cinematic content learned behaviour)

**Quick test command for any HDR10 frame:**

```bash
python3 tools/ml_compare.py \
  --video "G:\Heat.1995.2160p.Remux.HDR10plus.HEVC.DTS-HD.MA.5.1-SYS.mkv" \
  --pts 901.0 --nits 143
# Output saved to F:\DTMModelData\compare\
```

---

## Pending Work

### Immediate — Run 12: Naka-Rushton + Bounded Loss

**Goal:** Beat identity floor (0.00650) by replacing unconstrained polynomial output with
physically-bounded Naka-Rushton parameters and envelope-constrained loss.

```bat
REM Phase 1: Train backbone + NR head
python3 ml\dv_mlp_model.py ^
  --dataset F:\DTMModelData\train\train_dataset.csv ^
  --val-dataset F:\DTMModelData\val\val_dataset.csv ^
  --val-titles andor,euphoria,prehistoric,our,wondla ^
  --split-episodes our:E07,E08 ^
  --epochs 100 --dropout 0.3 --use-5x5 --no-trim --no-tier --nr --envelope ^
  --save F:\DTMModelData\ckpt_nr_run12 ^
  --log F:\DTMModelData\train_log_nr_run12.txt

REM Phase 2 (optional): Freeze backbone+NR, train color trim
REM python3 ml\dv_mlp_model.py ^
REM   --dataset ... --val-dataset ... --val-titles ... ^
REM   --phase 2 --resume-from F:\DTMModelData\ckpt_nr_run12_best.pt ^
REM   --epochs 5 --lr 1e-4 --nr --envelope ^
REM   --save F:\DTMModelData\ckpt_nr_run12 ^
REM   --log F:\DTMModelData\train_log_nr_run12_phase2.txt
```

**Expected improvements over Run 11:**
| Issue | Run 10/11 (poly) | Run 12 (NR) |
|---|---|---|
| Val MSE | 0.00923 | Target < 0.00650 |
| Monotonicity | 50× penalty (imperfect) | Guaranteed by NR equation |
| Letterbox lifting | Shadow artifacts | f(0)=0 pins black |
| Boost curves | 2.2% wins | Better — envelope guides expansion |
| Over-compression | Identity scenes damaged | Envelope anchors to identity |
| Gradient competition | Trim head steals gradients | Two-phase training isolates |

### Immediate — 25-Title Master Extraction (2026-08-18)

**Status:** All 25 titles registered in pipeline scripts. Ready for full re-extraction with curve-shape stratification.

The old 19-title dataset (94K scenes, L1 luminance-based stratification) is being replaced by:
- 25 titles (add: Shōgun, Silo, The Penguin, Bad.Batch, Tales.of.the.Empire, Ahsoka)
- Curve-shape stratification (shadow_dev / midtone_dev / highlight_dev bands)
- Algorithmic balancing (15K/cell cap on dominant cells)
- Inter-episode 80/20 split with zero frame-level leakage

**Pipeline:** `rpu_stage1_extract.py` → spot-check (`check_both_splits.py` + `curve_stratify_master.py --max-cell 0`) → balance → `stage2_pixel_extract.py`

Old titles are already on disk at `G:\Dataset\`. New titles must be placed there before extraction.

See `ml/plan.md` "Master 25-Title Pipeline" section and `tools/curve_stratify_master.py` for the full schema.

### ⚠️ CRITICAL: Correct Extraction Pipeline

**DO NOT use `batch_extract.py` for Stage 1** — it calls `dv_metadata_extract.py` which uses
ffprobe `-show_frames` to decode the full video stream. This takes **~30 min/episode**.

**Master 25-Title Pipeline (use for all future extraction):**

```bat
REM Step 1: Stage 1 RPU → CSV for all 25 titles
python3 tools\rpu_stage1_extract.py --workers 8

REM ★ SPOT-CHECK (MANDATORY): Aggregate train/val before proceeding ★
python3 tools\check_both_splits.py
REM Also run: python3 tools\curve_stratify_master.py --stage1 F:\DTMModelData\stage1_output.csv --output F:\DTMModelData\master_manifest.csv --max-cell 0
REM Verify: val has at least 20% of scenes per title; train has no val episode leakage
REM If spot-check fails, fix TITLES dict or VAL_EPISODE_THRESHOLDS before Step 3

REM Step 2: Balance + stratify (curves, not L1 luminance)
python3 tools\curve_stratify_master.py --stage1 F:\DTMModelData\stage1_output.csv --output F:\DTMModelData\balanced_manifest.csv

REM Step 3: Pixel extraction (daemon, ~0.4 scenes/s)
python3 tools\stage2_pixel_extract.py --split both --workers 1
```

**25-Title Inter-Episode Split Schema (80/20, zero leakage):**

| Title | Train episodes | Val episodes | Threshold |
|---|---|---|---|
| Andor S02 | 01–09 | 10–12 | ≥10 → val |
| Born.to.Be.Wild S01 | 01–05 | 06 | ≥6 → val |
| Euphoria S03 | 01–06 | 07–08 | ≥7 → val |
| For.All.Mankind S05 | 01–08 | 09–10 | ≥9 → val |
| House.of.the.Dragon S03 | 01–06 | 07–08 | ≥7 → val |
| Mindhunter S01 | 01–08 | 09–10 | ≥9 → val |
| Monarch S02 | 01–08 | 09–10 | ≥9 → val |
| Our.Living.World S01 | 01–03 | 04 | ≥4 → val |
| Our.Oceans S01 | 01–04 | 05 | ≥5 → val |
| Our.Planet S01 | 01–06 | 07–08 | ≥7 → val |
| Prehistoric.Planet S03 | 01–04 | 05 | ≥5 → val |
| Stranger.Things S05 | 01–06 | 07–08 | ≥7 → val |
| Ted.Lasso S03 | 01–09 | 10–12 | ≥10 → val |
| Last.of.Us S02 | 01–07 | 08–09 | ≥8 → val |
| Rings.of.Power S02 | 01–06 | 07–08 | ≥7 → val |
| Mandalorian S01 | 01–06 | 07–08 | ≥7 → val |
| Sandman S01 | 01–08 | 09–11 | ≥9 → val |
| Witcher S04 | 01–06 | 07–08 | ≥7 → val |
| WondLa S03 | 01–05 | 06 | ≥6 → val |
| Shōgun S01 | 01–08 | 09–10 | ≥9 → val |
| Silo S03 | 01–07 | — | all train (S03 only, 7 eps) |
| The.Penguin S01 | 01–06 | 07–08 | ≥7 → val |
| Bad.Batch S03 | 01–13 | 14–16 | ≥14 → val |
| Tales.of.the.Empire S01 | 01–05 | 06 | ≥6 → val |
| Ahsoka S01 | 01–06 | 07–08 | ≥7 → val |

**Key invariant:** episode boundaries are the isolation boundary — NO frame-level mixing
between train and val. This prevents temporal leakage where adjacent scenes in the same
episode share lighting/color grading that the model could memorize rather than generalise.

**Old fast path (legacy — only for single-title incremental runs):**

```bat
REM Step 1: Extract RPU binary files via dovi_tool (~90s/episode, parallel)
python3 tools\rpu_extract_batch.py -o F:\DTMModelData\rpu --title our_living_world_s01
python3 tools\rpu_extract_batch.py -o F:\DTMModelData\rpu --title our_oceans_s01

REM Step 2: Stage 1 CSVs from RPU — pure dovi_tool JSON parse, no HEVC decode (~5-15s/episode)
python3 tools\rpu_stage1_extract.py --rpu-dir F:\DTMModelData\rpu --title our_living_world_s01
python3 tools\rpu_stage1_extract.py --rpu-dir F:\DTMModelData\rpu --title our_oceans_s01

REM Step 3: Verify boost curves exist before committing to Stage 2
python3 tools\check_black_bars.py

REM Step 4: Stage 2 pixel features (daemon, ~3-4h per title)
python3 tools\stage2_pixel_extract.py --split train --workers 2
```

`batch_extract.py` is for Stage 2 only (full pixels via `--full-pixels` flag).
Stage 1 always uses the two-step rpu_extract_batch → rpu_stage1_extract pipeline.

### Spot-Check After Stage 1 (MANDATORY)

After running `rpu_stage1_extract.py`, always verify the data quality before proceeding:

```bat
REM Check both splits have reasonable coverage
python3 tools\check_both_splits.py

REM Quick stratification overview WITHOUT balancing (max-cell=0 = no cap)
python3 tools\curve_stratify_master.py --stage1 F:\DTMModelData\stage1_output.csv --output F:\DTMModelData\quick_check.csv --max-cell 0

REM Check:
REM   - Each title has scenes in BOTH train and val
REM   - Val has at least 20% of title's scenes
REM   - No title is 100% in one split (episode threshold is working)
REM   - Major cells (neutral-crush-crush, neutral-neutral-boost) are well-populated
REM   - Rare cells (boost-boost-boost, boost-boost-crush) have at least a few scenes
```

If spot-check reveals missing titles or split issues, fix `TITLES` dict in
`rpu_stage1_extract.py` or `VAL_EPISODE_THRESHOLDS` in `curve_stratify_master.py`.

### Improve Curve Quality

**1. Retrain with new titles + weighted loss**
After new data extraction:
```bat
python3 ml\dv_mlp_model.py ^
  --dataset F:\DTMModelData\train\train_dataset.csv ^
  --val-dataset F:\DTMModelData\val\val_dataset.csv ^
  --val-titles andor,euphoria,prehistoric,our,wondla ^
  --split-episodes our:E07,E08 ^
  --epochs 100 --dropout 0.3 --use-5x5 --no-trim --no-tier --weighted-loss ^
  --save F:\DTMModelData\ckpt12 ^
  --log F:\DTMModelData\train_log12.txt
```

**2. Curve-shape stratification (replaces luminance stratification)**

Current stratification selects scenes by L1 luminance stats only. Critical flaw: two scenes
with identical L1 stats but different polynomial shapes (dark drama → compress, nature doc →
lift) land in the same stratum → contradictory gradients → model learns the average (near-linear).

**Use `tools/curve_stratify_master.py`** — the definitive master stratification for all 25 titles.

For each scene, compute three regional polynomial deviations:
```python
shadow_dev    = mean(gold(x) - x)  for x ∈ [0.0, 0.2]   # + = lift, - = crush
midtone_dev   = mean(gold(x) - x)  for x ∈ [0.2, 0.5]   # + = lift, - = compress
highlight_dev = mean(gold(x) - x)  for x ∈ [0.5, maxscl] # + = boost, - = rolloff
```

Stratification cells: shadow × midtone × highlight bands (3×3×3 = 27 observed, up to 36 possible).

**Balancing strategy:** cap dominant cells (`neutral-crush-crush`, `neutral-neutral-boost`) at
`MAX_FRAMES_PER_CELL = 15000`. This forces the model to see rare cells (boost-boost-boost,
boost-boost-crush) that are currently <0.5% of training data.

**Band thresholds:** shadow (< -0.06: crush, > 0.02: boost), midtone (< -0.04: compress, > 0.02: boost),
highlight (< -0.03: rolloff, > 0.02: boost).

Observed from 120K+ scenes: 9 cells cover >99.9% of content. The 3 critical missing cells for
model training: `boost-boost-boost` (global expansion), `boost-boost-crush` (classic S-curve),
`boost-neutral-boost` (dual expansion).

**3. Zone-masked texture features (Phase 1 — no re-extraction)**

Colorists use a "qualifier" to grade within specific luminance bands. The model currently
has no concept of texture within a brightness region. Add derived features:

```python
# For each of 9 (3×3) zones, compute local contrast proxy:
local_contrast = zone_max_3x3 - zone_mean_3x3

# Classify each zone by its mean luminance, then aggregate:
shadow_texture    = mean(local_contrast for zones where zone_mean < 0.25)
midtone_texture   = mean(local_contrast for zones where 0.25 <= zone_mean <= 0.65)
highlight_texture = mean(local_contrast for zones where zone_mean > 0.65)
```

Why it helps: high `highlight_texture` (explosion debris, specular surfaces) tells the model
it cannot clip those highlights without destroying texture. Low `highlight_texture` (smooth sky,
blank wall) signals safe aggressive compression.

Implementable in `load_data()` in `dv_coef_model.py` from existing columns — zero new extraction.
Adds 3 features (84 total). Implement in `compute_zone_texture(df)` function.

---

### ⚠️ Model Behaviour Warning: Identity Case / Over-lifting

**Observed on Heat 1995 t=15:01 (dark scene, maxscl=0.445, target=143 nits):**
- Scene peaks at ~65 nits, display target is 143 nits → expansion case (target_yn=1.22)
- ML model applies +0.086 PQ shadow lift at midtones — 10× more than libplacebo's +0.009
- Visual result: "washed out" lifted scene that destroys Michael Mann's intentional dark grading
- libplacebo correctly outputs near-identity (content is already close to display range)

**Root cause:** Model learned from DV dark drama training that "dark + low avg = apply shadow lift."
This was correct for DV because the colorist *chose* to apply shadow lift. For HDR10, no such
creative decision was made — the correct answer is near-identity linear expansion.

**Two concrete fixes needed before production:**

**Fix A — Near-identity penalty in training loss:**
```python
# When expansion case (content darker than display), penalise deviation from identity
if target_yn >= 1.0:
    expansion_penalty = MSE(pred_curve, identity_curve) * expansion_weight
    total_loss += expansion_penalty
```
This teaches the model: "when display is brighter than content, don't impose DV shadow lift."

**Fix B — Strict positive derivative in LUT (no plateau/clipping):**
```python
# After UnivariateSpline fitting, enforce minimum slope to prevent plateaus
min_slope = target_yn / n_pts * 0.1  # 10% of average slope as floor
dy = np.maximum(np.diff(ys), min_slope)
ys = np.concatenate([[ys[0]], ys[0] + np.cumsum(dy)])
```
Eliminates the hard clip that destroys highlight texture in bright-specular regions.

**Context gate principle:**
The loss function should be asymmetric:
- Expansion (`target_yn > 1`): penalise any deviation from linear expansion more heavily
- Compression (`target_yn < 1`): allow the learned DV-style shaping

Currently the model applies compression/lift patterns regardless of whether expansion or
compression is needed — it has no context gate between these two regimes.

---

### Near-term — Trim Model

Separate XGBoost model for slope/offset/power using content-level aggregate features
(rolling mean over 60 scenes) rather than per-scene pixels. Title-level colorist constants
are unlearnable from single-frame pixels.

### Integration

**3. libplacebo integration**
- Map `pl_extract_ml_features()` output (78 features) to model's 81-feature input
  - Add l1_max_pq (GPU histogram peak) and l1_avg_pq (GPU histogram avg) to C API
  - Bar features: compute from `top_rows × 30 / 2160` via pixel scan at init
- Register as `pl_tone_map_function` for HDR10 fallback path
- For SDR display (<250 nits): use ML polynomial → then libplacebo spline for residual
- For HDR display (>250 nits): identity passthrough (polynomial is near-identity)

**4. ONNX export**

```python
torch.onnx.export(model, dummy_features, 'ml_dtm.onnx',
                  input_names=['features'], output_names=['poly_42'])
```

**5. Validate temporal stability**
Frame-level prediction may pump. Test with libplacebo's `smoothing_period` IIR filter.

---

## Baseline Evaluation

**Tool:** `tools/libplacebo_baseline_eval.c` + `tools/libplacebo_baseline_eval.py`

```bash
python3 tools/libplacebo_baseline_eval.py
# libplacebo MSE=0.02411 (vs gold RPU polynomial on val set)
```

Key finding: libplacebo is WORSE than identity (0.00650) at HDR tiers (1030/1669 nits)
because it compresses when the gold polynomial is near-identity. Our model beats libplacebo
by **61%** across all tiers.

---

## Dataset

### Stage 2 Datasets (pixel features + merged metadata)

| Split | Scenes | Size | Status |
|---|---|---|---|
| `val/val_dataset.csv` | 19,272 | 38 MB | ✅ Complete |
| `train/train_dataset.csv` | 94,146 | 188 MB | ✅ Complete |

Schema (148 cols): Stage 1 (65) + 77 daemon pixel features + `top_bar_norm` + `bottom_bar_norm`

### Training Data Curve Distribution (current 94K scenes)

| Curve type | Training scenes | Notes |
|---|---|---|
| strong_compress | ~44,523 (46%) | Dark dramas — model excels here |
| mild_compress | ~49,395 (51%) | Natural/bright — model 67% wins |
| near_identity | ~1,778 (2%) | Rare — model 22% wins |
| boost | ~966 (1%) | Signal lift — model 2% wins → needs Our.Living.World + Our.Oceans |

---

## Success Criteria (Updated)

| Metric | Target | Current | Status |
|---|---|---|---|
| Val MSE vs libplacebo | beat by >2× | **2.6×** (Run 11) | ✅ Achieved |
| Val MSE vs identity | beat identity | 0.00923 vs 0.00650 (Run 11) | ⏳ Run 12 NR target < 0.00650 |
| Boost curve wins | >50% | 2.2% (Run 11) | ❌ Run 12 + new data |
| Near_identity wins | >60% | 22.2% (Run 11) | ⏳ NR envelope helps |
| Monotonicity violations | 0% | Non-zero (poly kinks) | ✅ NR guarantees 0 |
| Letterbox artifacts | None | Present (shadow lift) | ✅ NR f(0)=0 |
| Held-out MAE | < 0.05 PQ | **0.064** (Run 11) | Close |
| Inference time | < 0.5ms CPU | Not measured | ⏳ NR simpler = faster |
| No temporal pumping | — | Not tested | ⏳ Pending |

---

## Pipeline Tools

| Tool | Purpose | Speed |
|---|---|---|
| `tools/pl_extract_features_daemon.c` | Persistent GPU context, 77 features/frame | 0.143s/frame |
| `tools/libplacebo_daemon_client.py` | Python IPC client for daemon | — |
| `tools/libplacebo_baseline_eval.c/py` | libplacebo spline baseline evaluation | ~8K curves/s |
| `tools/ml_viewer.py` | Streamlit interactive viewer (DV + HDR10) | — |
| `tools/ml_compare.py` | Command-line frame comparison | — |
| `tools/dv_render.c` | Headless DV frame renderer (all modes incl. ml-lut) | ~3-5s/frame |
| `tools/rpu_extract_batch.py` | Extract RPU binaries from MKV | ~90s/episode |
| `tools/rpu_stage1_extract.py` | Stage 1 CSVs from RPU (no HEVC decode) | ~5-15s/episode |
| `tools/curve_stratify_master.py` | Master: curve-stratify + 25-title balance + 80/20 split | ~30s/120K scenes |
| `tools/curve_stratify.py` | Legacy curve stratification (19 titles) | — |
| `tools/stage2_pixel_extract.py` | Stage 2 pixel extraction via daemon | ~0.4 scenes/s |
| `tools/batch_extract.py` | Full pipeline: RPU + Stage1 + Stage2 per title | — |
| `tools/check_stage2_progress.py` | Dataset progress check | instant |
| `tools/check_both_splits.py` | Schema + quality validation | ~10s |
| `tools/add_bar_features.py` | Post-process: add L5 bar normalisation features | ~5s |
| `tools/check_black_bars.py` | Letterbox + zone contamination analysis | ~30s |
| `ml/dv_mlp_model.py` | DVPolyMLP, Dataset, CurveLoss, training loop | — |
| `ml/dv_coef_model.py` | Feature definitions, data loading, GBR baseline | — |

---

## References

- libplacebo tone mapping API: `src/include/libplacebo/tone_mapping.h`
- DV RPU spec: ETSI TS 103 572 (L1/L2/L5 metadata, polynomial curves)
- ICtCp color space: ITU-R BT.2100 (signal_color_space=2 in RPU)
- pl_tone_map_sample: evaluates any registered tone map function at a single point
- Daemon: `tools/pl_extract_features_daemon.c` (stable for 1000+ frames, validated)
- Bar offsets: `tools/add_bar_features.py` (hardcoded per-title L5 values from RPU)
