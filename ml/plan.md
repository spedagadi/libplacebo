# ML Dynamic Tone-Mapping — Implementation Plan

**Last Updated**: 2026-08-17
**Branch**: build-v7.360.1
**Status**: Curve model + HDR10 inference viewer working — 61% better than libplacebo baseline

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

### Loss (current)

```
total = weighted_curve_MSE(eval_poly(pred_42), gold_256) + 50.0 × monotonicity_penalty
```

Weights: boost=5×, near_identity=6×, mild=1.5×, strong=1× (`--weighted-loss`).

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

### Immediate — Add New Training Titles

Two confirmed DV P5 nature documentaries downloaded to `G:\Dataset\`:
- `Our.Living.World.S01.2160p.NF.WEB-DL.DDP5.1.Atmos.DV.H.265-FLUX`
- `Our.Oceans.(2024).S01.(2160p.NF.WEB-DL.H265.DV.DDP.Atmos.5.1.English.-.HONE)`

Both confirmed P5 by daemon feature extraction (maxscl [0.35-0.69] range = ICtCp P5).
Already added to `tools/batch_extract.py` TITLES registry as "train".

**Expected value:** Fill the (bright, S-curve) training cell — boost-curve + midtone preservation for bright outdoor content. Currently only ~966 boost training scenes; these should add ~500-1000 more.

**Extraction steps:**
```bat
python3 tools\batch_extract.py --titles our_living_world_s01 --no-pixels
python3 tools\batch_extract.py --titles our_oceans_s01 --no-pixels
python3 tools\stage2_pixel_extract.py --split train --workers 2
```

After Stage 1, verify boost curves exist before committing to Stage 2 (full day):
```bash
python3 tools/check_black_bars.py  # check polynomial types for new titles
```

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

**2. Stratification improvement**
Current stratification is luminance-only. Issue: dark drama titles dominate the "mid-brightness" stratum with strong-compress polynomials, while nature docs with S-curves are absent. Fix:
- Add `gold_dev` (polynomial shape) as a stratification axis
- Ensure (bright, S-curve) and (dark, compress) are distinct strata
- Weighted loss (already implemented) compensates in the meantime

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
| Val MSE vs libplacebo | beat by >2× | **2.6×** | ✅ Achieved |
| Val MSE vs identity | beat identity | 0.00923 vs 0.00650 | ⏳ Not yet |
| Boost curve wins | >50% | 2.2% | ❌ Needs new training data |
| Near_identity wins | >60% | 22.2% | ⏳ Improving |
| Held-out MAE | < 0.05 PQ | **0.064** | Close |
| Inference time | < 0.5ms CPU | Not measured | ⏳ Pending |
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
| `tools/rpu_stage1_extract.py` | Stage 1 CSVs from RPU (no HEVC decode) | ~5-15s/episode |
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
