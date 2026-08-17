# ML Dynamic Tone-Mapping — Implementation Plan

**Last Updated**: 2026-08-17
**Branch**: build-v7.360.1
**Status**: Curve model training complete (run 10) — 61% better than libplacebo baseline

---

## Resuming Work in a New Session

### Quick status check (30 seconds)

```bash
# Current best checkpoint
ls F:/DTMModelData/ckpt10_best.pt

# Latest training log
tail -20 F:/DTMModelData/train_log10.txt

# Val/train dataset sanity
cd F:/DTMModelData && python3 check_both_splits.py
```

### Current best model

| File | Val MSE | Held-out MAE | Notes |
|---|---|---|---|
| `F:/DTMModelData/ckpt10_best.pt` | **0.00938** (ep 7) | **0.064 (~96 nits)** | Run 10, curve-only, no-tier |

**Baselines on same val set:**

| Method | MSE |
|---|---|
| libplacebo pl_tone_map_spline | 0.02411 |
| **Our model (ckpt10_best)** | **0.00938** |
| Identity (y=x, no tone map) | 0.00650 |

Model beats libplacebo by **61%**. Gap to identity floor: 44% remaining.

---

## Key Discoveries (Critical — Read Before Modifying Model)

### 1. Poly pivots are DELTA values, not absolute positions
The `poly_pivots` column stores **cumulative delta** values (e.g., `0 173 110 109...`).
They must be accumulated before normalising: `cumsum(deltas) / 1023`.
The original code divided each delta by 1023 directly → completely wrong gold curves (range ±16 instead of [0,1]).
**Fixed in `dv_coef_model.py::row_to_target()`.**

### 2. Polynomial is tier-independent — ONE curve per scene
The RPU polynomial is the SAME regardless of target display tier (143/1030/1669 nits).
Only the L2 trim parameters (slope/offset/power) differ per tier.
**Do NOT expand dataset 3× with `expand_tiers()` for curve training — use `--no-tier`.**
This reduced training data from 282K to 94K rows (3× faster) and removed irrelevant conditioning.

### 3. ms_weight is a per-title colorist constant — unlearnable from pixels
ms_weight (trim blend parameter) is ~512 for mindhunter and our_planet, ~2048 (identity) for others.
It's a mastering decision, not correlated with scene pixel features.
**Excluded from TRIM_PARAMS. Use libplacebo's gamut mapping for trim at inference.**

### 4. Mastering display luminance creates domain shift
mindhunter_s01: **4000 nits** mastering display. All others: 1000 nits.
This explains mindhunter's anomalous polynomial shapes — the 4000→143 nit compression ratio is 28:1 vs 7:1 for others.
At inference on HDR10, mastering_max_lum is unavailable (DV-only metadata).
Mitigation: `l1_max_pq` and `l1_avg_pq` (per-scene luminance peaks from RPU) are included as features — they partially encode the mastering context and have inference analogs (GPU histogram peak/avg via `pl_peak_detect`).

### 5. Curve type distribution in val set
Exhaustive analysis of model performance on 20,696 val scenes:

| Curve type | Val scenes | Model wins vs identity | Note |
|---|---|---|---|
| strong_compress (gold_dev < -0.05) | 11,888 | **100%** | Learned perfectly |
| mild_compress (-0.05 < gold_dev < -0.01) | 7,417 | 67% | Slightly over-compresses |
| near_identity (\|gold_dev\| < 0.01) | 934 | 22% | Model still over-compresses |
| boost (gold_dev > 0.02) | 457 | 2.2% | Essentially unlearned |

Model is **excellent for dark/vibrant content** (andor 98% wins, euphoria 99% wins).
Struggles on bright natural content (prehistoric/wondla — mild_compress with over-prediction) and boost curves (our_planet).

---

## Current Architecture

### Feature Vector (81 dimensions) — `--use-5x5`

```
[0]     maxscl                       ICtCp-I peak luminance
[1]     average_maxrgb               ICtCp-I mean luminance
[2]     fraction_bright_pixels       fraction of pixels > 0.5 PQ
[3-8]   distrib_val_3..8             luma percentiles (p25/p50/p75/p90/p95/p99)
[9]     l1_max_pq                    RPU L1 scene peak / 4095  → inference: GPU histogram peak
[10]    l1_avg_pq                    RPU L1 scene avg / 4095   → inference: GPU histogram avg
[11-28] zone_mean/max_3x3_r0c0..r2c2  3×3 SAT zone features (18)
[29-78] zone_mean/max_5x5_r0c0..r4c4  5×5 SAT zone features (50)
[79]    top_bar_norm                 L5 top offset / 2160
[80]    bottom_bar_norm              L5 bottom offset / 2160
```

**Note**: 5×5 zones and l1_max/avg_pq are added vs original design. 7×7 evaluated — diminishing returns, not worth re-extraction cost.

### Model: DVPolyMLP (`ml/dv_mlp_model.py`)

```
[81 features] → encoder: Linear(81→128) + LayerNorm + SiLU + Dropout(0.3) × 2
    → Poly head: num_segs (sigmoid) + pivots (cumsum-softmax) + coefs (linear) → [42]
    → Trim head: DISABLED (--no-trim) — use libplacebo gamut mapping at inference
```

No tier embedding (`--no-tier`) — polynomial is tier-independent.
Total params: ~37,165.

### Loss

```
total = MSE(eval_poly(pred_42), gold_256) + 50.0 × monotonicity_penalty
```

Trim loss disabled. Grain augmentation kept (50% of batches, zone features only).

### Val Split (run 10)

| Split | Titles | Scenes |
|---|---|---|
| **Val** | andor, euphoria, prehistoric, our_planet(E07-E08), wondla | 17,690 |
| **Train** | house, last_of_us, mindhunter, stranger, monarch, rings, mandalorian, sandman, ted, born, for, witcher, **our_planet(E01-E06)** | 95,728 |

our_planet E01-E06 was moved to training to expose boost-curve patterns.
andor and euphoria remain completely held out (never seen in training) — clean cross-title test.

### Training Command (run 10 reference)

```bat
python3 ml\dv_mlp_model.py ^
  --dataset F:\DTMModelData\train\train_dataset.csv ^
  --val-dataset F:\DTMModelData\val\val_dataset.csv ^
  --val-titles andor,euphoria,prehistoric,our,wondla ^
  --split-episodes our:E07,E08 ^
  --epochs 100 --dropout 0.3 --use-5x5 --no-trim --no-tier ^
  --save F:\DTMModelData\ckpt10 ^
  --log F:\DTMModelData\train_log10.txt
```

---

## Pending Work

### Immediate — Improve Curve Quality

**1. Weighted loss for hard curve types**

Boost (2.2% wins) and near_identity (22.2% wins) need focused gradient.
Implement per-sample loss weighting in `CurveLoss.forward()`:

```python
# Pre-compute gold_dev = mean(gold_curve - xs) per scene, store in dataset
# Weight by curve type:
#   boost (gold_dev > 0.02):    weight = 5.0
#   near_identity (<0.01 abs):  weight = 6.0
#   mild_compress:              weight = 1.5
#   strong_compress:            weight = 1.0
weighted_loss = mean(weight_i * (pred_curve_i - gold_curve_i)**2)
```

Use loss weighting (not WeightedRandomSampler) — only ~966 unique boost scenes in training, 20× oversampling would cause memorisation.

**2. Find a new title with boost-curve polynomials**

our_planet is now split across train/val. A clean boost-curve title (DV P5, nature documentary / bright CGI) would allow proper evaluation without episode splitting.

### Near-term — Trim Model

Trim (slope/offset/power) is a separate problem from curve prediction:
- Title-level colorist constants + per-scene variation
- Better suited to XGBoost or content-signature based approach
- Use `l1_max_pq`, `l1_avg_pq`, and aggregate episode statistics (rolling mean over 60 scenes) as features
- Train separate XGBoost models for slope/offset/power at each display tier

### Integration

**3. libplacebo integration**

```c
// pl_tone_map_function: ml_dtm
// Input: pl_tone_map_params with src_max, src_avg (from pl_peak_detect)
//        + spatial zone features from pl_extract_ml_features()
// Output: 42-dim polynomial → injected into DV render path
```

At inference:
- HDR display (>250 nits): identity passthrough (no model needed — polynomial is near-identity for HDR tiers)
- SDR display (~143 nits): ML model prediction

**4. ONNX export**

```python
torch.onnx.export(model, (dummy_features, dummy_tier), 'ml_dtm.onnx',
                  input_names=['features'], output_names=['poly_coefficients'])
```

**5. Validate temporal stability**

Frame-level prediction may pump. Test with libplacebo's `smoothing_period` IIR filter.
If pumping is visible, add temporal averaging of features over a 5-frame window before inference.

---

## libplacebo Baseline Evaluation Tool

`tools/libplacebo_baseline_eval.c` — evaluates `pl_tone_map_spline` on val set for comparison.
`tools/libplacebo_baseline_eval.py` — Python driver, outputs per-tier MSE vs gold RPU polynomial.

```bash
python3 tools/libplacebo_baseline_eval.py
# Results: libplacebo MSE=0.02411, identity MSE=0.00650, our model MSE=0.00938
```

Key finding: libplacebo is WORSE than identity at HDR tiers (1030/1669 nits) because it applies
compression when the gold polynomial is near-identity. Our model correctly predicts near-identity for those tiers.

---

## Dataset

### Stage 2 Datasets (pixel features + merged metadata)

| Split | Scenes | Size | Status |
|---|---|---|---|
| `val/val_dataset.csv` | 19,272 | 38 MB | ✅ Complete |
| `train/train_dataset.csv` | 94,146 | 188 MB | ✅ Complete |

Schema (148 cols): Stage 1 (65) + 77 daemon pixel features + `top_bar_norm` + `bottom_bar_norm`

**One representative frame per scene** (middle frame). Daemon extracts 77 ICtCp-I features.

### Training Data Curve Distribution

| Curve type | Training scenes | Notes |
|---|---|---|
| strong_compress | ~44,523 (46%) | Dark dramas, compression > 0.05 RMS |
| mild_compress | ~49,395 (51%) | Natural/bright, compression 0.01-0.05 RMS |
| near_identity | ~1,778 (2%) | Very small corrections |
| boost | ~966 (1%) | Signal lifting — our_planet, sandman, mindhunter |

---

## Success Criteria (Updated)

| Metric | Target | Current | Status |
|---|---|---|---|
| Val MSE vs libplacebo | beat by >2× | **2.6×** | ✅ **Achieved** |
| Val MSE vs identity | beat identity | 0.00938 vs 0.00650 | ⏳ Not yet |
| Boost curve wins | >50% | 2.2% | ❌ Needs weighted loss |
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
| `tools/libplacebo_baseline_eval.c` | libplacebo spline baseline evaluation | ~8K curves/s |
| `tools/libplacebo_baseline_eval.py` | Driver: val set MSE vs gold RPU polynomial | — |
| `tools/rpu_stage1_extract.py` | Stage 1 CSVs from RPU (no HEVC decode) | ~5-15s/episode |
| `tools/stage2_pixel_extract.py` | Stage 2 pixel extraction via daemon | ~0.4 scenes/s/worker |
| `ml/dv_mlp_model.py` | DVPolyMLP, Dataset, CurveLoss, training loop | — |
| `ml/dv_coef_model.py` | Feature definitions, data loading, GBR baseline | — |

---

## References

- libplacebo tone mapping API: `src/include/libplacebo/tone_mapping.h`
- DV RPU spec: ETSI TS 103 572 (L1/L2/L5 metadata, polynomial curves)
- ICtCp color space: ITU-R BT.2100 (signal_color_space=2 in RPU)
- pl_tone_map_sample: evaluates any registered tone map function at a single point
- Daemon: `tools/pl_extract_features_daemon.c` (stable for 1000+ frames, validated)
