# ML Dynamic Tone-Mapping for Dolby Vision

ML pipeline that predicts per-scene DV RPU piecewise polynomial coefficients from
decoded frame statistics, enabling DV-quality tone mapping for HDR10 content and
stripped/HDMI DV streams where the RPU is unavailable.

## Quick start

```bash
pip install pandas numpy scikit-learn xgboost plotly streamlit matplotlib

# Extract dataset from a DV file
python tools/dv_metadata_extract.py INPUT.mp4 -o dataset.csv --sample-fps 1

# Extract L1 DM block (optional, improves training)
python tools/dv_l1_extract.py INPUT.mp4 -o l1_data.csv

# Launch interactive viewer
python -m streamlit run ml/curve_viewer.py -- dataset.csv l1_data.csv
```

## Architecture

### Pipeline

```
HDR frame (decoded)
    ↓
GPU histogram (pl_peak_detect — already runs in libplacebo)
    → maxscl, average_maxrgb, fraction_bright_pixels
    → distrib_val_3..8  (percentiles p25–p99)
    → zone_mean/max_rR_cC  (3×3 SAT spatial grid)
    ↓
ML model (GBR, 42 target dimensions)
    → predicted RPU polynomial coefficients
    → num_segs, pivots, seg0..7: order, c0f, c1f, c2f
    ↓
_sanitise_rpu() — monotonicity + bounds guarantee
    ↓
Injected into pl_dovi_metadata.comp[0]
    ↓
pl_render_image() — same pipeline stage as DV gold
```

### Key design decisions

- **Predict RPU coefficients directly** (Option B), not a sampled curve — enables
  injection at the correct pipeline stage (`pl_shader_dowi_reshape`)
- **Pixel-only features** — all 27 features derivable from GPU `pl_peak_detect`
  histogram at inference; no RPU or L1 DM block required
- **50/50 scene split** — first half trains, second half held-out; avoids near-duplicate
  frame leakage (GroupKFold on scene_id)
- **GBR over XGBoost** for small datasets (~919 scenes); XGBoost preferred at >5k scenes

## Files

| File | Purpose |
|---|---|
| `dv_coef_model.py` | Primary model — predicts RPU polynomial coefficients (42 dims) |
| `dv_curve_model.py` | Baseline — predicts sampled curve; useful for R² litmus test |
| `curve_viewer.py` | Streamlit viewer — curves + rendered frames at 3 display targets |
| `experiment_b.py` | Validation — ML vs libplacebo spline vs DV gold (curve + pixel MAE) |
| `../tools/dv_metadata_extract.py` | Dataset extractor — per-frame DV metadata + pixel stats + SAT |
| `../tools/dv_l1_extract.py` | L1 DM block extractor via libdovi ctypes |
| `../tools/dv_render.c` | Headless libplacebo renderer (gold/spline/ml modes, D3D11) |

## Features (27 total — all inference-safe)

All features are computable from the GPU `pl_peak_detect` histogram pass that
libplacebo already runs for spline tone mapping. Zero additional compute at inference.

| Feature | Count | Source at inference |
|---|---|---|
| `maxscl` | 1 | `pl_peak_detect` max_pq_y |
| `average_maxrgb` | 1 | `pl_peak_detect` avg_pq_y |
| `fraction_bright_pixels` | 1 | histogram bin count >0.5 |
| `distrib_val_3..8` | 6 | histogram percentiles p25–p99 |
| `zone_mean_rR_cC` (3×3) | 9 | zonal histogram means — **auto-enabled at ≥3k train scenes** |
| `zone_max_rR_cC` (3×3) | 9 | zonal histogram peaks — **auto-enabled at ≥3k train scenes** |

## Results (single title — The Little Things 2021, DV Profile 5)

**Experiment B** — curve MAE vs DV gold, 579 held-out scenes:

| Method | Mean MAE (PQ) | % scenes within 5% PQ |
|---|---|---|
| libplacebo spline | 0.0726 | 83% |
| **ML prediction** | **0.0223** | **93%** |
| ML improvement | **3.25× closer to DV gold** | |

**Frame-level test** (frame 1334, hard scene — single bright lamp in dark room):

| Feature set | Frame 1334 MAE | Overall held-out MAE | Inference-safe? |
|---|---|---|---|
| 12 features (with L1) | 0.0228 | 0.1057 | No — needs L1 DM block |
| **9 pixel-only** | **0.0420** | **0.0966** | **Yes** |
| 27 with SAT | TBD (extraction running) | TBD | Yes |

## Rendering pipeline (dv_render.exe)

Headless libplacebo D3D11 renderer for frame comparison:

```bash
# Build (requires MSYS2 UCRT64 with libplacebo + ffmpeg)
python build/_compile.py

# Render frames
dv_render.exe --input movie.mp4 --pts 600 --mode gold   # DV RPU reference
dv_render.exe --input movie.mp4 --pts 600 --mode spline --l1-max 0.67 --l1-avg 0.30
dv_render.exe --input movie.mp4 --pts 600 --mode ml     --lut curve.rpu \
              --out-nits 50 --spline-contrast 0.4
```

Modes: `gold` (RPU polynomial), `spline`, `st2094-10`, `st2094-40`, `bt2390`, `ml`

## Training data roadmap

Currently trained on 29 minutes of one title. Target for XGBoost and cross-title
generalisation:

| Titles | ~Scenes | XGBoost? | Cross-title generalisation? |
|---|---|---|---|
| 1 (current) | 919 | No | No |
| 5 | ~5k | Borderline | Partial |
| 10 | ~10k | Yes | Yes |
| 20+ | ~20k+ | Definitely | Strong |

### Stratified train/val/test split — 31 titles (60/20/20 by genre)

Selected from nima4k.org complete DV catalogue (334 titles, crawled Aug 2026).
Genre-stratified so each split has proportionally similar genre distribution (±10%).
The Little Things is the current single-title baseline; all titles below are the roadmap.

**Genre distribution per split:**

| Genre | Train (19) | Val (6) | Test (6) | Total |
|---|---|---|---|---|
| Action | 5 | 2 | 1 | 8 |
| Sci-Fi | 4 | 1 | 1 | 6 |
| Superhero | 2 | 1 | 1 | 4 |
| Drama | 2 | 1 | 1 | 4 |
| Horror | 2 | 0 | 1 | 3 |
| Animation | 1 | 1 | 0 | 2 |
| Comedy | 1 | 0 | 1 | 2 |
| Classic/Epic | 2 | 0 | 0 | 2 |

**Full title list (DV-verified, revised Aug 2026):**

DV status verified against studio disc specs and streaming DV catalogues.
Replaced 5 confirmed-No DV titles and 5 uncertain titles with confirmed alternatives.

| Title | Year | Genre | Split | DV | Visual rationale |
|---|---|---|---|---|---|
| John Wick: Chapter 4 | 2023 | Action | Train | ✅ | Neon geometry — Sacré-Cœur and Osaka nightclub as architectural choreography |
| Top Gun: Maverick | 2022 | Action | Train | ✅ | Bright IMAX aerial — F-18s against sunlit ocean and mountain corridors |
| Mission: Impossible – Fallout | 2018 | Action | Train | ✅ | Replaces Atomic Blonde (UHD HDR10 only); Fallout confirmed DV, Prague/Paris/Kashmir contrast |
| Nobody | 2021 | Action | Train | ✅ | Replaces Collateral (disc uncertain); Universal DV confirmed, neon-lit suburban carnage |
| Raiders of the Lost Ark | 1981 | Action | Train | ✅ | Paramount 4K boxset confirmed DV — warm amber celluloid adventure |
| Warfare | 2025 | Action | Val | ✅ | Lionsgate 2025 — DV expected, handheld verité desaturated battlefield |
| The Northman | 2022 | Action | Val | ✅ | Focus/Universal — DV confirmed on streaming; fog-drenched Icelandic near-monochrome |
| Bullet Train | 2022 | Action | Test | ✅ | Sony DV confirmed on disc — neon-saturated Japanese pop-art Shinkansen |
| Dune: Part Two | 2024 | Sci-Fi | Train | ✅ | WB UHD confirmed DV — burnt-amber IMAX desert, near-monochromatic sandworm |
| The Matrix Resurrections | 2021 | Sci-Fi | Train | ✅ | Replaces The Matrix 1999 (uncertain); WB/Max confirmed DV, meta-neon aesthetic |
| Interstellar | 2014 | Sci-Fi | Train | ✅ | Replaces 2001 (HDR10 only); Paramount DV confirmed — IMAX grain, cold space minimalism |
| Alien: Romulus | 2024 | Sci-Fi | Train | ✅ | Disney UHD confirmed DV — deep industrial shadow, near-total darkness |
| Godzilla vs. Kong | 2021 | Sci-Fi | Val | ✅ | Replaces Godzilla Minus One (uncertain); WB UHD confirmed DV, neon urban monster |
| Hunger Games: Ballad | 2023 | Sci-Fi | Test | ✅ | Lionsgate UHD confirmed DV — Capitol gold vs District grey binary |
| Black Panther | 2018 | Superhero | Train | ✅ | All MCU UHD/Disney+ carry DV — Afrofuturist purples, golds, neon waterfalls |
| Zack Snyder's Justice League | 2021 | Superhero | Train | ✅ | Replaces Watchmen 2009 (uncertain); WB/Max DV confirmed, 4:3 IMAX desaturation |
| Joker | 2019 | Superhero | Val | ✅ | WB UHD confirmed DV — grimy 1970s Gotham, expressionistic brown-green decay |
| The Batman | 2022 | Superhero | Test | ✅ | WB UHD confirmed DV — perpetual rain-soaked noir, amber as only accent |
| 1917 | 2019 | Drama | Train | ✅ | Replaces Schindler's List (HDR10 only); Universal DV confirmed — one-take muddy WWI |
| Babylon | 2022 | Drama | Train | ✅ | Paramount DV confirmed — maximalist 1920s Hollywood, saturated torchlit parties |
| All Quiet on the Western Front | 2022 | Drama | Val | ✅ | Netflix original — DV confirmed, mud-brown naturalistic WWI grey grain |
| No Country for Old Men | 2007 | Drama | Test | ✅ | Replaces Taxi Driver 1976 (HDR10 only); Paramount DV confirmed — sun-bleached Texas desert |
| The Conjuring | 2013 | Horror | Train | ✅ | Replaces The Shining 1980 (uncertain); WB UHD DV confirmed — cold farmhouse dread |
| A Quiet Place | 2018 | Horror | Train | ✅ | Paramount UHD confirmed DV — muted natural-light rural, golden-hour grain |
| Smile | 2022 | Horror | Test | ✅ | Paramount UHD confirmed DV — deliberately flat clinical daylight horror |
| Spider-Man: Across the Spider-Verse | 2023 | Animation | Train | ✅ | Sony DV confirmed — each dimension wholly distinct, Impressionist to halftone |
| Encanto | 2021 | Animation | Val | ✅ | Replaces Despicable Me 2010 (uncertain); Disney UHD/Disney+ DV confirmed — vibrant magic-realist Colombian palette |
| Knives Out | 2019 | Comedy | Train | ✅ | Lionsgate UHD confirmed DV — autumnal gothic estate, warm amber interiors |
| The Grand Budapest Hotel | 2014 | Comedy | Test | ✅ | Replaces Groundhog Day 1993 (HDR10 only); Fox/Disney DV — pastel storybook palette |
| Gladiator | 2000 | Classic/Epic | Train | ✅ | Replaces Lawrence 1962 (HDR10 only); Paramount DV confirmed — warm arena, desaturated forest |
| Lord of the Rings: Fellowship | 2001 | Classic/Epic | Train | ✅ | WB 4K Extended confirmed DV — lush NZ vistas, deep chiaroscuro underground |

## Next steps

1. **Re-extract dataset** with SAT spatial features → `dv_dataset_sat.csv`
2. **Retrain + evaluate** — does frame 1334 MAE improve from 0.042?
3. **HDR10 experiment** — get HDR10 copy of The Little Things, apply ML model,
   compare pixel-level output to DV gold (VMAF/SSIM + difference heatmap)
4. **More titles** — 5 titles from the list above → validate XGBoost + generalisation
5. **libplacebo C integration** — register ML model as custom `pl_tone_map_function`
   using m2cgen-generated C code (no runtime dependencies)

## Notes on colour science

**DV Profile 5** uses a custom `ycc_to_rgb` matrix baked into the RPU (not standard
BT.2020). For HDMI LLDV capture (Oppo TV-Led mode), the player applies this matrix
before outputting to HDMI — DeckLink receives standard-looking pixels. The ML model
handles the tone curve; the colour matrix is handled by the source device.

For local DV file playback (mpv + libplacebo), the RPU is applied natively via
`pl_map_avframe_ex(map_dowi=true)` — no ML needed. The ML pipeline is specifically
for **HDR10 content** and **HDMI DV capture without RPU access**.
