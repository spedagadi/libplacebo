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

**Full title list:**

| Title | Year | Genre | Split | Visual rationale |
|---|---|---|---|---|
| John Wick: Chapter 4 | 2023 | Action | Train | Neon geometry — Sacré-Cœur and Osaka nightclub as architectural choreography |
| Top Gun: Maverick | 2022 | Action | Train | Bright IMAX aerial — F-18s against sunlit ocean and mountain corridors |
| Atomic Blonde | 2017 | Action | Train | Cold War Berlin neon cyan/magenta with heavy grain and deep shadow |
| Collateral | 2004 | Action | Train | Digital video LA night — ambient-lit blue-grey impossible on film |
| Raiders of the Lost Ark | 1981 | Action | Train | Warm amber celluloid adventure across sun-baked deserts and ruins |
| Warfare | 2025 | Action | Val | Handheld verité — desaturated sand-and-grey battlefield realism |
| The Northman | 2022 | Action | Val | Fog-drenched Icelandic near-monochrome, ash and flame contrast |
| Bullet Train | 2022 | Action | Test | Hyperrealistic neon-saturated Japanese pop-art Shinkansen interiors |
| Dune: Part Two | 2024 | Sci-Fi | Train | Burnt-amber IMAX desert; near-monochromatic sandworm sequences |
| The Matrix | 1999 | Sci-Fi | Train | Green-tinted digital world vs warm incandescent reality |
| 2001: A Space Odyssey | 1968 | Sci-Fi | Train | Clinical white-on-black vacuum vs psychedelic Stargate sequence |
| Alien: Romulus | 2024 | Sci-Fi | Train | Deep industrial shadow — wet corrugated metal in near-total darkness |
| Godzilla Minus One | 2023 | Sci-Fi | Val | Postwar muted grey + B&W mode; kaiju destruction at human scale |
| Hunger Games: Ballad | 2023 | Sci-Fi | Test | Capitol pastel gold/white vs drab grey District — hard chromatic binary |
| Black Panther | 2018 | Superhero | Train | Afrofuturist purples, golds, neon waterfalls — unique MCU visual identity |
| Watchmen | 2009 | Superhero | Train | Desaturated brown urban decay + saturated primary-colour costumes |
| Joker | 2019 | Superhero | Val | Grimy 1970s Gotham, film grain, brown-green decay, expressionistic |
| The Batman | 2022 | Superhero | Test | Perpetual rain-soaked noir — near-monochrome with amber as only accent |
| Schindler's List | 1993 | Drama | Train | B&W realism + singular red-coat device — zero chroma baseline |
| Babylon | 2022 | Drama | Train | Maximalist 1920s Hollywood — saturated torchlit parties, kinetic camera |
| All Quiet on the Western Front | 2022 | Drama | Val | Mud-brown naturalistic WWI — grey sky and cold grain throughout |
| Taxi Driver | 1976 | Drama | Test | Overexposed sodium-vapour NYC — Scorsese's grimy nocturnal urban purgatory |
| The Shining | 1980 | Horror | Train | Cold symmetrical Kubrick — clinical pastels punctuated by red bursts |
| A Quiet Place | 2018 | Horror | Train | Muted natural-light rural — golden-hour grain, silence over jump-scares |
| Smile | 2022 | Horror | Test | Deliberately flat clinical daylight — suburban horror as medical procedural |
| Spider-Man: Across the Spider-Verse | 2023 | Animation | Train | Each dimension wholly distinct — Impressionist, manga, LEGO, halftone |
| Despicable Me | 2010 | Animation | Val | Bright primary-colour CGI — clean suburban whites, saturated Minion yellow |
| Knives Out | 2019 | Comedy | Train | Autumnal gothic estate — warm amber lantern-lit widescreen interiors |
| Groundhog Day | 1993 | Comedy | Test | Classic Hollywood winter — natural light, white snow, warm incandescent |
| Lawrence of Arabia | 1962 | Classic/Epic | Train | 65mm Wadi Rum — no parallel for golden heat shimmer on celluloid |
| Lord of the Rings: Fellowship | 2001 | Classic/Epic | Train | Lush NZ vistas + deep chiaroscuro underground — classical wide-format |

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
