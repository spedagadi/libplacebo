# ML Dynamic Tone-Mapping for Dolby Vision

**Status (2026-08-14)**: 🚧 **Refactored to libplacebo-based feature extraction**

ML pipeline that predicts per-scene DV RPU piecewise polynomial coefficients from
decoded frame statistics, enabling DV-quality tone mapping for HDR10 content and
stripped/HDMI DV streams where the RPU is unavailable.

---

## 🆕 New Architecture (August 2026)

**Critical change**: Feature extraction moved from Python/numpy to libplacebo C library to eliminate train/test feature skew.

### Why the Change?

Training with Python/numpy and inference with libplacebo GPU created risk of silent model degradation due to:
- Different percentile interpolation methods
- Different SAT zone boundary rounding
- Different PQ normalization precision

**Solution**: Single source of truth - both training and inference use identical libplacebo C code.

### Native gamma inference

The trained gamma model can be exported for native C inference:

```bash
python ml/export_gamma_model_native.py \
  --input F:/DTMModelData/xgb_gamma_model.pkl \
  --model-output xgb_gamma_model.plxgb \
  --manifest-output xgb_gamma_model.manifest.json
```

The `.plxgb` artifact contains the 500-tree XGBoost regressor and the
manifest records the ordered 88-feature contract. Native inference is exposed
by `pl_ml_model_create()` and `pl_ml_model_predict()` in libplacebo. The
`pl_ml_model_eval` tool accepts an 88-float feature vector for parity testing.

### New Files (Start Here)

| Document | Purpose |
|---|---|
| **[PLAN.md](PLAN.md)** | Complete architecture + training strategy (755 lines) |
| **[EXTRACTION_INTEGRATION.md](EXTRACTION_INTEGRATION.md)** | Integration guide for libplacebo feature extraction |
| **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** | Current status + next steps |

### Quick Start (New Pipeline)

```bash
# 1. Build libplacebo with ML feature extraction
cd c:/Code/libplacebo
meson setup build
meson compile -C build pl_extract_features

# 2. Extract features for training (uses libplacebo + Python RPU parsing)
python tools/dv_metadata_extract.py INPUT.mkv -o dataset.csv --sample-fps 1

# 3. Train MLP model (multi-task: DTM curve + L2 trim)
python ml/train_mlp.py --data-dir F:/DTMModelData/train --output models/ml_dtm_v1.pt

# 4. Export to C for inference
python ml/export_model.py --model models/ml_dtm_v1.pt --output ml/ml_model.c
```

**Key changes**:
- **77 features** (was 27) - added multi-scale SAT (3×3 + 5×5)
- **Multi-task learning** - predicts curve + L2 color trim simultaneously
- **libplacebo feature extraction** - ensures bit-exact train/inference matching
- **RPU parsing stays in Python** - training-only, zero skew risk

---

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

## Results (prototype baseline — The Little Things 2021, WEB-DL P5)

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

Scale targets for XGBoost and cross-title generalisation:

| Titles | ~Scenes | XGBoost? | Cross-title generalisation? |
|---|---|---|---|
| 1 (current) | 919 | No | No |
| 5 | ~5k | Borderline | Partial |
| 10 | ~10k | Yes | Yes |
| 20+ | ~20k+ | Definitely | Strong |

### Dataset inventory — confirmed DV titles (scanned Aug 2026)

**DV verification method:** All on-disk titles stream-probed via UNSPEC62 RPU NAL scan and/or `ffprobe` DOVI configuration record. No `?` profiles remain.

**Status key:** ✅ Ready to extract · ⚠️ Needs work · ⬇️ Download needed  
**DV key:** ✅ Stream-verified · ❌ No DV  
**Format:** BDMV = complete disc folder · ISO = disc image · MKV = remux/encode  
**Profile support:**
- **P5** (pure DV single layer) — **primary training format.** Streaming/WEB-DL seasonal content. Confirmed real, non-identity per-scene polynomials — the actual DTM curve authored for the streaming encode.
- **P7** (dual-layer BL+EL) — **excluded from training** (Aug 2026 finding: all BDMV/disc titles tested show 100% identity luma polynomials — HDR10 BL is already tone-mapped, DV is colour-matrix-only on disc). Retained only for **Test** — community-benchmark visual rendering, not curve training.
- **P8** (HDR10-compatible single layer) — same disc-identity finding as P7; visual rendering only.

> **Extractor — source auto-discovery (Aug 2026):** `dv_metadata_extract.py` accepts any source format. For P5 streaming files, pass the per-episode `.mkv` directly. For legacy BDMV/disc titles, pass the disc folder path — `discover_sources()` probes for UNSPEC62 RPU NALs in v:0 and v:1, finds the EL stream automatically, and calibrates BDMV timestamp offsets. ISOs: mount via `Mount-DiskImage` in PowerShell, then pass the mount point (`E:\`) as the folder input.

> **Source quality note (revised Aug 2026):** Disc remuxes (BDMV/P7/P8) looked like the right training source, but the HDR10 base layer on disc is already tone-mapped in mastering — DV there is colour-matrix-only, so `comp[0]` is always the identity polynomial. **P5 streaming encodes are the only confirmed source of real per-scene DTM polynomials** — the streaming colorist authors the curve directly into the P5 RPU against that same encode's pixel statistics. Disc titles are kept only as **Test** visual-rendering benchmarks (colour-matrix correctness, not curve prediction).

#### Movies — WEB-DL P5 (unassigned to a split yet)

> These validated the P5 hypothesis before the seasonal corpus below existed. Real per-scene polynomials confirmed on The Little Things; the other three are unconfirmed profile (`?`) and need a stream probe before use.

| Title | Year | Location | Format | DV | Profile | Note |
|---|---|---|---|---|---|---|
| The Little Things | 2021 | `D:\Jdownloader\TeLtlTig...` | WEB-DL mp4 | ✅ | 5 | 2060 scenes extracted; original hypothesis validation |
| 28 Years Later: The Bone Temple | 2026 | `G:\28.Years.Later.The.Bone.Temple...mkv` | WEB-DL MKV | ✅ | ? | MA/HBO streaming encode — needs profile probe |
| A House of Dynamite | 2025 | `G:\A House of Dynamite...mkv` | WEB-DL MKV | ✅ | ? | Netflix streaming encode — needs profile probe |
| Predator: Killer of Killers | 2025 | `G:\Predator - Killer of Killers...mkv` | WEB-DL MKV | ✅ | ? | Disney+ streaming encode — needs profile probe |

#### On disk / to download — G:\Dataset\ P5 streaming seasons (primary corpus)

**Storage layout:** `G:\Dataset\<Title>\S0#E0#.mkv` — one folder per show/season, per-episode MKV files.
**Status:** ✅ Ready to extract · ⬇️ Download needed

---

##### Train (P5 streaming — 6 shows)

Picked to spread bright/dark and colour-cast extremes across the set — not just genre variety.

| Title | Season | Genre | HDR content type | Location | Status |
|---|---|---|---|---|---|
| Ted Lasso | S03 | Comedy | Naturalistic daylight pitch/office; floodlit night matches — dynamic-range **baseline**, few extremes | `G:\Dataset\Ted.Lasso.S03...\` | ✅ 12 eps on disk |
| For All Mankind | S05 | Sci-Fi/Drama | Spacewalk sun glare vs void black (extreme highlight/shadow pair); console/HUD glow; Mars daylight | `G:\Dataset\For.All.Mankind.S05...\` | ✅ 10 eps on disk |
| The Witcher | **S04** ⚠️ | Fantasy | Torchlit castles/dungeons (low-key); magic-FX bright particles; forest daylight | `G:\Dataset\The.Witcher.S04...\` | ✅ 8 eps on disk |
| Andor | S02 | Sci-Fi/Action | Industrial low-key interiors; blaster/ship explosions; neon-lit Coruscant night city | `G:\Dataset\Andor.S02...\` | ✅ 12 eps on disk |
| Stranger Things | S05 | Horror/Sci-Fi | Upside Down practical darkness (deep black, sparse highlight); 80s neon; climactic fire/explosions | `G:\Dataset\Stranger.Things.S05\` | ⬇️ downloading |
| The Mandalorian | S01 | Sci-Fi/Action | Tatooine desert sun (extreme highlight); dark cantina interiors; lava/explosion finale | `G:\Dataset\The.Mandalorian.S01...\` | ✅ 8 eps on disk |

> ⚠️ **Witcher season correction (Aug 2026):** planned as S02, but S04 is what downloaded — kept as-is rather than re-downloading. `batch_extract.py` and the short name (`the_witcher_s04`) already reflect this.

---

##### Val (P5 streaming — 4 shows)

Chosen to cover content types **absent from Train** — nature daylight, underwater, and non-photoreal animation grading — so held-out performance isn't just "more of the same six shows."

| Title | Season | Genre | HDR content type | Location | Status |
|---|---|---|---|---|---|
| Prehistoric Planet | S03 | Documentary/Nature | Savanna sun extremes; underwater desaturated blue-green; night bioluminescence; volcanic lava | `G:\Dataset\Prehistoric.Planet.2022.S03...\` | ✅ 5 eps on disk |
| Born to Be Wild | S01 | Documentary/Nature | Global wildlife daylight — arctic snow/ice (high-key diffuse), desert, jungle canopy shade | `G:\Dataset\Born.to.Be.Wild.2025.S01...\` | ✅ 6 eps on disk |
| WondLa | S03 | Animation | Stylized high-saturation alien vistas — non-photoreal grading, stresses colour-cast generalisation | `G:\Dataset\WondLa.S03...\` | ✅ 6 eps on disk |
| Monarch: Legacy of Monsters | S02 | Sci-Fi/Action | Kaiju-scale fire/explosions; bioluminescent creature glow; dark urban destruction | `G:\Dataset\Monarch.Legacy.of.Monsters.S02\` | ⬇️ downloading |

**Coverage check (Train + Val combined):** explosions/fire (Witcher, Andor, Stranger Things, Mandalorian, Monarch), extreme highlight/glare (For All Mankind, Mandalorian, Prehistoric Planet), dark low-key interiors (Witcher, Andor, Stranger Things, Mandalorian), neon/night urban (Andor, Stranger Things, Monarch), natural daylight baseline (Ted Lasso, Prehistoric Planet, Born to Be Wild), underwater (Prehistoric Planet only), snow/ice (Born to Be Wild only), non-photoreal animation (WondLa only). **Gaps to watch:** underwater, snow/ice, and stylized-animation grading each rely on a single title — if that title fails to extract cleanly (profile mismatch, block-addition errors, etc.) that content type drops out entirely with no backup.

---

##### Calibration — TBD

> Previous disc-based Cal titles (Mad Max: Fury Road, The Revenant, Sicario) are deprioritized — disc DV is colour-matrix-only, no polynomial diversity to tune against. A Cal set should be carved from held-out episodes of the Train/Val shows above once extraction confirms per-scene polynomial diversity, rather than sourced separately.
>
> When selecting which held-out episodes to use, match the original Cal rationale to content type rather than picking arbitrarily: **λ** (segment-boundary weight) wants rapid-cut, mixed-content episodes (Mandalorian or Stranger Things action episodes); **Q** (within-scene smoothness) wants slow, gradually-changing illumination (Ted Lasso dialogue-heavy episodes, Prehistoric Planet long wildlife takes); **R** (highlight rolloff) wants scenes with precise, deliberate highlight control (For All Mankind console/glare scenes, Witcher torchlit interiors).

---

##### Test (community benchmarks — BDMV, disc-based visual rendering only)

> Kept from the original disc corpus — disc DV here is used for AVForums/AVS-style visual comparison (colour-matrix correctness), not curve training. `*` = primary community benchmark.

> These are titles AVForums / AVS Forum members actively post DV comparisons for. Results shown on these titles to the community prove generalisation, not memorisation. `*` = primary community benchmark. Since these render visually rather than train the curve, content-type diversity here matters for **exposing rendering bugs** (colour-matrix edge cases, block-addition parsing) rather than for model generalisation.

| Title | Year | Genre | HDR content type | Format | DV | P | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| **Dune: Part Two** * | 2024 | Sci-Fi | Desert extreme highlight; dark cave/night battles; nuclear explosion flash | BDMV | ✅ | **7** | ✅ | P7 v:1; EL auto-discovered |
| **Top Gun: Maverick** * | 2022 | Action | Bright aerial sky glare; cockpit HUD glow; explosions | ISO | ✅ | **7** | ✅ | P7 v:1 confirmed; mount ISO → pass mount point as folder |
| **Godzilla Minus One** * | 2023 | Sci-Fi | Dark night ocean; atomic-breath extreme highlight; fire/explosions | BDMV | ✅ | **7** | ✅ | JPN disc; EL auto-discovered |
| **Civil War** * | 2024 | War/Action | War explosions/fire; daylight urban; night raids | MKV | ✅ | **7** | ✅ | P7; identity polynomial (visual rendering demo only) |
| **Spider-Man: ATSV** * | 2023 | Animation | Highly stylized neon, mixed animation styles/colour grading | MKV | ✅ | **7** | ✅ | P7; identity polynomial; German audio; visual demo only |
| **John Wick: Ch4** * | 2023 | Action | Neon-lit night action; dark practical interiors | MKV | ✅ | **7** | ✅ | P7; identity polynomial; German audio; visual demo only |
| Predator Badlands | 2025 | Sci-Fi/Action | Dark jungle; practical low-light creature scenes | BDMV | ✅ | **7** | ✅ | P7 v:1; EL auto-discovered |
| Gladiator II | 2024 | Action/Epic | Arena daylight; torchlit interiors; blood/fire | ISO | ✅ | **7** | ✅ | P7 v:1 confirmed; mount ISO → pass mount point as folder |
| No Time to Die | 2021 | Action | Daylight exteriors; night action; explosions | ISO | ✅ | **7** | ✅ | P7 v:1 confirmed; mount ISO → pass mount point as folder |
| **The Batman** * _(download)_ | 2022 | Superhero/Noir | Perpetual rain noir — extreme low-key darkness; contrast to WW1984 | — | ✅ | — | ⬇️ | WB UHD |
| **Blade Runner 2049** * _(download)_ | 2017 | Sci-Fi/Noir | Neon noir extreme; desert daylight extremes (Vegas ruins) | — | ✅ | — | ⬇️ | Most-discussed AVForums HDR benchmark; not on disk |
| **Joker** * _(download)_ | 2019 | Superhero/Drama | Grimy low-key urban; neon subway | — | ✅ | — | ⬇️ | Community DV discussion title |

## Extraction Pipeline — Data-Driven Stratification (Phase 0 → Stage 1 → Stage 2)

Traditional "extract all episodes" wastes terabytes on redundant mid-tone dialogue frames. XGBoost needs **balanced feature space** — edge cases (shadow floors, HDR peaks, colorist interventions) must have equal weight to neutral scenes. The RPU itself is the oracle: L1 delta spikes, heavy L2 trim corrections, and active L8 saturation blocks explicitly mark "this scene is hard."

### Phase 0: RPU Stratification (Metadata-Only, Fast)

Lightweight RPU scanner — reads NAL metadata only (no pixel decode), runs in seconds per episode.

```bash
# Scan all titles (outputs stratification_manifest.csv with L1/L2/L8 per scene)
python tools/rpu_stratify.py --output F:/DTMModelData/stratification_manifest.csv

# Or single title
python tools/rpu_stratify.py --title ted_lasso_s03
```

**Output**: `stratification_manifest.csv` — per-scene L1 min/max/avg, L1 delta (scene-to-scene luma jump), L2 trim variance (colorist intervention magnitude), L8 saturation activity.

### Bucket Selection: 5-Variance Target Quotas

Rank scenes by variance, allocate ~150-200 high-information scenes per title across 5 buckets:

| Bucket | Filter criterion | Target | Why XGBoost needs it |
|---|---|---|---|
| **Deep shadow floor** | `L1_min < 50` AND `L1_avg < 200` | 25 scenes | Shadow detail preservation, prevents clipping dark regions |
| **High DR peaks** | `L1_max > 2000` | 35 scenes | Highlight rolloff, compression without blowing specular extremes |
| **Heavy trim variance** | `L2_trim_variance > 90th percentile` | 35 scenes | Human colorist interventions — the model learns where linear formulas failed |
| **High L1 delta** | `L1_delta > 500` | 25 scenes | Extreme transitions (dark cockpit → blinding desert), aggressive curve shifts |
| **Mid-tone neutral** | `L1_delta < 100`, fill remainder | 50-60 scenes | Baseline — prevents over-correcting standard daylight frames |

```bash
# Apply bucket quotas, output prioritized episode/scene list
python tools/rpu_bucket_select.py \
    --input F:/DTMModelData/stratification_manifest.csv \
    --output F:/DTMModelData/priority_extraction_list.csv \
    --quota 180
```

**Output**: `priority_extraction_list.csv` — filtered to ~180 scenes/title, tagged with bucket assignments. This becomes the extraction plan for Stage 1.

### Stage 1: Manifest Extraction (RPU + Polynomials, No Pixels)

Fast RPU-only pass — extracts polynomial coefficients + L1 metadata for all frames in prioritized episodes (or full episodes if running exhaustive first-pass). No pixel decode yet.

```bash
# All prioritized titles
python tools/batch_extract.py --workers 2

# Single title
python tools/batch_extract .py --title ted_lasso_s03
```

**Output**: `F:\DTMModelData\{split}\{title}.csv` — concatenated per-episode CSVs with RPU polynomial + L1 + scene_refresh for every frame.

### Stage 2: Pixel Extraction (Full Features)

Decode pixels for prioritized scenes only (filtered by `priority_extraction_list.csv` scene IDs), extract histogram + SAT features. Slower; optionally use `--nvdec` for 4× GPU speedup.

```bash
python tools/batch_extract.py --full-pixels --nvdec --workers 1
```

**Output**: Same CSVs, now with `maxscl`, `average_maxrgb`, `distrib_val_3..8`, `zone_mean/max_rR_cC` columns filled.

### Critical: Pixel Pipeline Safety

The current decoder outputs **raw YUV** (`yuv420p10le`) via plain ffmpeg `scale` — **no RPU is applied**. libplacebo's `pl_peak_detect` histogram runs on the untouched HDR PQ frame. If the pipeline accidentally bakes DV processing into the video before libplacebo calculates features, the model trains on already-transformed data and learns nothing.

**Verified safe**: `dv_metadata_extract.py` line 491-505 uses `-vf scale` only, no `-vf dovi` or `map_dowi=true`.

### Next Steps (Post-Extraction)

1. **Confirm polynomial diversity per title** — non-identity, non-trivial segment counts — before committing a title to Train/Val
2. **Retrain + evaluate** on the P5 stratified corpus; compare held-out MAE against the single-title WEB-DL prototype baseline
3. **Carve a Calibration split** from held-out episodes (λ/Q/R tuning) once per-scene polynomial diversity is confirmed
4. **Download P5 community benchmarks** (Dune Part Two, Joker, Blade Runner 2049 WEB-DL variants) for blind Test
5. **libplacebo C integration** — register ML model as custom `pl_tone_map_function` using m2cgen-generated C code (no runtime dependencies)

## Notes on colour science

**DV Profile 5** uses a custom `ycc_to_rgb` matrix baked into the RPU (not standard
BT.2020). For HDMI LLDV capture (Oppo TV-Led mode), the player applies this matrix
before outputting to HDMI — DeckLink receives standard-looking pixels. The ML model
handles the tone curve; the colour matrix is handled by the source device.

For local DV file playback (mpv + libplacebo), the RPU is applied natively via
`pl_map_avframe_ex(map_dowi=true)` — no ML needed. The ML pipeline is specifically
for **HDR10 content** and **HDMI DV capture without RPU access**.
