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
- **P7** (dual-layer BL+EL) — primary training format; all USA UHD discs and COMPLETE.UHD.BLURAY rips. EL carries the RPU, auto-discovered via `discover_sources()`.
- **P8** (HDR10-compatible single layer) — hybrid remuxes (disc video + streaming RPU); supported, lower priority.
- **P5** (pure DV single layer) — streaming/WEB-DL only; **excluded from training corpus** (pixel stats don't match the mastering environment). Used only for initial hypothesis testing.

> **Extractor — source auto-discovery (Aug 2026):** `dv_metadata_extract.py` now accepts any source format. Pass a disc folder path for BDMV titles — `discover_sources()` probes for UNSPEC62 RPU NALs in v:0 and v:1, finds the EL stream automatically, and calibrates BDMV timestamp offsets. Tested: Rush P7 MKV (126 rows) and Spotlight P7 BDMV (124 rows), both with full polynomial + pixel + SAT features. ISOs: mount via `Mount-DiskImage` in PowerShell, then pass the mount point (`E:\`) as the folder input.

> **Source quality note:** Training data should come from **disc remuxes only** (BDMV/MKV remux). WEB-DL and streaming encodes are re-compressed from a different master than the one the colorist used when authoring the DV RPU metadata. The pixel statistics extracted from a WEB-DL do not faithfully represent the feature distribution the DV colorist was responding to — this adds noise to the feature→label relationship. WEB-DL titles may be used for **hypothesis testing and prototyping** but should not be part of the training corpus.

#### Hypothesis baseline — WEB-DL (prototype only, excluded from training corpus)

| Title | Year | Location | Format | DV | Profile | Note |
|---|---|---|---|---|---|---|
| The Little Things | 2021 | `D:\Jdownloader\TeLtlTig...` | WEB-DL mp4 | ✅ | 5 | 2060 scenes extracted; initial hypothesis validation only |
| 28 Years Later: The Bone Temple | 2026 | `G:\28.Years.Later.The.Bone.Temple...mkv` | WEB-DL MKV | ✅ | ? | MA/HBO streaming encode |
| A House of Dynamite | 2025 | `G:\A House of Dynamite...mkv` | WEB-DL MKV | ✅ | ? | Netflix streaming encode |
| Predator: Killer of Killers | 2025 | `G:\Predator - Killer of Killers...mkv` | WEB-DL MKV | ✅ | ? | Disney+ streaming encode |

#### On disk — G:\ disc remuxes

**Source tiers:** P = Pure disc (pixel+RPU from same master) · H = Hybrid disc (disc video, streaming RPU) · W = WEB-DL (excluded)  
**Status:** ✅ Ready · ⚠️ Needs work · 🔍 Unverified (ISO)

---

##### Calibration (0 on disk — downloads needed)

> **Aug 2026 finding:** All 5 original calibration titles (Everest, Hurt Locker, Troy DC, John Wick Ch4, Kingdom of Heaven DC) are **P7/P8 MKV remuxes with 100% identity luma polynomials** — DV is colour-matrix-only for these titles. Useless for Bayesian HPO of the DTM luma polynomial model. Moved to Reserve-MKV below.
>
> New calibration titles must be BDMV pure disc with real per-scene luma polynomials (confirmed by non-trivial polynomial diversity after extraction).

| Title | Year | Genre | Format | Priority | Split | Calibration rationale |
|---|---|---|---|---|---|---|
| **Mad Max: Fury Road** | 2015 | Action | BDMV | ⬇️ **Critical** | Cal | Hundreds of rapid cuts (best for λ tuning); extreme highlights vs shadow; Warner COMPLETE.UHD.BLURAY |
| **The Revenant** | 2015 | Drama/Adventure | BDMV | ⬇️ **Critical** | Cal | Slow pacing, gradual illumination (best for Q within-scene); extreme snow/fire contrast; Fox UHD |
| **Sicario** | 2015 | Thriller | BDMV | ⬇️ **Critical** | Cal | Precise studio lighting, tension-driven cuts (calibrates R); Lionsgate UHD; fills Thriller calibration gap |

---

##### Train (9 titles — BDMV only)

> **Aug 2026:** All P7/P8 MKV remuxes removed — confirmed 100% identity luma polynomials, useless for DTM training. BDMV-only corpus going forward.

| Title | Year | Genre | Format | DV | P | Status | Notes |
|---|---|---|---|---|---|---|---|
| Alien: Romulus | 2024 | Sci-Fi/Horror | BDMV | ✅ | **7** | ✅ | EL 5528 kbps; extraction in progress |
| Atomic Blonde | 2017 | Action/Spy | BDMV | ✅ | **7** | ✅ | Extraction in progress |
| Furiosa | 2024 | Action | BDMV | ✅ | **7** | ✅ | EL 2106 kbps; extraction in progress |
| Warfare | 2025 | War/Action | BDMV | ✅ | **7** | ✅ | Extraction pending |
| Spotlight | 2015 | Drama | BDMV | ✅ | **7** | ✅ | EL 8522 kbps — high bitrate EL, expected real polynomials |
| Zodiac | 2007 | Crime/Thriller | BDMV | ✅ | **7** | ✅ | EL 14879 kbps — highest EL bitrate, priority extraction |
| Wonder Woman 1984 | 2020 | Superhero | BDMV | ✅ | **7** | ✅ | Extraction pending |
| First Blood | 1982 | Action | BDMV | ✅ | **7** | ✅ | Extraction pending |
| 28 Years Later | 2025 | Horror | BDMV | ✅ | **7** | ✅ | Extraction pending |

---

##### Val (6 titles — BDMV only)

| Title | Year | Genre | Format | DV | P | Status | Notes |
|---|---|---|---|---|---|---|---|
| How to Train Your Dragon | 2025 | Animation | BDMV | ✅ | **7** | ✅ | EL 5917 kbps |
| MI: The Final Reckoning | 2025 | Action | BDMV | ✅ | **7** | ✅ | EL 4029 kbps |
| The Invisible Man | 2020 | Horror/Sci-Fi | BDMV | ✅ | **7** | ✅ | EL 7082 kbps |
| Tron: Legacy | 2010 | Sci-Fi | BDMV | ✅ | **7** | ✅ | Extraction pending |
| F1: The Movie | 2025 | Sport/Drama | BDMV | ✅ | **7** | ✅ | Extraction pending |
| Ballerina | 2025 | Action | BDMV | ✅ | **7** | ✅ | Extraction pending |

---

##### Test (community benchmarks — never used in Train/Val)

> These are titles AVForums / AVS Forum members actively post DV comparisons for. Results shown on these titles to the community prove generalisation, not memorisation. `*` = primary community benchmark.

| Title | Year | Genre | Format | DV | P | Status | Notes |
|---|---|---|---|---|---|---|---|
| **Dune: Part Two** * | 2024 | Sci-Fi | BDMV | ✅ | **7** | ✅ | P7 v:1; EL auto-discovered |
| **Top Gun: Maverick** * | 2022 | Action | ISO | ✅ | **7** | ✅ | P7 v:1 confirmed; mount ISO → pass mount point as folder |
| **Godzilla Minus One** * | 2023 | Sci-Fi | BDMV | ✅ | **7** | ✅ | JPN disc; EL auto-discovered |
| **Civil War** * | 2024 | War/Action | MKV | ✅ | **7** | ✅ | P7; identity polynomial (visual rendering demo only) |
| **Spider-Man: ATSV** * | 2023 | Animation | MKV | ✅ | **7** | ✅ | P7; identity polynomial; German audio; visual demo only |
| **John Wick: Ch4** * | 2023 | Action | MKV | ✅ | **7** | ✅ | P7; identity polynomial; German audio; visual demo only |
| Predator Badlands | 2025 | Sci-Fi/Action | BDMV | ✅ | **7** | ✅ | P7 v:1; EL auto-discovered |
| Gladiator II | 2024 | Action/Epic | ISO | ✅ | **7** | ✅ | P7 v:1 confirmed; mount ISO → pass mount point as folder |
| No Time to Die | 2021 | Action | ISO | ✅ | **7** | ✅ | P7 v:1 confirmed; mount ISO → pass mount point as folder |
| **The Batman** * _(download)_ | 2022 | Superhero/Noir | — | ✅ | — | ⬇️ | Rain noir — contrast to WW1984; WB UHD |
| **Blade Runner 2049** * _(download)_ | 2017 | Sci-Fi/Noir | — | ✅ | — | ⬇️ | Most-discussed AVForums HDR benchmark; not on disk |
| **Joker** * _(download)_ | 2019 | Superhero/Drama | — | ✅ | — | ⬇️ | Grimy Gotham; community DV discussion title |

---

##### Reserve — BDMV ISOs (usable for visual rendering, polynomial TBD)

| Title | Year | Genre | Format | DV | P | Notes |
|---|---|---|---|---|---|---|
| MI: Dead Reckoning Pt 1 | 2023 | Action | ISO | ✅ | **7** | ✅ | P7 v:1 confirmed; use if more Action needed |

##### Reserve — MKV only (identity polynomial confirmed — visual rendering only)

> These titles have confirmed 100% identity luma polynomials. Usable in the Streamlit viewer for frame comparison (DV colour matrices apply correctly) but provide no useful training signal for the DTM luma polynomial model.

| Title | Notes |
|---|---|
| Everest (P8 Hybrid MKV) | Identity |
| Hurt Locker (P7 MKV, German custom dub) | Identity + MKV block addition errors (pts<500 fail in dv_render) |
| Troy DC (P7 MKV) | Identity; renders OK for pts>500 |
| John Wick Ch4 (P7 MKV, German) | Identity; 2-pivot single segment |
| Kingdom of Heaven DC (P7 MKV, German) | Identity; 2-pivot single segment |
| Rush (P7 MKV) | Identity; MKV block addition errors |
| Pacific Rim / Prometheus / The Creator / KotPotA (P8 Hybrid MKV) | Identity |
| Weapons (P7 Hybrid MKV) | Identity |

#### On disk — no DV (skip for training)

DV verified by stream probe (RPU NAL scan and/or BDNFO EL track check). All confirmed HDR10 only.

| Title | Year | Location | HDR | How confirmed |
|---|---|---|---|---|
| Oppenheimer | 2023 | `G:\Oppenheimer...ESiR` | HDR10 | BDNFO — single video track; EUR disc (US disc has DV) |
| Se7en | 1995 | `G:\Se7en...` | HDR10 | BDNFO — single video track |
| Last Breath | 2025 | `G:\Last.Breath...` | HDR10 | BDNFO — single video track |
| Heat | 1995 | `G:\Heat.1995...mkv` | HDR10+ | Filename — no DV tag; HDR10+ only |
| Nope | 2022 | `G:\Nope 2022...mkv` | HDR10 | Filename — no DV tag |
| Exodus: Gods and Kings | 2014 | `G:\Exodus Gods and Kings.m2ts` | HDR10 | Single m2ts, no RPU NALs |
| V for Vendetta | 2005 | `G:\V.for.Vendetta...iso` | HDR10 | ISO probed — 0 RPU NALs v:0; v:1 is H.264 BD combo track |
| Wonder Woman 2017 | 2017 | `G:\Wonder.Woman.2017...iso` | HDR10 | ISO probed — 0 RPU NALs; single HEVC stream; pre-DV Warner press |
| American Sniper | 2014 | `G:\American.Sniper...MTeam` | HDR10 | BDNFO — single video track; pre-DV Warner 2014 |
| Edge of Tomorrow | 2014 | `G:\Edge.of.Tomorrow...MAXAGAZ` | HDR10 | BDNFO — single video track; pre-DV Warner 2014 |
| Monkey Man | 2024 | `G:\Monkey.Man...B0MBARDiERS` | HDR10 | BDNFO — single video track; Universal disc HDR10 only |

#### Download needed

`*` = community benchmark (Test only). All downloads must be **COMPLETE.UHD.BLURAY** or equivalent pure disc BDMV format — MKV remuxes confirmed as identity polynomial and excluded from training.

| # | Title | Year | Genre | Format | Priority | Split | Why needed |
|---|---|---|---|---|---|---|---|
| C1 | **Mad Max: Fury Road** | 2015 | Action | BDMV | **Critical** | **Cal** | Replaces 5 identity-polynomial MKV calibration titles; hundreds of cuts (λ tuning) + extreme dynamic range |
| C2 | **The Revenant** | 2015 | Drama/Adv | BDMV | **Critical** | **Cal** | Slow pacing + snow/fire extremes; calibrates Q within-scene; Fox/Disney UHD |
| C3 | **Sicario** | 2015 | Thriller | BDMV | **Critical** | **Cal** | Precise studio lighting, tension-driven cuts; Lionsgate UHD; fills Thriller calibration gap |
| D1 | Knives Out | 2019 | Comedy | BDMV | **Critical** | Train | Only Comedy candidate; Lionsgate UHD |
| D2 | A Quiet Place | 2018 | Horror | BDMV | **Critical** | Train | Need 2nd Horror Train title |
| D3 | **The Batman** * | 2022 | Superhero/Noir | BDMV | High | **Test** | Community benchmark; perpetual rain noir |
| D4 | **Blade Runner 2049** * | 2017 | Sci-Fi/Noir | BDMV | High | **Test** | Most-discussed AVForums HDR benchmark |
| D5 | **Joker** * | 2019 | Superhero/Drama | BDMV | High | **Test** | Community DV benchmark; grimy Gotham |
| D6 | 1917 | 2019 | Drama/War | BDMV | Medium | Train | Universal UHD; fills pure Drama Train slot |
| D7 | The Grand Budapest Hotel | 2014 | Comedy | BDMV | Medium | Val | Fox/Disney; fills Comedy Val |
| D8 | All Quiet on the Western Front | 2022 | Drama/War | BDMV | Medium | Val | Netflix DV; fills Drama Val |
| D9 | Interstellar | 2014 | Sci-Fi | BDMV | Low | Train | Paramount UHD; IMAX grain |
| D10 | Lord of the Rings: Fellowship | 2001 | Epic | BDMV | Low | Train | WB 4K Extended; Classic/Epic Train |
| D11 | Encanto | 2021 | Animation | BDMV | Low | Val | Disney+; fills Animation Val |

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
