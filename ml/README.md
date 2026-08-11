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

Scale targets for XGBoost and cross-title generalisation:

| Titles | ~Scenes | XGBoost? | Cross-title generalisation? |
|---|---|---|---|
| 1 (current) | 919 | No | No |
| 5 | ~5k | Borderline | Partial |
| 10 | ~10k | Yes | Yes |
| 20+ | ~20k+ | Definitely | Strong |

### Dataset inventory — confirmed DV titles (scanned Aug 2026)

**DV verification method:** MKVs probed via `ffprobe` DOVI configuration record (`side_data_type=DOVI configuration record` confirmed on all 13 MKVs). BDMV folders verified via BDNFO EL track presence. ISOs unverified — likely based on known disc specs.

**Status key:** ✅ Ready to extract · ⚠️ Needs work (see Notes) · ⬇️ Download needed  
**DV key:** ✅ Stream-verified · 🔍 Likely (known disc, not stream-probed) · ❌ No DV  
**Format:** BDMV = complete disc folder · ISO = disc image · MKV = remux/encode  
**Profile:** 5 = RPU in single layer (v:0) · 7 = BL+EL, RPU in EL (MKV: EL muxed into v:0; BDMV: EL is separate stream file) · 8 = HDR10-compatible single layer (v:0)

> **Extractor note — Profile 7 BDMV:** The RPU is in the EL stream file (`BDMV/STREAM/` — separate m2ts from the 4K BL). `dv_metadata_extract.py` must target the EL file, not the main title. Profile 7 MKVs mux BL+EL into a single `v:0` stream — `v:0` works and yields RPU NALs normally (verified: 123 RPU NALs/5s on Rush).

> **Source quality note:** Training data should come from **disc remuxes only** (BDMV/MKV remux). WEB-DL and streaming encodes are re-compressed from a different master than the one the colorist used when authoring the DV RPU metadata. The pixel statistics extracted from a WEB-DL do not faithfully represent the feature distribution the DV colorist was responding to — this adds noise to the feature→label relationship. WEB-DL titles may be used for **hypothesis testing and prototyping** but should not be part of the training corpus.

#### Hypothesis baseline — WEB-DL (prototype only, excluded from training corpus)

| Title | Year | Location | Format | DV | Profile | Note |
|---|---|---|---|---|---|---|
| The Little Things | 2021 | `D:\Jdownloader\TeLtlTig...` | WEB-DL mp4 | ✅ | 5 | 2060 scenes extracted; initial hypothesis validation only |
| 28 Years Later: The Bone Temple | 2026 | `G:\28.Years.Later.The.Bone.Temple...mkv` | WEB-DL MKV | ✅ | ? | MA/HBO streaming encode |
| A House of Dynamite | 2025 | `G:\A House of Dynamite...mkv` | WEB-DL MKV | ✅ | ? | Netflix streaming encode |
| Predator: Killer of Killers | 2025 | `G:\Predator - Killer of Killers...mkv` | WEB-DL MKV | ✅ | ? | Disney+ streaming encode |

#### On disk — G:\ disc remuxes

**Source tiers for training quality (best → worst):**
1. **Pure disc** — BDMV or MKV remux, pixel+RPU both from the same physical disc master
2. **Hybrid disc** — disc video + RPU sourced from streaming (RPU authored for slightly different encode; tolerable since both derive from the same colorist session)
3. **WEB-DL** — pixel+RPU both from streaming re-encode; excluded from training corpus

> **Test set integrity:** Test titles must be **community benchmark titles** — ones that AVForums, AVS Forum, and Blu-ray.com members actively post DV comparisons for. If a title appears in training, any results shown to the community on that title can be dismissed as memorisation. Titles marked `*` are community benchmarks and must never appear in Train or Val.

| # | Title | Year | Genre | Location | Format | DV | Profile | Tier | Status | Split | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Alien: Romulus | 2024 | Sci-Fi/Horror | `G:\Alien.Romulus...DOUHD` | BDMV | ✅ | 7 | Pure disc | ⚠️ | Train | EL 5528 kbps — use EL stream file for RPU |
| 2 | Furiosa | 2024 | Action | `G:\Furiosa...GLiMMER` | BDMV | ✅ | 7 | Pure disc | ⚠️ | Train | EL 2106 kbps — use EL stream file for RPU |
| 3 | Warfare | 2025 | War/Action | `G:\Warfare...TMT` | BDMV | ✅ | ? | Pure disc | ✅ | Train | Profile TBD — verify |
| 4 | Spotlight | 2015 | Drama | `G:\Spotlight...MTeam` | BDMV | ✅ | 7 | Pure disc | ⚠️ | Train | EL 8522 kbps — use EL stream file for RPU |
| 5 | Zodiac | 2007 | Crime/Thriller | `G:\Zodiac...CHDBits` | BDMV | ✅ | 7 | Pure disc | ⚠️ | Train | EL 14879 kbps — use EL stream file for RPU |
| 6 | Rush | 2013 | Sport/Drama | `G:\Rush...HDT` | MKV | ✅ | **7** | Pure disc | ✅ | Train | DOVI confirmed; EL in v:0 — 123 RPU NALs/5s |
| 7 | Wonder Woman 1984 | 2020 | Superhero | `G:\Wonder.Woman.1984...` | BDMV | ✅ | ? | Pure disc | ✅ | Train | Profile TBD — verify |
| 8 | First Blood | 1982 | Action | `G:\First.Blood...BLoz` | BDMV | ✅ | ? | Pure disc | ✅ | Train | Profile TBD — classic grain |
| 9 | Pacific Rim | 2013 | Sci-Fi/Action | `G:\Pacific Rim...REMUX.mkv` | MKV | ✅ | **8** | Hybrid disc | ✅ | Train | Disc video + streaming RPU; DOVI confirmed |
| 10 | Prometheus | 2012 | Sci-Fi/Horror | `G:\Prometheus...mkv` | MKV | ✅ | **8** | Hybrid disc | ✅ | Train | Disc video + streaming RPU; DOVI confirmed |
| 11 | The Creator | 2023 | Sci-Fi | `G:\The Creator...mkv` | MKV | ✅ | **8** | Hybrid disc | ✅ | Train | Disc video + streaming RPU; DOVI confirmed |
| 12 | Everest | 2015 | Drama/Adventure | `G:\Everest...mkv` | MKV | ✅ | **8** | Hybrid disc | ✅ | Train | Disc video + streaming RPU; DOVI confirmed |
| 13 | Kingdom of the Planet of the Apes | 2024 | Sci-Fi | `G:\Kingdom.of.the.Planet...mkv` | MKV | ✅ | **8** | Hybrid disc | ✅ | Train | Disc video + streaming RPU; DOVI confirmed |
| 14 | 28 Years Later | 2025 | Horror | `G:\28.Years.Later...TMT` | BDMV | ✅ | ? | Pure disc | ✅ | Train | Profile TBD — moved from Val; fills Horror Train |
| 15 | How to Train Your Dragon | 2025 | Animation | `G:\How.to.Train...B3LLUM` | BDMV | ✅ | 7 | Pure disc | ⚠️ | Val | EL 5917 kbps — use EL stream file for RPU |
| 16 | MI: The Final Reckoning | 2025 | Action | `G:\Mission.Impossible-The.Final...` | BDMV | ✅ | 7 | Pure disc | ⚠️ | Val | EL 4029 kbps — use EL stream file for RPU |
| 17 | The Invisible Man | 2020 | Horror/Sci-Fi | `G:\The.Invisible.Man...BeyondHD` | BDMV | ✅ | 7 | Pure disc | ⚠️ | Val | EL 7082 kbps — use EL stream file for RPU |
| 18 | Tron: Legacy | 2010 | Sci-Fi | `G:\Tron.Legacy...TMT` | BDMV | ✅ | ? | Pure disc | ✅ | Val | Profile TBD — verify |
| 19 | F1: The Movie | 2025 | Sport/Drama | `G:\F1.The.Movie...TMT` | BDMV | ✅ | ? | Pure disc | ✅ | Val | Profile TBD — verify |
| 20 | Weapons | 2025 | Thriller | `G:\Weapons.2025...mkv` | MKV | ✅ | **7** | Hybrid disc | ✅ | Val | Disc video + streaming RPU; DOVI confirmed |
| 21 | Ballerina | 2025 | Action | `G:\Ballerina...` | BDMV | ✅ | ? | Pure disc | ✅ | Val | Profile TBD — filler Val slot |
| 22 | **Dune: Part Two** * | 2024 | Sci-Fi | `G:\Dune.Part.Two...DOUHD` | BDMV | ✅ | 7 | Pure disc | ⚠️ | **Test** | Community benchmark — amber IMAX desert; EL stream |
| 23 | **Top Gun: Maverick** * | 2022 | Action | `G:\Top.Gun.Maverick...` | RAR | ✅ | ? | Pure disc | ⚠️ | **Test** | Community benchmark — IMAX aerial; extract RAR first |
| 24 | **Spider-Man: ATSV** * | 2023 | Animation | `G:\Spider.Man.Across...mkv` | MKV | ✅ | **7** | Pure disc | ⚠️ | **Test** | Community benchmark; DOVI confirmed; German audio |
| 25 | **Godzilla Minus One** * | 2023 | Sci-Fi | `G:\Godzilla.Minus.One...AREY` | BDMV | ✅ | 7 | Pure disc | ⚠️ | **Test** | Community benchmark — JPN disc DV; EL stream |
| 26 | **Civil War** * | 2024 | War/Action | `G:\Civil War 2024...mkv` | MKV | ✅ | **7** | Pure disc | ✅ | **Test** | A24 DV benchmark; DOVI confirmed; EL in v:0 |
| 27 | Predator Badlands | 2025 | Sci-Fi/Action | `G:\Predator.Badlands...MTeam` | BDMV | ✅ | 7 | Pure disc | ⚠️ | Test | EL 3957 kbps — use EL stream file for RPU |
| 28 | Gladiator II | 2024 | Action/Epic | `G:\Gladiator.II...SharpHD.iso` | ISO | 🔍 | ? | Pure disc | ⚠️ | Test | ISO — mount + ffprobe to confirm DV/profile |
| 29 | No Time to Die | 2021 | Action | `G:\No.Time.to.Die...ISO` | ISO | 🔍 | ? | Pure disc | ⚠️ | Test | ISO — mount + ffprobe to confirm |
| 30 | Wonder Woman | 2017 | Superhero | `G:\Wonder.Woman.2017.../...iso` | ISO | 🔍 | ? | Pure disc | ⚠️ | Test | ISO — mount + ffprobe to confirm |
| 31 | John Wick: Chapter 4 * | 2023 | Action | `G:\John.Wick.Kapitel.4...mkv` | MKV | ✅ | **7** | Pure disc | ⚠️ | **Test** (EN needed) | Community benchmark; DOVI confirmed; currently German — get EN disc |
| 32 | Troy (Director's Cut) | 2004 | Epic | `G:\Troy.2004...mkv` | MKV | ✅ | **7** | Pure disc | ⚠️ | — | DOVI confirmed; EL in v:0; German audio; reserve |
| 33 | Kingdom of Heaven (DC) | 2005 | Epic/Historical | `G:\Koenigreich.der.Himmel...mkv` | MKV | ✅ | **7** | Pure disc | ⚠️ | — | DOVI confirmed; EL in v:0; German audio; reserve |
| 34 | The Hurt Locker | 2008 | War/Drama | `G:\Toedliches.Kommando...mkv` | MKV | ✅ | **7** | Pure disc | ⚠️ | — | DOVI confirmed; EL in v:0; German audio; reserve |
| 35 | V for Vendetta | 2005 | Action/Sci-Fi | `G:\V.for.Vendetta...iso` | ISO | 🔍 | ? | Pure disc | ⚠️ | — | ISO — mount + ffprobe to confirm |
| 35 | MI: Dead Reckoning Pt 1 | 2023 | Action | `G:\Mission.Impossible.Dead.Reckoning...iso` | ISO | 🔍 | ? | Pure disc | ⚠️ | — | ISO — mount + ffprobe to confirm |
| 36 | Top Gun: Maverick | 2022 | Action | `G:\Top.Gun.Maverick...` | RAR | ✅ | ? | Pure disc | ⚠️ | — | In RAR archives — extract first |

#### On disk — no DV (skip for training)

| Title | Year | Location | HDR | Note |
|---|---|---|---|---|
| Oppenheimer | 2023 | `G:\Oppenheimer...ESiR` | HDR10 | EUR disc — HDR10 only; US disc has DV |
| Se7en | 1995 | `G:\Se7en...` | HDR10 | Disc confirmed HDR10 only |
| Last Breath | 2025 | `G:\Last.Breath...` | HDR10 | No DV track on disc |
| Heat | 1995 | `G:\Heat.1995...mkv` | HDR10+ | No DV |
| Nope | 2022 | `G:\Nope 2022...mkv` | HDR10 | No DV |
| Exodus: Gods and Kings | 2014 | `G:\Exodus Gods and Kings.m2ts` | HDR10 | Single m2ts, likely no DV |

#### Download needed

`*` = community benchmark title; must go to Test regardless of genre slot.

| # | Title | Year | Genre | DV | Priority | Split | Why needed |
|---|---|---|---|---|---|---|---|
| D1 | Knives Out | 2019 | Comedy/Mystery | ✅ | **Critical** | Train | Only Comedy candidate; Lionsgate UHD pure disc |
| D2 | A Quiet Place | 2018 | Horror | ✅ | **Critical** | Train | 28 Yrs Later moved to Train; need 2nd horror Train title |
| D3 | **The Batman** * | 2022 | Superhero/Noir | ✅ | High | **Test** | Community benchmark; perpetual rain noir — contrast to WW1984 |
| D4 | **Blade Runner 2049** * | 2017 | Sci-Fi/Noir | ✅ | High | **Test** | Most-discussed HDR/DV benchmark on AVForums; not on disk at all |
| D5 | **Joker** * | 2019 | Superhero/Drama | ✅ | High | **Test** | Community DV benchmark; grimy Gotham — colour science interest |
| D6 | 1917 | 2019 | Drama/War | ✅ | Medium | Train | Universal UHD; one-take WWI fills pure Drama Train slot |
| D7 | The Grand Budapest Hotel | 2014 | Comedy | ✅ | Medium | Val | Fox/Disney; pastel storybook — fills Comedy Val |
| D8 | All Quiet on the Western Front | 2022 | Drama/War | ✅ | Medium | Val | Netflix DV; naturalistic grey — fills Drama Val |
| D9 | **John Wick: Ch4** * (EN) | 2023 | Action | ✅ | Medium | **Test** | Community benchmark; have German disc — need EN version |
| D10 | Encanto | 2021 | Animation | ✅ | Low | Val | Disney+; fills Animation Val (Spider-Verse is German) |
| D11 | Interstellar | 2014 | Sci-Fi | ✅ | Low | Train | Paramount UHD; IMAX grain — Sci-Fi Train depth |
| D12 | Lord of the Rings: Fellowship | 2001 | Epic | ✅ | Low | Train | WB 4K Extended; fills Classic/Epic Train in English |

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
