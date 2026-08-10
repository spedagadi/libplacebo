# DV curve experiment — handoff

Foundation code for the ML dynamic tone-mapping litmus test. Working, reproducible.

## Run
    pip install pandas numpy scikit-learn
    python3 dv_curve_model.py dv_dataset_full.csv

## What it does
1. **eval_curve** — reconstructs the per-scene tone curve from the DV RPU piecewise
   polynomial. This is the *corrected* version (10-bit input domain, per-segment order,
   single-segment handling) that lifted R² from 0.08 → 0.35. Produces 100% monotonic curves.
2. **prepare** — cleans, dedups frames→scenes (scene_refresh cumsum), reconstructs curves.
3. **litmus** — scene-grouped 5-fold (GroupKFold, no leakage) GBR regressor: features→curve,
   vs mean-curve baseline. Reports improvement% and R².
4. **feature_importance** — confirms the signal is driven by sensible features (peak first).

## Verified result (single title, 2062 frames / 576 scenes)
    improvement over baseline = 21.2%   R² = 0.354   monotonic = 100%
    top feature = maxscl (peak), as expected

## The known caveats (don't skip)
- **Label not externally validated.** eval_curve is monotonic + plausible but NOT yet
  cross-checked against libplacebo/dovi_tool's actual DV curve. Some of the unexplained 65%
  variance may be residual label error, not missing features. → validate_against_reference().
- **Coefficient scaling inferred** (input as x/1023). Adjust in eval_curve if your extractor
  used a different convention (this is the first thing to check in validation).
- **Single title** — within-title signal only; generalization needs many titles.
- **Global features only** — no spatial. The likely lever to beat 0.35 is integral-image
  highlight-concentration features, which need DECODED FRAMES.

## Copilot extension points (stubs in the code, marked `# COPILOT:`)
- validate_against_reference() — **do this first.** Overlay reconstructed curve vs
  libplacebo/dovi_tool actual output. Gates trusting R² and scaling to many titles.
- visualize_curves() — plot/overlay curves across scenes (how much do they vary?).
- libplacebo_computepeak_baseline() — **the real test:** is the ML curve closer to the DV
  gold standard than libplacebo's metadata-free --hdr-compute-peak curve? Quantifies value.
- Swap GBR → XGBoost/LightGBM/torch MLP in litmus. Keep GroupKFold(scene_id) — never a
  random frame split (leakage).
- Add spatial features (needs decoded frames) and re-run litmus; check if R² rises.

## Data contract (columns used)
- Features: maxscl, average_maxrgb, fraction_bright_pixels, distrib_val_3..8
- Label source: poly_pivots, poly_num_segs, seg{0..7}_order, seg{0..7}_c0/c1/c2
- Scene seg: scene_refresh (cumsum → scene_id)
- Cross-check: pixel-computed peak should ~match maxscl (parity check for future extractor)