"""
libplacebo_baseline_eval.py
===========================
Runs libplacebo's pl_tone_map_spline over the val dataset and computes
MSE against the gold RPU polynomial curves.

Usage:
    python3 tools/libplacebo_baseline_eval.py
"""

import sys, os, subprocess, numpy as np, pandas as pd, time

# Ensure DLLs are findable (libplacebo + MSYS2 runtime)
os.environ['PATH'] = (
    r'C:\Code\libplacebo\build\src;'
    r'C:\msys64\ucrt64\bin;'
    + os.environ.get('PATH', '')
)
sys.path.insert(0, 'ml')
from dv_coef_model import load_data, row_to_target, target_to_rpu, eval_rpu

VAL_TITLES   = {'andor', 'euphoria', 'prehistoric', 'our', 'wondla'}
DISPLAY_NITS = [143, 1030, 1669]
N_PTS        = 256
TOOL_EXE     = 'c:/Code/libplacebo/build/tools/libplacebo_baseline_eval.exe'
MAX_SCENES   = 5000   # sample size for speed (set to None for all)

print("Loading datasets...", flush=True)
val_df   = load_data('F:/DTMModelData/val/val_dataset.csv')
train_df = load_data('F:/DTMModelData/train/train_dataset.csv')
all_df   = pd.concat([val_df, train_df], ignore_index=True)
df       = all_df[all_df['title'].isin(VAL_TITLES)].reset_index(drop=True)
print(f"Val scenes: {len(df):,}", flush=True)

if MAX_SCENES and len(df) > MAX_SCENES:
    df = df.sample(MAX_SCENES, random_state=42).reset_index(drop=True)
    print(f"Sampled: {len(df):,}", flush=True)

# Build input to C tool and gold curves simultaneously
# C tool input: scene_id, maxscl, l1_avg_pq, target_nits
# One row per (scene, tier) combination
input_rows  = ["scene_id,maxscl,l1_avg_pq,target_nits"]
gold_curves = {}   # (scene_idx, tier_idx) -> np.array [N_PTS]
scene_meta  = []   # (scene_idx, tier_idx, maxscl, nits) ordered same as input_rows

scene_idx = 0
skipped   = 0
for _, row in df.iterrows():
    t   = row_to_target(row)
    if t is None:
        skipped += 1
        continue
    rpu    = target_to_rpu(t)
    maxscl = float(row.get('maxscl', 0.5))
    l1_avg = float(row.get('l1_avg_pq', 0.3))
    # Evaluate gold curve on domain [0, maxscl] — same as C tool output
    xs     = np.linspace(0.0, maxscl, N_PTS)
    ys     = eval_rpu(rpu, xs)
    if np.any(np.isnan(ys)):
        skipped += 1
        continue
    for tier_idx, nits in enumerate(DISPLAY_NITS):
        row_id = scene_idx * len(DISPLAY_NITS) + tier_idx
        input_rows.append(f"{row_id},{maxscl:.6f},{l1_avg:.6f},{nits}")
        gold_curves[(scene_idx, tier_idx)] = ys
        scene_meta.append((scene_idx, tier_idx, maxscl, nits))
    scene_idx += 1

n_scenes = scene_idx
print(f"Valid: {n_scenes:,} scenes  Skipped: {skipped}  "
      f"Total curves: {len(scene_meta):,}", flush=True)

# Run C tool
print(f"\nRunning {TOOL_EXE} ...", flush=True)
t0   = time.time()
proc = subprocess.run(
    [TOOL_EXE],
    input='\n'.join(input_rows),
    capture_output=True, text=True
)
elapsed = time.time() - t0
if proc.returncode != 0:
    print(f"ERROR: tool exited {proc.returncode}")
    print(proc.stderr[:500])
    sys.exit(1)
print(f"  done ({elapsed:.1f}s, {len(scene_meta)/elapsed:.0f} curves/s)", flush=True)

# Parse output
out_lines = proc.stdout.strip().split('\n')
print(f"  Output lines: {len(out_lines)} (expected {len(scene_meta)+1})", flush=True)

# Compute MSE: libplacebo vs gold, identity vs gold
lp_mse   = {t: [] for t in range(3)}
id_mse   = {t: [] for t in range(3)}
mn_mse   = {t: [] for t in range(3)}

# Collect all gold curves to compute mean
all_gold = [[] for _ in range(3)]
for (si, ti, _, _) in scene_meta:
    all_gold[ti].append(gold_curves[(si, ti)])
mean_gold = [np.mean(g, axis=0) for g in all_gold]

for i, line in enumerate(out_lines[1:]):   # skip header
    if i >= len(scene_meta): break
    parts = line.split(',')
    if len(parts) < N_PTS + 1: continue
    si, ti, maxscl, nits = scene_meta[i]
    lp_curve = np.array([float(x) for x in parts[1:N_PTS+1]])
    gold     = gold_curves[(si, ti)]
    xs       = np.linspace(0.0, maxscl, N_PTS)

    lp_mse[ti].append(np.mean((lp_curve - gold)**2))
    id_mse[ti].append(np.mean((xs        - gold)**2))
    mn_mse[ti].append(np.mean((mean_gold[ti] - gold)**2))

# Report
print(f"\n{'='*60}")
print(f"libplacebo pl_tone_map_spline vs gold RPU polynomial")
print(f"Val set: {n_scenes:,} scenes × 3 tiers")
print(f"{'='*60}")
print(f"\n{'Method':<35}  {'MSE':>8}  {'RMS':>7}  {'~nits':>7}")
print("-"*58)

all_lp = sum(lp_mse.values(), [])
all_id = sum(id_mse.values(), [])
all_mn = sum(mn_mse.values(), [])

def row(label, vals):
    mse = np.mean(vals)
    rms = np.sqrt(mse)
    print(f"  {label:<33}  {mse:8.5f}  {rms:7.4f}  {rms*0.8*1500:7.0f}")

row("Identity (no tone map)",      all_id)
row("Mean gold curve (dumb)",       all_mn)
row("libplacebo pl_tone_map_spline", all_lp)
print(f"  {'Our model run8 ep10':<33}  0.01426  0.1194     143")

print(f"\nBy target display:")
for ti, nits in enumerate(DISPLAY_NITS):
    lp = np.mean(lp_mse[ti])
    id_ = np.mean(id_mse[ti])
    print(f"  {nits:4d} nits:  libplacebo={lp:.5f} (RMS={np.sqrt(lp):.4f})  "
          f"identity={id_:.5f}  "
          f"{'LP better' if lp < np.mean(all_lp) else 'LP worse'}")
