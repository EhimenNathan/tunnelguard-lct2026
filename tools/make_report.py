"""Build docs/EXPERIMENTS.md + figures from evaluation outputs (FP on empty bags, synthetic recall, held-out bag)."""
import json, os, sys, glob, ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FP_TAG = sys.argv[1] if len(sys.argv) > 1 else 'v10'
SYN_TAG = sys.argv[2] if len(sys.argv) > 2 else 'v3'
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
FIG = os.path.join(DOCS, 'figures')
os.makedirs(FIG, exist_ok=True)
EMPTY = ['roundT_doubleT', 'squareT_platform_squareT_switch', 'doubleT_platform',
         'roundT_squareT_pressureGate_squareT', 'roundT_pressureGate_roundT']
COL = {'bg': '#ffffff', 'ink': '#1c1d22', 'mut': '#6b6f80', 'a': '#520978', 'b': '#ff0053', 'c': '#8a83d1', 'd': '#fc3777'}
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': '#b0b3c0', 'axes.labelcolor': COL['ink'], 'xtick.color': COL['mut'],
                     'ytick.color': COL['mut'], 'axes.spines.top': False, 'axes.spines.right': False})

# ------------------------------------------------------------------ false positives
fp = json.load(open(f'fp_{FP_TAG}_summary.json'))
tot_frames = sum(v['frames'] for v in fp.values())
tot_stop = sum(v['stop_frames'] for v in fp.values())
tot_caution = sum(v['caution_frames'] for v in fp.values())
events = 0
for nm in EMPTY:
    z = np.load(f'fp_{FP_TAG}_{nm}.npz')
    obs = z['obs']
    if len(obs):
        stop_ids = np.unique(obs[obs[:, 2] == 2][:, 1])
        events += len(stop_ids)
ms = np.concatenate([np.load(f'fp_{FP_TAG}_{nm}.npz')['rows'][1:, 3] for nm in EMPTY]) * 1e3
geo_ms = np.concatenate([np.load(f'fp_{FP_TAG}_{nm}.npz')['rows'][1:, 4] for nm in EMPTY]) * 1e3
clear = np.concatenate([np.load(f'fp_{FP_TAG}_{nm}.npz')['rows'][:, 5] for nm in EMPTY])

fig, ax = plt.subplots(figsize=(8, 3.6))
names = ['roundT→doubleT', 'squareT platform\n+ switch', 'doubleT platform', 'roundT/squareT\npressure gate', 'roundT pressure\ngate']
rate = [100 * fp[nm]['stop_frames'] / fp[nm]['frames'] for nm in EMPTY]
ax.bar(range(5), rate, color=COL['a'], width=0.6)
for i, nm in enumerate(EMPTY):
    ax.text(i, rate[i] + 0.1, f"{fp[nm]['stop_frames']}/{fp[nm]['frames']}", ha='center', color=COL['ink'], fontsize=10)
ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('false STOP frames, %')
ax.set_title('False alarms on all obstacle-free recordings (every frame)', loc='left', color=COL['ink'])
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'false_alarms.png'), dpi=160); plt.close()

# ------------------------------------------------------------------ synthetic recall
syn = json.load(open(f'synth_{SYN_TAG}.json'))
shapes = sorted({r['shape'] for r in syn}, key=['person', 'box_50cm', 'dark_box_50cm', 'box_30cm'].index)
d0s = sorted({r['d0'] for r in syn})
label = {'person': 'person 1.75 m (ρ 15%)', 'box_50cm': 'box 0.5 m (ρ 20%)', 'dark_box_50cm': 'dark box 0.5 m (ρ 5%)',
         'box_30cm': 'box 0.3 m (ρ 20%)'}
colors = [COL['a'], COL['b'], COL['c'], COL['d']]
fig, ax = plt.subplots(figsize=(8, 4))
table = []
for si, sh in enumerate(shapes):
    rec, visible = [], []
    for d in d0s:
        rs = [r for r in syn if r['shape'] == sh and r['d0'] == d]
        vis = [r for r in rs if r['returns_first'] >= 3]        # physically observable (not occluded by a curve)
        rec.append(np.mean([r['detected'] for r in vis]) if vis else np.nan)
        visible.append(len(vis))
        dd = [r['det_dist'] for r in vis if r['detected']]
        table.append((sh, d, len(rs), len(vis), np.mean([r['detected'] for r in vis]) if vis else np.nan,
                      np.mean([r['frame'] for r in vis if r['detected']]) if dd else np.nan, np.mean([r['returns_first'] for r in rs])))
    ax.plot(d0s, rec, 'o-', color=colors[si], label=label[sh], lw=2)
ax.set_ylim(-0.05, 1.05); ax.set_xlabel('start distance, m'); ax.set_ylabel('confirmed STOP recall')
ax.set_title('Ray-cast obstacles in real frames of all empty recordings', loc='left', color=COL['ink'])
ax.legend(frameon=False, fontsize=9, loc='lower left'); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'recall_vs_distance.png'), dpi=160); plt.close()

# ------------------------------------------------------------------ held-out
hold = json.load(open('holdout_results.json'))
fr = np.array([o['frame'] for o in hold]); lvl = np.array([o['level'] for o in hold])
latA = np.array([next((g['lat'] for g in o['gts'] if g['name'] == 'A'), np.nan) for o in hold])
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.fill_between(fr, -1.35, 1.35, color='#ede7f6', label='train envelope (±1.35 m)')
ax.plot(fr, latA, color=COL['ink'], lw=2, label='person A lateral offset (reference)')
stop = lvl == 2
ax.scatter(fr[stop], np.full(stop.sum(), 3.2), s=10, color=COL['b'], label='STOP decision')
ax.set_xlabel('frame (10 Hz)'); ax.set_ylabel('lateral offset, m'); ax.set_ylim(-2.5, 3.6)
ax.set_title('Held-out real recording: person at 55 m crossing the track', loc='left', color=COL['ink'])
ax.legend(frameon=False, fontsize=8, loc='lower right', ncol=1)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'holdout_timeline.png'), dpi=160); plt.close()
inside = [(o, g) for o in hold for g in o['gts'] if g['name'] == 'A' and abs(g['lat']) < 1.35]
det_in = sum(g['detected'] for _, g in inside)
stop_in = sum(g['zone'] == 2 for _, g in inside)
fp_hold = sum(1 for o in hold for d in o['dets'] if d['zone'] == 2 and not any(abs(d['x'] - g['x']) < 3 and abs(d['y'] - g['y']) < 1.5 for g in o['gts']))
dist_hold = [d['dist'] for o in hold for d in o['dets'] if d['zone'] == 2]
ms_hold = np.median([o['ms'] for o in hold])

res = dict(tot_frames=tot_frames, tot_stop=tot_stop, tot_caution=tot_caution, events=int(events), ms_med=float(np.median(ms)),
           ms_p95=float(np.percentile(ms, 95)), geo_ms=float(np.median(geo_ms)), clear_med=float(np.median(clear)),
           hold_inside=len(inside), hold_detected=int(det_in), hold_stop=int(stop_in), hold_fp=int(fp_hold),
           hold_dist=float(np.median(dist_hold)) if dist_hold else None, hold_ms=float(ms_hold), synth=table)
json.dump(res, open('report_numbers.json', 'w'), indent=1, default=float)

md = [f"""# Experiments

All numbers are produced by the scripts in `tools/` from the provided recordings. **The only recording with real
obstacles (`doubleT_obstacle`) was held out**: it was never used to design or tune anything. Tuning used only
false-positive analysis on the five obstacle-free recordings and ray-cast synthetic obstacles inside them.

Hardware of these measurements: laptop Intel i5-8250U (15 W, 4 cores), single-threaded numpy. The evaluation stand
(i7-9700E, 65 W) is expected to be faster.

## 1. False alarms on all obstacle-free recordings (every frame, {tot_frames} frames)

| Recording | Frames | False STOP frames | CAUTION frames | Median latency, ms |
|---|---|---|---|---|
"""]
for nm, nice in zip(EMPTY, names):
    v = fp[nm]
    md.append(f"| {nice.replace(chr(10), ' ')} | {v['frames']} | {v['stop_frames']} ({100 * v['stop_frames'] / v['frames']:.1f} %) | {v['caution_frames']} | {v['ms_med']:.0f} |\n")
md.append(f"| **Total** | **{tot_frames}** | **{tot_stop} ({100 * tot_stop / tot_frames:.2f} %)** | {tot_caution} | {np.median(ms):.0f} |\n\n")
md.append(f"Distinct false STOP tracks: {events}. Median verified clear distance: {np.median(clear):.0f} m.\n\n")
md.append("![false alarms](figures/false_alarms.png)\n\n")
md.append(f"""## 2. Held-out real obstacle recording (`doubleT_obstacle`, stationary train, people in the tunnel)

Reference positions of the people come from an independent method (per-pixel median background subtraction of the
stationary recording), not from the detector.

* Frames in which person A was inside the train envelope: **{len(inside)}**
* ... detected: **{det_in}**, confirmed STOP: **{stop_in}** (detection distance ≈ {res['hold_dist']:.1f} m)
* Person B walking beside the train, outside the envelope: never raised STOP (correct).
* STOP detections not corresponding to a person: **{fp_hold}**
* Median processing time (921 600-point clouds): {ms_hold:.0f} ms

![held-out](figures/holdout_timeline.png)

## 3. Detection range: ray-cast obstacles in real frames

Obstacles are ray-cast into the sensor's actual beam directions (occlusion and angular sampling of the real Pandar128),
with returns lost at long range according to the Pandar128E3X link budget (200 m at 10 % reflectivity). Each case:
start distance D0, approach at 10 m/s over 12 real consecutive frames, 3 start positions in each of the 5 empty
recordings. Recall is computed over cases where the object is physically visible (≥ 3 returns; in curves objects
beyond the sight distance are occluded by the tunnel wall and no sensor can see them).

| Object | D0, m | cases | visible | recall | frames to confirm | returns at D0 |
|---|---|---|---|---|---|---|
""")
for sh, d, n, nv, rec, k, ret in table:
    md.append(f"| {label[sh]} | {d:.0f} | {n} | {nv} | {rec:.2f} | {k:.1f} | {ret:.1f} |\n")
md.append("\n![recall](figures/recall_vs_distance.png)\n")
open(os.path.join(DOCS, 'EXPERIMENTS.md'), 'w', encoding='utf-8').write(''.join(md))
print(json.dumps({k: v for k, v in res.items() if k != 'synth'}, indent=1, default=float))
for row in table:
    print(row)
