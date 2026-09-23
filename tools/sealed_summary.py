"""Summarise the sealed final test of the new line (last 5 min of the 20-min drive, t >= 900 s, 2.8 km; never used for
training or tuning) over far-field subsampling seeds -> sealed_summary.json.  Run from the scratchpad directory after
run_drive.py runs  seal_<name>_s<seed>  (pieces 169-220).   usage: python sealed_summary.py"""
import json

import numpy as np

KM = 2.81            # lidar odometry, t >= 900 s
RUNS = [('v1: ⅓ LGBM + ⅓ CatBoost\n+ ⅓ PI-MLP', 'seal_v1_s{}'), ('½ LGBM + ½ CatBoost', 'seal_efalse_s{}'),
        ('½ LGBM + ½ CatBoost\n+ ЭГО-тест (финал)', 'seal_ego2_s{}')]


def sealed(d):
    frames = stops = 0
    ids = set()
    ms = []
    for line in open(f'{d}/frames.jsonl'):
        f = json.loads(line)
        if f['t'] < 900:
            continue
        frames += 1
        stops += f['level'] == 2
        ids |= {o['id'] for o in f['obs'] if o['zone'] == 2}
        ms.append(f['ms'])
    return frames, stops, len(ids), float(np.median(ms))


rows = []
for label, pat in RUNS:
    r = [sealed(pat.format(s)) for s in range(3)]
    ev = [x[2] for x in r]
    fr = [x[1] for x in r]
    rows.append(dict(label=label, events=ev, stop_frames=fr, frames=r[0][0], events_mean=float(np.mean(ev)),
                     events_min=min(ev), events_max=max(ev), per_km=float(np.mean(ev)) / KM,
                     stop_frame_pct=100 * float(np.mean(fr)) / r[0][0], ms_median=float(np.median([x[3] for x in r]))))
    print(f"{label:22s} events {ev} mean {np.mean(ev):.1f} ({np.mean(ev) / KM:.1f}/km)  STOP frames {fr} "
          f"({rows[-1]['stop_frame_pct']:.2f} %)  median {rows[-1]['ms_median']:.0f} ms")
json.dump(dict(km=KM, seeds=3, rows=rows), open('sealed_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
