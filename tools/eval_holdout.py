"""Held-out evaluation on the real obstacle recording (doubleT_obstacle). Never used for tuning.

Reference positions of the two people come from an independent method (per-pixel median background subtraction,
computed earlier on this stationary recording, obst_change_rows.npy), not from the detector.
Person A: ~55 m ahead, moving across the tracks.  Person B: appears next to the train at t~14 s and walks away.
"""
import os
import sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'tunnel_guard'))
import numpy as np
from loader import BagCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig

b = BagCache('doubleT_obstacle')
rows = np.load('obst_change_rows.npy')          # frame, X, Y, Z (forward frame)
det = ObstacleDetector(DetectorConfig(scorer_model=__import__('os').environ.get('TG_SCORER', ''), ))         # auto axis, sensor-frame input as from ROS
out = []
for f in range(b.n):
    rng, inten = b.frame(f)
    V = rng > 0.5
    fwd = b.dirs[V] * rng[V][:, None]
    xyz = np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1)
    res = det.process(xyz, b.t[f], intensity=inten[V])
    g = res.geometry
    r = rows[rows[:, 0] == f]
    gts = []
    for name, m in (('A', (r[:, 1] > 48) & (r[:, 1] < 62) & (r[:, 3] > -2.9) & (r[:, 3] < -0.5)),
                    ('B', (r[:, 1] > 1.5) & (r[:, 1] < 25) & (r[:, 2] > -2.6) & (r[:, 2] < -1.2) & (f >= 140))):
        if m.sum() >= 10:
            c = np.median(r[m, 1:4], axis=0)
            c[1] = -c[1]            # loader frame is mirrored (Y = -x_lidar); detector frame is right-handed (Y = +x_lidar)
            s_, l_, h_ = g.to_track(c[None, :]) if g.ok else (np.array([c[0]]), np.array([np.nan]), np.array([np.nan]))
            gts.append(dict(name=name, x=float(c[0]), y=float(c[1]), lat=float(l_[0])))
    dets = [dict(id=o['id'], zone=o['zone'], dist=round(o['distance'], 1), lat=round(o['lateral'], 2),
                 x=float(o['centroid_fwd'][0]), y=float(o['centroid_fwd'][1]), h=round(o['height'], 2), n=o['n'])
            for o in res.obstacles]
    for gt in gts:
        match = [d for d in dets if abs(d['x'] - gt['x']) < 3.0 and abs(d['y'] - gt['y']) < 1.5]
        gt['detected'] = bool(match)
        gt['zone'] = max((d['zone'] for d in match), default=0)
    out.append(dict(frame=f, t=float(b.t[f]), level=res.level, nearest=None if np.isnan(res.nearest_distance) else round(res.nearest_distance, 1),
                    clear=round(res.clear_distance, 1), gts=gts, dets=dets, ms=round(1e3 * res.timings['total'], 1)))
    if f % 20 == 0:
        print(f, res.level, gts, [(d['dist'], d['lat'], d['zone']) for d in dets], flush=True)
json.dump(out, open('holdout_results.json', 'w'), indent=1)
lv = np.bincount([o['level'] for o in out], minlength=3)
print('levels CLEAR/CAUTION/STOP', lv)
for name in 'AB':
    fr = [o for o in out for g in o['gts'] if g['name'] == name]
    gt_in = [g for o in out for g in o['gts'] if g['name'] == name]
    inside = [g for g in gt_in if abs(g['lat']) < 1.35]
    print(f'person {name}: frames present {len(gt_in)}, inside nominal gauge {len(inside)}, '
          f'detected (any zone) {sum(g["detected"] for g in gt_in)}, detected while inside gauge {sum(g["detected"] for g in inside)}, '
          f'IN_GAUGE while inside {sum(g["zone"] == 2 for g in inside)}')
unmatched = sum(1 for o in out for d in o['dets'] if d['zone'] == 2 and not any(abs(d['x'] - g['x']) < 3 and abs(d['y'] - g['y']) < 1.5 for g in o['gts']))
print('IN_GAUGE detections not matching a person:', unmatched)
print('ms median', np.median([o['ms'] for o in out]))
