"""False-positive evaluation on the obstacle-free bags (every frame)."""
import os
import sys, time, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'tunnel_guard'))
import numpy as np
from loader import BagCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig

EMPTY = ['roundT_doubleT', 'squareT_platform_squareT_switch', 'doubleT_platform',
         'roundT_squareT_pressureGate_squareT', 'roundT_pressureGate_roundT']
names = sys.argv[1].split(',') if len(sys.argv) > 1 else EMPTY
tag = sys.argv[2] if len(sys.argv) > 2 else 'v1'
summary = {}
for nm in names:
    b = BagCache(nm)
    det = ObstacleDetector(DetectorConfig(scorer_model=__import__('os').environ.get('TG_SCORER', ''), ))
    rows, cand_log, ob_log = [], [], []
    for f in range(b.n):
        rng, inten = b.frame(f)
        V = rng > 0.5
        xyz_fwd = b.dirs[V] * rng[V][:, None]
        # back to the sensor frame convention (forward = -y) to exercise axis detection
        xyz = np.stack([-xyz_fwd[:, 1], -xyz_fwd[:, 0], xyz_fwd[:, 2]], 1)
        ring = np.broadcast_to(np.arange(128)[None, :], V.shape)[V]
        res = det.process(xyz, b.t[f], intensity=inten[V], ring=ring)
        g = res.geometry
        rows.append([f, res.level, len(res.obstacles), res.timings['total'], res.timings['geometry'], res.clear_distance,
                     g.rail_range, g.valid_range])
        for c in res.candidates:
            cand_log.append([f, c['s_min'], c['l_mean'], c['h_min'], c['h_max'], c['n'], c['in_gauge'], c['rings']])
        for o in res.obstacles:
            ob_log.append([f, o['id'], o['zone'], o['distance'], o['lateral'], o['height'], o['n'], *o['size']])
    rows = np.array(rows); cand_log = np.array(cand_log).reshape(-1, 8); ob_log = np.array(ob_log).reshape(-1, 10)
    np.savez(f'fp_{tag}_{nm}.npz', rows=rows, cand=cand_log, obs=ob_log)
    s = dict(frames=int(b.n), stop_frames=int((rows[:, 1] == 2).sum()), caution_frames=int((rows[:, 1] == 1).sum()),
             ms_med=float(np.median(rows[1:, 3]) * 1e3), ms_p95=float(np.percentile(rows[1:, 3], 95) * 1e3),
             geo_ms=float(np.median(rows[1:, 4]) * 1e3), clear_med=float(np.median(rows[:, 5])),
             n_obstacle_ids=int(len(np.unique(ob_log[:, 1]))) if len(ob_log) else 0)
    summary[nm] = s
    print(nm, s, flush=True)
json.dump(summary, open(f'fp_{tag}_summary.json', 'w'), indent=1)
