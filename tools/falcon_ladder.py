"""Distance ladder: synthetic person / boxes / dust cloud ray-cast into real Falcon frames at fixed distances.
For each object and distance a fresh detector runs 30 frames (start-up) and then 50 frames with the object standing on
the track; the recording segment is chosen so that the real walking person is at least 25 m away from the object.
usage: python falcon_ladder.py <cache.bin> <truth.json> <out.json>"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard'))
import numpy as np

from falcon_cache import frames_cached
from falcon_eval import fill_gaps
from falcon_inject import inject
from tunnel_guard.core.detector import DetectorConfig, ObstacleDetector

CFG = os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json')
DISTANCES = [60, 90, 100, 120, 150, 175, 180, 190, 200, 215, 220, 230, 250, 260, 270, 280, 300]
OBJECTS = {
    'person': dict(kind='person', size=(0.5, 0.3, 1.75), rho=0.10),
    'box_0.5': dict(kind='box', size=(0.5, 0.5, 0.5), rho=0.20, z0=-1.80),       # standing on the rails
    'box_0.3': dict(kind='box', size=(0.3, 0.3, 0.3), rho=0.20, z0=-1.80),
    'dust': dict(kind='dust', size=(2.4, 3.0, 2.5), sigma=0.6, rho=0.05, z0=-1.9),
}
WARM, ON = 30, 50


def segment_for(d, truth):
    """First frame of an 80-frame stretch where the real person stays >= 25 m from distance d."""
    xs = np.array([truth.get(k, {}).get('x', np.nan) if truth.get(k, {}).get('n') else np.nan for k in range(4443)])
    for start in range(100, 4443 - WARM - ON, 20):
        seg = xs[start:start + WARM + ON]
        if np.all(np.isnan(seg) | (np.abs(seg - d) > 25)):
            return start
    return 100


def main(cache, truth_path, out):
    truth = {r['k']: r for r in fill_gaps(json.load(open(truth_path)))}
    rows = []
    t0 = time.time()
    for name, ob in OBJECTS.items():
        for d in DISTANCES:
            start = segment_for(d, truth)
            rng = np.random.default_rng(d)
            det = ObstacleDetector(DetectorConfig(scorer_model=CFG))
            hits, stops, first, pts, near_other = [], [], None, [], 0
            for i, (k, t, xyz, inten) in enumerate(frames_cached(cache, start, start + WARM + ON)):
                objs = [dict(ob, x=float(d))] if i >= WARM else []
                p, it, cnt = inject(xyz, inten, objs, rng)
                res = det.process(p, t, intensity=it)
                if i < WARM:
                    continue
                pts.append(cnt[0] if cnt else 0)
                near = [o for o in res.obstacles if abs(o['distance'] - d) < 4.0]
                hit = bool(near)
                hits.append(hit)
                stops.append(any(o['zone'] == 2 for o in near))
                if hit and first is None:
                    first = i - WARM
            rows.append(dict(object=name, distance=d, segment=start, detected=float(np.mean(hits)),
                             stop=float(np.mean(stops)), first_frame=first, returns_per_frame=float(np.mean(pts))))
            r = rows[-1]
            print(f"{name:8s} {d:4d} m: returns/frame {r['returns_per_frame']:5.1f}  detected {100 * r['detected']:5.1f}%  "
                  f"STOP {100 * r['stop']:5.1f}%  first after {r['first_frame']} frames   ({time.time() - t0:.0f}s)", flush=True)
            json.dump(rows, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:4])
