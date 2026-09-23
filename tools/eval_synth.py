"""Synthetic-obstacle recall evaluation on the obstacle-free bags (the real obstacle bag stays held out).

For several start frames in every empty bag, a warmed-up detector is copied and fed a short real frame sequence
into which an obstacle is ray-cast on the track at a start distance D0, approaching at a nominal 10 m/s.
Reports, per shape and D0, whether and at which distance the obstacle became a confirmed in-gauge detection.
"""
import os
import sys, time, copy, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'tunnel_guard'))
import numpy as np
from loader import BagCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.geometry import GeometryEstimator
from tunnel_guard.core.synth import SHAPES, inject
from tunnel_guard.core.gauge import IN_GAUGE

EMPTY = ['roundT_doubleT', 'squareT_platform_squareT_switch', 'doubleT_platform',
         'roundT_squareT_pressureGate_squareT', 'roundT_pressureGate_roundT']
args = dict(a.split('=') for a in sys.argv[1:])
shapes = args.get('shapes', 'person').split(',')
D0S = [float(v) for v in args.get('d0', '40,80,120,160,200').split(',')]
N_START = int(args.get('starts', '3'))
SEQ = int(args.get('seq', '10'))
LAT = float(args.get('lat', '0.0'))
tag = args.get('tag', 'v1')
names = args.get('bags', ','.join(EMPTY)).split(',')
SPEED = 10.0
rng = np.random.default_rng(0)
results = []
t0 = time.time()
for nm in names:
    b = BagCache(nm)
    starts = np.linspace(30, b.n - SEQ - 1, N_START).astype(int)
    for f0 in starts:
        # reference geometry for placement: clean pass with its own estimator
        geo_ref = GeometryEstimator()
        refs = {}
        for f in range(f0 - 12, f0 + SEQ):
            p, _, _ = b.points(f)
            refs[f] = geo_ref.estimate(p[p[:, 0] > 1.0])
        det = ObstacleDetector(DetectorConfig(forward_axis='x', scorer_model=__import__('os').environ.get('TG_SCORER', '')))
        for f in range(f0 - 12, f0):
            p, I, _ = b.points(f)
            det.process(p, b.t[f])
        for shape_name in shapes:
            shape = SHAPES[shape_name]
            for d0 in D0S:
                d = copy.deepcopy(det)
                first = None
                n_ret = []
                fp_other = 0
                for k in range(SEQ):
                    f = f0 + k
                    dist = d0 - SPEED * 0.1 * k
                    g = refs[f]
                    centre = np.array([dist, float(g.centre(dist)) + LAT, float(g.rail_z(dist) + g.roll * LAT)])
                    rimg, inten = b.frame(f)
                    rimg2, nr = inject(rimg, b.dirs, centre, shape, rng)
                    n_ret.append(nr)
                    V = rimg2 > 0.5
                    pts = b.dirs[V] * rimg2[V][:, None]
                    res = d.process(pts, b.t[f], intensity=inten[V])
                    hit = [o for o in res.obstacles if o['zone'] == IN_GAUGE and abs(o['distance'] - dist) < 3 + 0.03 * dist]
                    other = [o for o in res.obstacles if o not in hit]
                    fp_other += len(other)
                    if hit and first is None:
                        first = (k, dist)
                results.append(dict(bag=nm, f0=int(f0), shape=shape_name, d0=d0, detected=first is not None,
                                    frame=None if first is None else first[0], det_dist=None if first is None else first[1],
                                    returns_first=n_ret[0], returns_mean=float(np.mean(n_ret)), fp_other=fp_other))
                r = results[-1]
                print(f'{nm} f{f0} {shape_name} d0={d0:.0f}: det={r["detected"]} at k={r["frame"]} dist={r["det_dist"]} '
                      f'returns {n_ret[0]}..{n_ret[-1]} other_obst={fp_other}  [{time.time() - t0:.0f}s]', flush=True)
json.dump(results, open(f'synth_{tag}.json', 'w'), indent=1)
for shape_name in shapes:
    for d0 in D0S:
        rs = [r for r in results if r['shape'] == shape_name and r['d0'] == d0]
        print(f'{shape_name:16s} D0={d0:5.0f}: recall {np.mean([r["detected"] for r in rs]):.2f}  '
              f'mean returns {np.mean([r["returns_first"] for r in rs]):.1f}  frames-to-confirm '
              f'{np.mean([r["frame"] for r in rs if r["detected"]]) if any(r["detected"] for r in rs) else float("nan"):.1f}')
