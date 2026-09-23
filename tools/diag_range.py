"""Detection funnel at long range: for ray-cast objects approaching from far away, record per frame the first pipeline
stage that loses the object.  usage (scratchpad):  python diag_range.py [out.json]"""
import copy
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))
sys.path.insert(0, os.getcwd())
import numpy as np

from loader import BagCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.geometry import GeometryEstimator, warmup
from tunnel_guard.core.gauge import IN_GAUGE
from tunnel_guard.core.synth import SHAPES, inject

MODEL = os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json')
WINDOWS = [('squareT_platform_squareT_switch', 289), ('squareT_platform_squareT_switch', 300),
           ('roundT_squareT_pressureGate_squareT', 283), ('roundT_doubleT', 185)]
D0S = [140.0, 160.0, 180.0, 200.0, 220.0]
SHAPE_NAMES = ['person', 'box_50cm', 'trolley']
SEQ = 12
WARMUP = int(os.environ.get('DIAG_WARMUP', 150))
out_path = sys.argv[1] if len(sys.argv) > 1 else 'diag_range.json'
warmup()
rs = np.random.default_rng(1)
rows = []
for nm, f0 in WINDOWS:
    b = BagCache(nm)
    ref = GeometryEstimator()
    refs = {}
    for f in range(max(0, f0 - WARMUP), f0 + SEQ):      # warm clean-frame reference geometry for placement
        p, _, _ = b.points(f)
        g_ = ref.estimate(p[p[:, 0] > 1.0])
        if f >= f0:
            refs[f] = g_
    base = ObstacleDetector(DetectorConfig(forward_axis='x', scorer_model=MODEL))
    for f in range(max(0, f0 - WARMUP), f0):     # long warm-up: far geometry is built up by temporal fusion
        p, _, _ = b.points(f)
        base.process(p, b.t[f])
    for shape_name in SHAPE_NAMES:
        for d0 in D0S:
            det = copy.deepcopy(base)
            for k in range(SEQ):
                f = f0 + k
                dist = d0 - 1.0 * k
                gr = refs[f]
                centre = np.array([dist, float(gr.centre(dist)), float(gr.rail_z(dist))])
                rimg, inten = b.frame(f)
                rimg2, nret = inject(rimg, b.dirs, centre, SHAPES[shape_name], rs)
                V = rimg2 > 0.5
                res = det.process(b.dirs[V] * rimg2[V][:, None], b.t[f], intensity=inten[V])
                g = res.geometry
                cfg = det.cfg
                sight = res.timings['sight']
                horizon = min(cfg.max_range, max(g.valid_range, g.rail_range, g.trusted_range) + cfg.horizon_slack, sight) if g.ok else 0
                measured = max(g.valid_range, g.rail_range) + cfg.horizon_slack
                near = [c for c in (res.all_candidates or []) if abs(c['s_min'] - dist) < 3 + 0.03 * dist]
                P = res.forward_points
                obj = (np.abs(P[:, 0] - dist) < 1.0) & (np.abs(P[:, 1] - centre[1]) < 0.8)
                in_zone = int(np.sum(res.zone[obj] == IN_GAUGE)) if res.zone is not None else 0
                any_zone = int(np.sum(res.zone[obj] > 0)) if res.zone is not None else 0
                confirmed = any(o['zone'] == IN_GAUGE and abs(o['distance'] - dist) < 3 + 0.03 * dist for o in res.obstacles)
                if confirmed:
                    stage = 'detected (STOP)'
                elif nret < 1:
                    stage = 'no lidar return'
                elif dist > sight:
                    stage = 'beyond line of sight'
                elif dist > horizon:
                    stage = 'beyond geometry horizon'
                elif any_zone == 0:
                    stage = 'outside envelope / h_min'
                elif dist > measured and cfg.two_tier:
                    stage = 'beyond measured range (CAUTION)'
                elif not near:
                    stage = 'cluster too small'
                else:
                    c = max(near, key=lambda c: c['n'])
                    if c['in_gauge_raw'] < cfg.min_in_gauge_points:
                        stage = 'too few in-gauge points'
                    elif c.get('shell'):
                        stage = 'shell test'
                    elif not c.get('contained', True):
                        stage = 'containment test'
                    elif 'score' in c and c['score'] < det.scorer.threshold:
                        stage = 'ML score below threshold'
                    elif c.get('gravity_fail') or c.get('shape_fail'):
                        stage = 'gravity/shape test'
                    else:
                        stage = 'awaiting M-of-N confirmation'
                rows.append(dict(bag=nm, f0=f0, shape=shape_name, d0=d0, k=k, dist=dist, returns=int(nret), obj_pts=int(obj.sum()),
                                 in_zone=in_zone, sight=float(sight), horizon=float(horizon), measured=float(measured),
                                 valid=float(g.valid_range), trusted=float(g.trusted_range), stage=stage,
                                 score=float(max((c.get('score', 0) for c in near), default=0)),
                                 cand_n=int(max((c['n'] for c in near), default=0))))
            last = [r for r in rows if r['bag'] == nm and r['f0'] == f0 and r['shape'] == shape_name and r['d0'] == d0]
            print(nm[:12], f0, shape_name, d0, Counter(r['stage'] for r in last).most_common(3),
                  'ret', [r['returns'] for r in last][:4], 'meas', int(last[0]['measured']), 'sight', int(last[0]['sight']), flush=True)
json.dump(rows, open(out_path, 'w'))
for band in [(125, 150), (150, 175), (175, 200), (200, 225)]:
    sel = [r for r in rows if band[0] <= r['dist'] < band[1]]
    print(band, len(sel), Counter(r['stage'] for r in sel).most_common())
