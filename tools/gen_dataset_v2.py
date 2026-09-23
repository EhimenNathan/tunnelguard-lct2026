"""Candidate dataset v2 for the learned scorer: warm geometry and long-range coverage (no leakage: the real obstacle
recording is never used).

Differences to v1 (gen_dataset.py), found with tools/diag_range.py:
  * v1 started every ray-cast sequence from a detector warmed up for 12 frames, so the fused far geometry (measured to
    ~180 m in normal operation) was only ~115 m deep and far objects rarely became candidates.  v2 snapshots the
    continuously running pass-A detector at the start of every sequence.
  * v1 drew object distances uniformly in 20-195 m.  v2 draws half of the objects in 120-245 m.
  * v2 runs with two_tier=False so candidates between the measured range and the line of sight keep their in-envelope
    evidence; the decision policy decides what to do with them.
usage (scratchpad):  python gen_dataset_v2.py bag1,bag2 tag
"""
import copy
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))
sys.path.insert(0, os.getcwd())
import numpy as np

from loader import BagCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.features import FEATURE_NAMES
from tunnel_guard.core.geometry import GeometryEstimator, warmup
from tunnel_guard.core.synth import Shape, inject

ALL = ['roundT_doubleT', 'squareT_platform_squareT_switch', 'doubleT_platform', 'roundT_squareT_pressureGate_squareT',
       'roundT_pressureGate_roundT']
bags = sys.argv[1].split(',')
tag = sys.argv[2]
SEQ, EVERY, START = 10, 25, 40
rng = np.random.default_rng(abs(hash(tag)) % 2**32)
warmup()


def random_shape():
    if rng.random() < 0.45:
        return Shape('cylinder', (float(rng.uniform(0.15, 0.32)), float(rng.uniform(0.8, 1.95))), float(rng.uniform(0.05, 0.45)),
                     'person-like')
    return Shape('box', (float(rng.uniform(0.2, 1.4)), float(rng.uniform(0.2, 1.2)), float(rng.uniform(0.2, 1.6))),
                 float(rng.uniform(0.04, 0.5)), 'box')


def cfg():
    return DetectorConfig(forward_axis='x', two_tier=False)


rows_X, rows_meta, objs = [], [], []


def record(res, det, bag_i, pas, seq, f, objects):
    g = res.geometry
    measured = max(g.valid_range, g.rail_range) + det.cfg.horizon_slack if g.ok else 0.0
    for c, x in zip(res.all_candidates, res.features):
        label, obj_id = 0, -1
        for o in objects:
            if abs(c['s_min'] - o['dist']) < 1.5 + 0.02 * o['dist'] and abs(c['l_mean'] - o['lat']) < 0.9:
                label, obj_id = 1, o['id']
        tr = c['ml_view'].get('track')
        rows_X.append(x)
        rows_meta.append((bag_i, pas, seq, f, tr.id if tr is not None else -1, label, obj_id,
                          int(c['contained'] and not c['shell']),
                          int(c['in_gauge_raw'] >= 2 and not (c['gravity_fail'] or c['shape_fail'])),
                          int(c['in_gauge_raw']), int(c['s_min'] > measured)))


t0 = time.time()
for nm in bags:
    bag_i = ALL.index(nm)
    b = BagCache(nm)
    # pass A1: clean-frame geometry of every frame (placement reference for the ray-cast objects)
    ref = GeometryEstimator()
    geos = {}
    for f in range(b.n):
        p, _, _ = b.points(f)
        g = ref.estimate(p[p[:, 0] > 1.0])
        geos[f] = (g.yc.copy(), g.zr.copy(), g.roll, g.ok)
    # pass A2: detector on every clean frame (negatives); at each sequence start a snapshot runs pass B immediately
    det = ObstacleDetector(cfg())
    starts = set(range(START, b.n - SEQ, EVERY))
    seq_id = bag_i * 1000
    for f in range(b.n):
        if f in starts:
            seq_id += 1
            d = copy.deepcopy(det)
            objects, used = [], []
            for kk in range(2):
                for _ in range(30):
                    d0 = float(rng.uniform(120, 245)) if rng.random() < 0.5 else float(rng.uniform(20, 245))
                    if all(abs(d0 - u) > 15 for u in used):
                        break
                used.append(d0)
                objects.append(dict(id=seq_id * 10 + kk, d0=d0, speed=float(rng.choice([0.0, rng.uniform(3, 15)])),
                                    lat=float(rng.uniform(-1.2, 1.2)), shape=random_shape()))
            for k in range(SEQ):
                fk = f + k
                yc, zr, roll, ok = geos[fk]
                rimg, inten = b.frame(fk)
                now = []
                for o in objects:
                    dist = o['d0'] - o['speed'] * 0.1 * k
                    xi = int(np.clip(round(dist), 0, len(yc) - 1))
                    centre = np.array([dist, yc[xi] + o['lat'], zr[xi] + roll * o['lat']])
                    rimg, nret = inject(rimg, b.dirs, centre, o['shape'], rng)
                    now.append(dict(id=o['id'], dist=dist, lat=o['lat']))
                    objs.append((bag_i, seq_id, fk, k, o['id'], dist, o['lat'], nret, o['shape'].kind, *o['shape'].size,
                                 o['shape'].reflectivity))
                V = rimg > 0.5
                res = d.process(b.dirs[V] * rimg[V][:, None], b.t[fk], intensity=inten[V])
                record(res, d, bag_i, 1, seq_id, fk, now)
            del d
        p, I, _ = b.points(f)
        res = det.process(p, b.t[f], intensity=I)
        record(res, det, bag_i, 0, -1, f, [])
    print(f'{nm} done {time.time() - t0:.0f}s rows {len(rows_X)}', flush=True)
    np.savez_compressed(f'ds2_{tag}.npz', X=np.asarray(rows_X, np.float32), meta=np.asarray(rows_meta, np.int32),
                        names=np.array(FEATURE_NAMES), objs=np.array(objs, dtype=object))
print('done', time.time() - t0)
