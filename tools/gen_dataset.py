"""Candidate dataset for the learned second-stage scorer (no leakage: the real obstacle recording is not used).

Per recording:
  pass A  every frame, clean               -> all candidates are negatives (label 0); per-frame track geometry saved
  pass B  sequences with 2 randomised ray-cast obstacles placed with the pass-A geometry
          -> candidates matched to an object are positives (label 1), others negatives
Saved per candidate: features, label, recording, pass, sequence, frame, feature-track id, rule-path outcome.
Saved per object/frame: ground truth for recall computation.
"""
import os
import sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'tunnel_guard'))
import numpy as np
from loader import BagCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.features import FEATURE_NAMES
from tunnel_guard.core.synth import Shape, inject, random_hazard

bags = sys.argv[1].split(',')
tag = sys.argv[2]
SEQ, WARM, EVERY = 10, 12, 30
rng = np.random.default_rng(abs(hash(tag)) % 2**32)


def random_shape():
    if rng.random() < 0.45:
        return Shape('cylinder', (float(rng.uniform(0.15, 0.32)), float(rng.uniform(0.8, 1.95))), float(rng.uniform(0.05, 0.45)), 'person-like')
    return Shape('box', (float(rng.uniform(0.2, 1.4)), float(rng.uniform(0.2, 1.2)), float(rng.uniform(0.2, 1.6))),
                 float(rng.uniform(0.04, 0.5)), 'box')


rows_X, rows_meta, objs = [], [], []


def record(res, bag_i, pas, seq, f, objects):
    for c, x in zip(res.all_candidates, res.features):
        label, obj_id = 0, -1
        for o in objects:
            if abs(c['s_min'] - o['dist']) < 1.5 + 0.02 * o['dist'] and abs(c['l_mean'] - o['lat']) < 0.9:
                label, obj_id = 1, o['id']
        tr = c['ml_view'].get('track')
        rows_X.append(x)
        rows_meta.append((bag_i, pas, seq, f, tr.id if tr is not None else -1, label, obj_id,
                          int(c['contained'] and not c['shell']), int(c['in_gauge_raw'] >= 2 and not (c['gravity_fail'] or c['shape_fail'])),
                          int(c['in_gauge_raw'])))


t0 = time.time()
for bag_i_local, nm in enumerate(bags):
    bag_i = ['roundT_doubleT', 'squareT_platform_squareT_switch', 'doubleT_platform',
             'roundT_squareT_pressureGate_squareT', 'roundT_pressureGate_roundT'].index(nm)
    b = BagCache(nm)
    det = ObstacleDetector(DetectorConfig(forward_axis='x'))
    geos = {}
    for f in range(b.n):
        p, I, _ = b.points(f)
        res = det.process(p, b.t[f], intensity=I)
        g = res.geometry
        geos[f] = (g.yc.copy(), g.zr.copy(), g.roll, g.ok)
        record(res, bag_i, 0, -1, f, [])
    print(f'{nm} pass A done {time.time() - t0:.0f}s rows {len(rows_X)}', flush=True)
    seq_id = bag_i * 1000
    for f0 in range(WARM + 5, b.n - SEQ, EVERY):
        seq_id += 1
        det = ObstacleDetector(DetectorConfig(forward_axis='x'))
        for f in range(f0 - WARM, f0):
            p, I, _ = b.points(f)
            det.process(p, b.t[f], intensity=I)
        objects = []
        used = []
        for k in range(2):
            for _ in range(20):
                d0 = float(rng.uniform(20, 195))
                if all(abs(d0 - u) > 15 for u in used):
                    break
            used.append(d0)
            objects.append(dict(id=seq_id * 10 + k, d0=d0, speed=float(rng.choice([0.0, rng.uniform(3, 15)])),
                                **random_hazard(rng)))
        for k in range(SEQ):
            f = f0 + k
            yc, zr, roll, ok = geos[f]
            rimg, inten = b.frame(f)
            now = []
            for o in objects:
                dist = o['d0'] - o['speed'] * 0.1 * k
                xi = int(np.clip(round(dist), 0, len(yc) - 1))
                centre = np.array([dist, yc[xi] + o['lat'], zr[xi] + roll * o['lat'] + o['lift']])
                rimg, nret = inject(rimg, b.dirs, centre, o['shape'], rng)
                now.append(dict(id=o['id'], dist=dist, lat=o['lat']))
                objs.append((bag_i, seq_id, f, k, o['id'], dist, o['lat'], nret, o['shape'].kind, *o['shape'].size, o['shape'].reflectivity, o['lift'], o['shape'].name))
            V = rimg > 0.5
            res = det.process(b.dirs[V] * rimg[V][:, None], b.t[f], intensity=inten[V])
            record(res, bag_i, 1, seq_id, f, now)
    print(f'{nm} pass B done {time.time() - t0:.0f}s rows {len(rows_X)}', flush=True)
    np.savez_compressed(f'ds_{tag}.npz', X=np.asarray(rows_X, np.float32), meta=np.asarray(rows_meta, np.int32),
                        names=np.array(FEATURE_NAMES), objs=np.array(objs, dtype=object))
print('done', time.time() - t0)
