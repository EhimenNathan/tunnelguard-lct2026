"""Candidate dataset from the new 20-minute drive (dataset2) for domain-robust training of the scorer.

The drive is verified obstacle-free (the train later drives through the position of every detection - selflabel.py /
traversal test), so every candidate on the clean frames is a real negative from a line the scorer has never seen.
Positives: ray-cast objects inserted into the real beams from warm detector snapshots (as gen_dataset_v2.py).
The drive is split into four 300 s blocks; block 3 (t >= 900 s) is the untouched final test.
Meta columns: group, pass, seq, frame, ftid, label, objid, rule_kept, rule_in, in_raw, beyond, t   (group = 10 + block)
usage (scratchpad):  python gen_dataset_v3.py cache/ds2 ds3_drive.npz
"""
import copy
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))
sys.path.insert(0, os.getcwd())
import numpy as np

from run_drive import DriveCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.features import FEATURE_NAMES
from tunnel_guard.core.geometry import GeometryEstimator, warmup
from tunnel_guard.core.synth import Shape, inject, random_hazard

SEQ, EVERY, START = 10, 40, 30
rng = np.random.default_rng(20260918)


def random_shape():
    if rng.random() < 0.45:
        return Shape('cylinder', (float(rng.uniform(0.15, 0.32)), float(rng.uniform(0.8, 1.95))), float(rng.uniform(0.05, 0.45)),
                     'person-like')
    return Shape('box', (float(rng.uniform(0.2, 1.4)), float(rng.uniform(0.2, 1.2)), float(rng.uniform(0.2, 1.6))),
                 float(rng.uniform(0.04, 0.5)), 'box')


def main():
    root, out = sys.argv[1], sys.argv[2]
    drive = DriveCache(root)
    warmup()
    t0 = time.time()
    # pass A1: clean-frame reference geometry (placement)
    ref = GeometryEstimator()
    geos = {}
    for i in range(len(drive)):
        xyz, _ = drive.cloud(i)
        fwd = np.stack([-xyz[:, 1], -xyz[:, 0], xyz[:, 2]], 1)
        g = ref.estimate(fwd[fwd[:, 0] > 1.0])
        geos[i] = (g.yc.astype(np.float32), g.zr.astype(np.float32), float(g.roll), bool(g.ok))
    print(f'reference geometry done {time.time() - t0:.0f}s', flush=True)
    rows_X, rows_meta, objs = [], [], []

    def record(res, det, block, pas, seq, i, objects):
        g = res.geometry
        measured = max(g.valid_range, g.rail_range) + det.cfg.horizon_slack if (g is not None and g.ok) else 0.0
        for c, x in zip(res.all_candidates or [], res.features if res.features is not None else []):
            label, obj_id = 0, -1
            for o in objects:
                # objects are placed in the loader frame, whose lateral axis is mirrored relative to the detector's
                if abs(c['s_min'] - o['dist']) < 1.5 + 0.02 * o['dist'] and abs(c['l_mean'] + o['lat']) < 0.9:
                    label, obj_id = 1, o['id']
            tr = c['ml_view'].get('track')
            rows_X.append(x)
            rows_meta.append((10 + block, pas, seq, i, tr.id if tr is not None else -1, label, obj_id,
                              int(c['contained'] and not c['shell']),
                              int(c['in_gauge_raw'] >= 2 and not (c['gravity_fail'] or c['shape_fail'])),
                              int(c['in_gauge_raw']), int(c['s_min'] > measured), int(round(drive.t[i] * 10))))

    det = ObstacleDetector(DetectorConfig(scorer_model=os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json')))
    starts = set(range(START, len(drive) - SEQ, EVERY))
    seq_id = 100000
    for i in range(len(drive)):
        block = min(int(drive.t[i] // 300), 3)
        if i in starts:
            seq_id += 1
            d = copy.deepcopy(det)
            objects, used = [], []
            for kk in range(2):
                for _ in range(30):
                    d0 = float(rng.uniform(15, 160))
                    if all(abs(d0 - u) > 15 for u in used):
                        break
                used.append(d0)
                objects.append(dict(id=seq_id * 10 + kk, d0=d0, speed=float(rng.choice([0.0, rng.uniform(3, 15)])),
                                    **random_hazard(rng)))
            for k in range(SEQ):
                j = i + k
                yc, zr, roll, ok = geos[j]
                pi, kk_ = drive.index[j]
                b = drive.pieces[pi]
                rimg, inten = b.frame(kk_)
                now = []
                for o in objects:
                    dist = o['d0'] - o['speed'] * 0.1 * k
                    xi = int(np.clip(round(dist), 0, len(yc) - 1))
                    centre = np.array([dist, yc[xi] + o['lat'], zr[xi] + roll * o['lat'] + o['lift']])
                    rimg, nret = inject(rimg, b.dirs, centre, o['shape'], rng)
                    now.append(dict(id=o['id'], dist=dist, lat=o['lat']))
                    objs.append((10 + block, seq_id, j, k, o['id'], dist, o['lat'], nret, o['shape'].kind, *o['shape'].size,
                                 o['shape'].reflectivity, o['lift'], o['shape'].name))
                V = rimg > 0.5
                fwd = b.dirs[V] * rimg[V][:, None]
                res = d.process(np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1), drive.t[j], intensity=inten[V])
                record(res, d, block, 1, seq_id, j, now)
            del d
        xyz, inten = drive.cloud(i)
        res = det.process(xyz, drive.t[i], intensity=inten)
        record(res, det, block, 0, -1, i, [])
        if i % 1000 == 0:
            print(f'{i}/{len(drive)} frames, rows {len(rows_X)}  [{time.time() - t0:.0f}s]', flush=True)
    np.savez_compressed(out, X=np.asarray(rows_X, np.float32), meta=np.asarray(rows_meta, np.int64),
                        names=np.array(FEATURE_NAMES), objs=np.array(objs, dtype=object),
                        n_frames=np.array([sum(1 for i in range(len(drive)) if min(int(drive.t[i] // 300), 3) == blk)
                                           for blk in range(4)]))
    print('done', time.time() - t0, 'rows', len(rows_X))


if __name__ == '__main__':
    main()
