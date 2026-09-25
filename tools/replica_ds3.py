"""Replica of the organisers' synthetic-obstacle test on our own data (evaluation only - never used for training).

The organisers' bag (dataset3) places ten objects ~100 m apart, in the lidar frame on a flat, straight plane at the
rail-head level under the train (measured from their data: objects float above a descending track, lateral offsets drift
on curves).  Only 29 % of their file is readable, so this script rebuilds the whole sequence in the real beams of a
moving train on the new line (dataset2) and scores every object:  expected STOP for objects inside the envelope,
no STOP for objects outside or above it.
usage (scratchpad):  python replica_ds3.py out_dir [t_start] [k=v config overrides]
"""
import json
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
from tunnel_guard.core.geometry import warmup
from tunnel_guard.core.synth import Shape, inject

HC = 1.15     # height of the envelope centre above the rail head [m] (the organisers' 0.3 m "centre" cube: 1.0-1.3 m)
# (name, box (along, lateral, height), lateral centre [m, + left], bottom above rail [m], expect STOP?)
OBJECTS = [
    ('1 · 2×2 м в центре габарита', (2.0, 2.0, 2.0), 0.0, HC - 1.0, True),
    ('2 · 0.3 м в центре габарита', (0.3, 0.3, 0.3), 0.0, HC - 0.15, True),
    ('3 · 0.3 м на рельсе', (0.3, 0.3, 0.3), 0.8, 0.0, True),
    ('4 · 0.3 м у края габарита', (0.3, 0.3, 0.3), 1.15, HC - 0.15, True),
    ('5 · 0.3 м за габаритом, рядом', (0.3, 0.3, 0.3), 1.65, HC - 0.15, False),
    ('6 · 2×2 м у края, в габарите', (2.0, 2.0, 2.0), 0.35, HC - 1.0, True),
    ('7 · 2×2 м за габаритом', (2.0, 2.0, 2.0), 2.45, HC - 1.0, False),
    ('8 · 2×2 м над габаритом', (2.0, 2.0, 2.0), 0.0, 3.45, False),
    ('9 · 2×0.2 м лежит на рельсах', (0.2, 2.0, 0.2), 0.0, 0.0, True),
    ('10 · 0.05 м свисает с потолка', (0.05, 0.05, 1.8), 0.0, 3.8 - 1.8, True),
]
SPACING, FIRST = 100.0, 130.0


def main():
    out = sys.argv[1]
    rest = sys.argv[2:]
    t_start = float(rest[0]) if rest and '=' not in rest[0] else 1028.0
    cfg = DetectorConfig(scorer_model=os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json'))
    for kv in [a for a in rest if '=' in a]:
        k, v = kv.split('=', 1)
        obj = cfg
        for p_ in k.split('.')[:-1]:
            obj = getattr(obj, p_)
        cur = getattr(obj, k.split('.')[-1])
        setattr(obj, k.split('.')[-1], v.lower() in ('1', 'true') if isinstance(cur, bool) else type(cur)(v))
    os.makedirs(out, exist_ok=True)
    warmup()
    drive = DriveCache('cache/ds2')
    idx = [i for i in range(len(drive)) if drive.t[i] >= t_start - 15.0]
    det = ObstacleDetector(cfg)
    rng = np.random.default_rng(3)
    X0 = None
    z0 = yref = None
    fout = open(os.path.join(out, 'frames.jsonl'), 'w')
    t0 = time.time()
    for n, i in enumerate(idx):
        t = float(drive.t[i])
        pi, kf = drive.index[i]
        piece = drive.pieces[pi]
        rimg, _ = piece.frame(kf)
        placed = []
        if t >= t_start and z0 is not None:
            if X0 is None:
                X0 = det.calib.x
            for k, (name, size, lat, bottom, stop) in enumerate(OBJECTS):
                dist = X0 + FIRST + k * SPACING - det.calib.x
                if 3.0 < dist < 230.0:
                    # flat & straight in the lidar frame (loader frame: lateral axis mirrored w.r.t. the detector)
                    centre = np.array([dist, -(yref + lat), z0 + bottom])
                    rimg, nret = inject(rimg, piece.dirs, centre, Shape('box', size, 0.2), rng)
                    placed.append(dict(k=k, dist=round(float(dist), 2), hits=int(nret)))
        V = rimg > 0.5
        fwd = piece.dirs[V] * rimg[V][:, None]
        res = det.process(np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1), t)
        g = res.geometry
        if g is not None and g.ok and (X0 is None):
            z0, yref = float(g.rail_z(8.0)), float(g.centre(8.0))     # plane frozen when the sequence starts
        obs = [dict(zone=int(o['zone']), s=round(float(o['distance']), 2), l=round(float(o['lateral']), 2),
                    h=round(float(o['height']), 2)) for o in res.obstacles]
        near = [dict(s=round(float(c['s_min']), 2), l=round(float(c['l_mean']), 2), l0=round(float(c['l_min']), 2),
                     l1=round(float(c['l_max']), 2), h0=round(float(c['h_min']), 2), h1=round(float(c['h_max']), 2),
                     n=int(c['n']), ig=int(c.get('in_gauge_raw', c['in_gauge'])), sc=round(float(c.get('score', -1)), 3),
                     shell=bool(c['shell']), cont=bool(c['contained']), grav=bool(c['gravity_fail']),
                     shape=bool(c['shape_fail']))
                for c in (res.all_candidates or []) if any(abs(c['s_min'] - p['dist']) < 3 + 0.03 * p['dist'] for p in placed)]
        fout.write(json.dumps(dict(t=round(t, 2), level=int(res.level), x=round(float(det.calib.x), 2), placed=placed,
                                   obs=obs, cands=near)) + '\n')
        if X0 is not None and det.calib.x - X0 > FIRST + len(OBJECTS) * SPACING:
            break
        if n % 200 == 0:
            print(n, 'frames', round(time.time() - t0), 's', flush=True)
    fout.close()
    print('done', round(time.time() - t0), 's', flush=True)


def score(run):
    L = [json.loads(l) for l in open(f'{run}/frames.jsonl')]
    rows = []
    for k, (name, size, lat, bottom, stop) in enumerate(OBJECTS):
        seen = [(f, next(p for p in f['placed'] if p['k'] == k)) for f in L if any(p['k'] == k for p in f['placed'])]
        first_c = first_s = None
        stop_n = within = 0
        for f, p in seen:
            d = p['dist']
            match = [o for o in f['obs'] if abs(o['s'] - d) < 3 + 0.03 * d]
            if match and first_c is None:
                first_c = d
            st = [o for o in match if o['zone'] == 2]
            if st and first_s is None:
                first_s = d
            if d < 100:
                within += 1
                stop_n += bool(st)
        rows.append((name, stop, first_c, first_s, stop_n, within))
    other = sum(1 for f in L if f['level'] == 2 and not any(
        abs(o['s'] - p['dist']) < 3 + 0.03 * p['dist'] for o in f['obs'] if o['zone'] == 2 for p in f['placed']))
    return rows, other, len(L)


if __name__ == '__main__':
    if sys.argv[1] == 'score':
        for run in sys.argv[2:]:
            rows, other, n = score(run)
            print('==', run, f'({n} frames; STOP frames not on any object: {other})')
            for name, stop, fc, fs, sn, w in rows:
                verdict = ('OK' if (fs is not None) == stop else 'MISS' if stop else 'FALSE STOP')
                print(f"  {name:34s} expect {'STOP' if stop else 'no STOP':8s} first alarm {fc if fc else '—':>6} m  "
                      f"first STOP {fs if fs else '—':>6} m  STOP in {sn:3d}/{w:3d} frames <100 m   {verdict}")
    else:
        main()
