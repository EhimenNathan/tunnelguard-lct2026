"""Self-supervised "closer look" labels for an unannotated drive (run_drive.py output).

Physics: a real object in the train's path produces a dense cluster in the core of the envelope once the train is close
(the number of returns grows ~1/d^2), whereas a far false alarm leaves the core empty when looked at from close range.
With lidar odometry every detection is pinned to a tunnel coordinate X = x(t) + d.

  precision : each STOP event is REAL if at 12-40 m the envelope core around X is densely occupied, FALSE if it is at the
              empty-tunnel baseline, UNRESOLVED if the train never gets that close (end of data, stop, odometry gap)
  recall    : objects = dense near-range occupancy of the envelope core (12-40 m) clustered in X; each is DETECTED if a
              STOP was issued for its position, with the distance of the first STOP
usage (scratchpad):  python selflabel.py drive_out [json_out]
"""
import json
import os
import sys

import numpy as np

out = sys.argv[1]
L = [json.loads(l) for l in open(os.path.join(out, 'frames.jsonl'))]
occ = np.load(os.path.join(out, 'occupancy_core.npy')) if os.path.exists(os.path.join(out, 'occupancy_core.npy')) \
    else np.load(os.path.join(out, 'occupancy.npy'))
odo = np.load(os.path.join(out, 'odometry.npy'))
n = len(L)
step = odo[:, 0].copy()
step[~np.isfinite(step)] = 0.0
x = np.cumsum(step)                                   # tunnel coordinate of the train (m), per frame
t = np.array([f['t'] for f in L])
gap = np.r_[False, np.diff(t) > 0.25]                 # missing frames break the odometry chain
seg = np.cumsum(gap)
S = occ.shape[1]
# empty-tunnel baseline per distance bin: robust typical occupancy
base = np.percentile(occ, 75, axis=0).astype(float)
NEAR = (12, 40)


def core_count(j, d, half=1.5):
    lo, hi = max(int(d - half), 0), min(int(d + half) + 1, S)
    if hi <= lo:
        return 0.0, 0.0
    return float(occ[j, lo:hi].sum()), float(base[lo:hi].sum())


def closer_look(i0, X, tol0=2.0):
    """Occupancy at the tunnel position X seen from 12-40 m in later frames of the same continuous segment."""
    best = None
    for j in range(i0 + 1, n):
        if seg[j] != seg[i0]:
            break
        d = X - x[j]
        if d < NEAR[0]:
            break
        if d > NEAR[1]:
            continue
        tol = tol0 + 0.03 * (x[j] - x[i0])            # odometry drift allowance
        c, b = core_count(j, d, half=tol)
        r = (c + 1.0) / (b + 1.0)
        if best is None or c > best[1]:
            best = (j, c, b, r, d)
    return best


# ---------------------------------------------------------------- STOP events -> precision
events = {}
for i, f in enumerate(L):
    for o in f['obs']:
        if o['zone'] == 2:
            key = (seg[i], o['id'])
            if key not in events:
                events[key] = dict(i0=i, d0=o['s'], X=x[i] + o['s'], l=o['l'], h=o['h'], n=o['n'], score=o['score'], frames=0)
            events[key]['frames'] += 1
labels = []
for key, e in events.items():
    cl = closer_look(e['i0'], e['X'])
    if cl is None:
        lab = 'UNRESOLVED'
        info = {}
    else:
        j, c, b, r, d = cl
        lab = 'REAL' if (c >= 25 and r >= 4.0) else ('FALSE' if r < 2.0 else 'UNCERTAIN')
        info = dict(near_frame=j, near_dist=round(d, 1), core_points=int(c), baseline=round(b, 1), ratio=round(r, 1))
    labels.append(dict(event=f'{key[0]}:{key[1]}', t=round(t[e['i0']], 1), first_stop_m=e['d0'], lateral=e['l'], height=e['h'],
                       points=e['n'], score=e['score'], frames=e['frames'], label=lab, **info))

# ---------------------------------------------------------------- near-range objects -> recall
objs = []
for j in range(n):
    for d in range(NEAR[0], NEAR[1]):
        c, b = core_count(j, d + 0.5, half=0.5)
        if c >= 25 and (c + 1) / (b + 1) >= 6.0:
            objs.append((seg[j], x[j] + d + 0.5, j, d + 0.5, c))
objs.sort()
clusters = []
for sg, X, j, d, c in objs:
    if clusters and clusters[-1]['seg'] == sg and abs(X - clusters[-1]['X']) < 4.0:
        cl = clusters[-1]
        cl['n'] += 1
        cl['peak'] = max(cl['peak'], c)
        cl['frames'].append(j)
    else:
        clusters.append(dict(seg=sg, X=X, n=1, peak=c, frames=[j]))
objects = []
for cl in clusters:
    if cl['n'] < 3:                                   # seen in at least 3 frames from close range
        continue
    first_near = min(cl['frames'])
    first_stop = None
    for i in range(0, first_near + 1):
        if seg[i] != cl['seg']:
            continue
        for o in L[i]['obs']:
            if o['zone'] == 2 and abs(x[i] + o['s'] - cl['X']) < 3.0 + 0.03 * max(cl['X'] - x[i], 0):
                first_stop = (i, o['s'])
                break
        if first_stop:
            break
    objects.append(dict(t=round(t[first_near], 1), X=round(cl['X'], 1), frames=cl['n'], peak_points=int(cl['peak']),
                        detected=first_stop is not None,
                        first_stop_m=None if first_stop is None else first_stop[1]))

cnt = {k: sum(1 for e in labels if e['label'] == k) for k in ('REAL', 'FALSE', 'UNCERTAIN', 'UNRESOLVED')}
resolved = cnt['REAL'] + cnt['FALSE']
det = [o for o in objects if o['detected']]
res = dict(frames=n, stop_frames=sum(1 for f in L if f['level'] == 2),
           events=len(labels), labels=cnt,
           precision_resolved=round(cnt['REAL'] / resolved, 3) if resolved else None,
           near_objects=len(objects), near_objects_detected=len(det),
           recall_near_objects=round(len(det) / len(objects), 3) if objects else None,
           first_stop_distance_m=dict(median=float(np.median([o['first_stop_m'] for o in det])) if det else None,
                                      max=float(max(o['first_stop_m'] for o in det)) if det else None),
           odometry_speed_ms=dict(median=float(np.nanmedian(odo[:, 0]) * 10)))
json.dump(dict(summary=res, events=labels, objects=objects),
          open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(out, 'selflabel.json'), 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
