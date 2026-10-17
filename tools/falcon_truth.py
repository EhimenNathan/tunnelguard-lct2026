"""Reference trajectory of the walking person in the Falcon recording (static train), independent of the detector:
static background = 0.5 m voxels occupied in >= 25 % of sampled frames; per frame, the non-background points at person
height inside the corridor are clustered and the largest cluster is taken as the person.
usage: python falcon_truth.py <cache.bin> <out.json>"""
import json
import sys

import numpy as np

from falcon_cache import frames_cached

V = 0.5
Y0, HALF = -1.9, 2.6                 # corridor centre / half width in the sensor frame [m]


def floor_z(x):
    return -2.1 - 0.042 * x          # floor line measured in the recording (lidar tilted ~2.4 deg up)


def vox(p):
    return np.floor(p / V).astype(np.int32)


def keys(v):
    return (v[:, 0].astype(np.int64) << 32) ^ ((v[:, 1].astype(np.int64) & 0xFFFF) << 16) ^ (v[:, 2].astype(np.int64) & 0xFFFF)


def main(cache, out):
    counts, nsamp = {}, 0
    for k, t, xyz, I in frames_cached(cache, step=5):
        u = np.unique(keys(vox(xyz)))
        for q in u.tolist():
            counts[q] = counts.get(q, 0) + 1
        nsamp += 1
    bg = np.array([q for q, c in counts.items() if c >= 0.25 * nsamp], np.int64)
    bg.sort()
    print('background voxels', len(bg), 'from', nsamp, 'frames', flush=True)
    traj = []
    for k, t, xyz, I in frames_cached(cache):
        h = xyz[:, 2] - floor_z(xyz[:, 0])
        m = (np.abs(xyz[:, 1] - Y0) < HALF) & (h > 0.5) & (h < 2.4) & (xyz[:, 0] > 3)
        q = xyz[m]
        kk = keys(vox(q))
        pos = np.searchsorted(bg, kk)
        pos[pos >= len(bg)] = 0
        dyn = q[bg[pos] != kk]
        rec = dict(k=k, t=round(float(t), 2), n=0, clusters=[])
        if len(dyn) >= 1:
            o = np.argsort(dyn[:, 0])
            d = dyn[o]
            cut = np.nonzero(np.diff(d[:, 0]) > 1.5)[0] + 1
            for part in np.split(d, cut):
                hh = part[:, 2] - floor_z(part[:, 0])
                rec['clusters'].append([float(np.median(part[:, 0])), float(np.median(part[:, 1])), int(len(part)),
                                        float(hh.max()), float(hh.min())])
        traj.append(rec)
        if k % 1000 == 0:
            print(k, len(rec['clusters']), 'clusters', flush=True)
    traj = link(traj)
    json.dump(traj, open(out, 'w'))


def link(traj, gate=3.0, coast=30):
    """Follow the person through the per-frame clusters: seed at the frame with the strongest tall cluster, then extend
    forwards and backwards choosing the cluster nearest to the last position (gate grows while coasting)."""
    tall = lambda c: c[3] > 0.6
    seed = max(range(len(traj)), key=lambda i: max([c[2] for c in traj[i]['clusters'] if tall(c)] or [0]))
    best = max([c for c in traj[seed]['clusters'] if tall(c)], key=lambda c: c[2])
    out = {seed: best}
    for direction in (1, -1):
        last, miss, i = best, 0, seed + direction
        while 0 <= i < len(traj):
            cand = [c for c in traj[i]['clusters'] if tall(c) and abs(c[0] - last[0]) < gate + 0.3 * miss
                    and abs(c[1] - last[1]) < 1.5]
            if cand:
                c = min(cand, key=lambda c: abs(c[0] - last[0]))
                out[i] = c
                last, miss = c, 0
            else:
                miss += 1
                if miss > coast:
                    miss = coast
            i += direction
    for i, r in enumerate(traj):
        c = out.get(i)
        r.pop('clusters', None)
        if c is None:
            r.update(n=0)
        else:
            r.update(n=c[2], x=c[0], y=c[1], h_top=c[3], h_bot=c[4])
    return traj


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
