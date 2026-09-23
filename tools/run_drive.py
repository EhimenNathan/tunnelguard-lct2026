"""Run the detector continuously over a time-ordered, per-piece cached drive (tools/convert_ds2.py output) and save
what the "closer look" self-labelling needs:
  frames.jsonl      per frame: index, time, level, clear distance, latency, obstacles (id, zone, s, l, h, n, score)
  occupancy.npy     (frames, 220) uint16: points inside the train envelope per 1 m band ahead (0-220 m), h 0.15-2.8 m
  occupancy_core.npy  same for the envelope core (|l| < 1.0 m, h 0.3-2.5 m): used by the closer-look labels
  odometry.npy      (frames,) float: train displacement since the previous frame from along-track registration
usage (scratchpad):  python run_drive.py cache/ds2 drive_out [first_piece last_piece] [config overrides as k=v ...]
"""
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))
sys.path.insert(0, os.getcwd())
import numpy as np

from loader import BagCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.geometry import warmup

MODEL = os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json')
S_BINS = 220
SIG = np.arange(6.0, 90.0, 0.1)            # along-track signature grid for odometry (0.1 m)


class DriveCache:
    """Pieces new_data_<N> of one split recording, in time order."""

    def __init__(self, root):
        names = [os.path.basename(p)[:-10] for p in glob.glob(os.path.join(root, '*_meta.json'))]
        names.sort(key=lambda n: int(re.findall(r'\d+$', n)[0]))
        self.pieces = [BagCache(n, root=root) for n in names]
        self.names = names
        self.index = [(pi, k) for pi, b in enumerate(self.pieces) for k in range(b.n)]
        t0 = self.pieces[0].stamps[0]
        self.t = np.concatenate([b.stamps - t0 for b in self.pieces])

    def __len__(self):
        return len(self.index)

    def cloud(self, i):
        pi, k = self.index[i]
        b = self.pieces[pi]
        rng, inten = b.frame(k)
        V = rng > 0.5
        fwd = b.dirs[V] * rng[V][:, None]
        # back to the sensor-frame convention (forward = -y) exactly as the ROS node receives it
        return np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1), inten[V]


def signature(s, l, h):
    """Detrended along-track profile of tunnel-wall structure (outside the envelope): discrete features such as brackets,
    lamps and ring joints dominate after removing the smooth density trend."""
    m = (np.abs(l) > 1.6) & (np.abs(l) < 4.5) & (h > 0.3) & (h < 4.0) & (s > SIG[0]) & (s < SIG[-1])
    hist = np.histogram(s[m], bins=np.r_[SIG, SIG[-1] + 0.1])[0].astype(float)
    trend = np.convolve(hist, np.ones(21) / 21, mode='same')               # 2 m moving average
    hist = (hist - trend) / np.sqrt(trend + 1.0)                             # Poisson-normalised residual
    hist = np.convolve(hist, np.array([0.25, 0.5, 0.25]), mode='same')
    n = np.linalg.norm(hist)
    return hist / n if n > 0 else hist


def shift_between(prev, cur, prior=None, max_shift=45):
    """Displacement d >= 0 [m] with cur(s) ~ prev(s + d) (the tunnel comes closer as the train advances), 0.1 m grid with
    parabolic refinement; a weak prior around the previous speed resolves periodic structure (ring spacing)."""
    cs = np.empty(max_shift + 1)
    for d in range(max_shift + 1):
        a, b_ = prev[d:], cur[:len(cur) - d]
        cs[d] = float(np.dot(a, b_) / max(np.linalg.norm(a) * np.linalg.norm(b_), 1e-9))
    score = cs.copy()
    if prior is not None and np.isfinite(prior):
        score = cs - 0.5 * ((np.arange(max_shift + 1) * 0.1 - prior) / 0.6) ** 2 * 0.05
    k = int(np.argmax(score))
    frac = 0.0
    if 0 < k < max_shift:
        y0, y1, y2 = cs[k - 1], cs[k], cs[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    return 0.1 * (k + frac), float(cs[k])


def main():
    root, out = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    span = [int(v) for v in rest[:2]] if len(rest) >= 2 and rest[0].isdigit() else None
    overrides = [a for a in rest if '=' in a]
    os.makedirs(out, exist_ok=True)
    drive = DriveCache(root)
    cfg = DetectorConfig(scorer_model=MODEL)
    for kv in overrides:
        k, v = kv.split('=', 1)
        obj = cfg
        parts = k.split('.')
        for p_ in parts[:-1]:
            obj = getattr(obj, p_)
        cur = getattr(obj, parts[-1])
        setattr(obj, parts[-1], type(cur)(v) if not isinstance(cur, bool) else v.lower() in ('1', 'true'))
    warmup()
    det = ObstacleDetector(cfg)
    idx = range(len(drive))
    if span:
        idx = [i for i in idx if span[0] <= int(re.findall(r'\d+$', drive.names[drive.index[i][0]])[0]) <= span[1]]
    occ = np.zeros((len(drive), S_BINS), np.uint16)
    occ_core = np.zeros((len(drive), S_BINS), np.uint16)
    odo = np.full(len(drive), np.nan)
    corr = np.full(len(drive), np.nan)
    prev_sig = None
    fout = open(os.path.join(out, 'frames.jsonl'), 'w')
    t0 = time.time()
    for n_done, i in enumerate(idx):
        xyz, inten = drive.cloud(i)
        res = det.process(xyz, drive.t[i], intensity=inten)
        g = res.geometry
        if g is not None and g.ok:
            s, l, h = g.to_track(res.forward_points)
            inside = (np.abs(l) < 1.35) & (h > 0.15) & (h < 2.8) & (s > 0) & (s < S_BINS)
            occ[i] = np.bincount(s[inside].astype(int), minlength=S_BINS)[:S_BINS].astype(np.uint16)
            core = (np.abs(l) < 1.0) & (h > 0.3) & (h < 2.5) & (s > 0) & (s < S_BINS)
            occ_core[i] = np.bincount(s[core].astype(int), minlength=S_BINS)[:S_BINS].astype(np.uint16)
            sig = signature(s, l, h)
            if prev_sig is not None:
                prior = np.nanmedian(odo[max(0, i - 5):i]) if i > 0 and np.isfinite(odo[max(0, i - 5):i]).any() else None
                odo[i], corr[i] = shift_between(prev_sig, sig, prior)
            prev_sig = sig
        else:
            prev_sig = None
        obs = [dict(id=int(o['id']), zone=int(o['zone']), s=round(float(o['distance']), 2), l=round(float(o['lateral']), 2),
                    h=round(float(o['height']), 2), n=int(o['n']), score=round(float(o['confidence']), 3)) for o in res.obstacles]
        pi, k = drive.index[i]
        fout.write(json.dumps(dict(i=i, piece=drive.names[pi], k=k, t=round(float(drive.t[i]), 3), level=int(res.level),
                                   clear=round(float(res.clear_distance), 1), ms=round(1e3 * res.timings['total'], 1),
                                   obs=obs)) + '\n')
        if n_done % 500 == 0:
            fout.flush()
            print(f'{n_done}/{len(idx)} frames  [{time.time() - t0:.0f}s]', flush=True)
    fout.close()
    np.save(os.path.join(out, 'occupancy.npy'), occ)
    np.save(os.path.join(out, 'occupancy_core.npy'), occ_core)
    np.save(os.path.join(out, 'odometry.npy'), np.stack([odo, corr], 1))
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
