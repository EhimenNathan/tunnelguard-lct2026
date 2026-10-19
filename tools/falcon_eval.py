"""Run the detector on segments of the Falcon cache and compare with the reference trajectory (falcon_truth.py).
usage: python falcon_eval.py <cache.bin> <truth.json> <out.json> [start:stop[:step] ...]  [key=value config overrides]
Default segments cover the person at ~260, ~300, ~230, ~130 and ~60 m."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard'))
import numpy as np

from falcon_cache import frames_cached
from tunnel_guard.core.detector import DetectorConfig, ObstacleDetector
from tunnel_guard import params as P

CFG_DIR = os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard', 'config')
if not os.path.isdir(CFG_DIR):
    from ament_index_python.packages import get_package_share_directory
    CFG_DIR = os.path.join(get_package_share_directory('tunnel_guard'), 'config')


def make_detector(overrides):
    cfg = DetectorConfig(scorer_model=os.path.join(CFG_DIR, 'obstacle_scorer.json'))
    for kv in overrides:
        k, v = kv.split('=', 1)
        P.apply(cfg, k, json.loads(v))
    return ObstacleDetector(cfg)


def fill_gaps(traj, max_gap=30):
    """The person moves continuously: interpolate the reference over short gaps (frames where too few points were
    found to form a cluster)."""
    known = [i for i, r in enumerate(traj) if r.get('n', 0) > 0]
    for a, b in zip(known, known[1:]):
        if 1 < b - a <= max_gap:
            for i in range(a + 1, b):
                w = (i - a) / (b - a)
                ra, rb = traj[a], traj[b]
                traj[i] = dict(traj[i], n=1, x=ra['x'] + w * (rb['x'] - ra['x']), y=ra['y'] + w * (rb['y'] - ra['y']),
                               interp=True)
    return traj


def run(cache, truth, segments, overrides):
    T = {r['k']: r for r in fill_gaps(json.load(open(truth)))}
    rows = []
    for seg in segments:
        a = [int(v) for v in seg.split(':')]
        start, stop, step = a[0], a[1], (a[2] if len(a) > 2 else 1)
        det = make_detector(overrides)
        for k, t, xyz, inten in frames_cached(cache, start, stop, step):
            t0 = time.perf_counter()
            res = det.process(xyz, t, intensity=inten)
            ms = 1e3 * (time.perf_counter() - t0)
            tr = T.get(k, {})
            obs = [(round(o['distance'], 1), round(o['lateral'], 2), o['zone'], int(o['n'])) for o in res.obstacles]
            rows.append(dict(k=k, seg=seg, level=int(res.level), nearest=None if np.isnan(res.nearest_distance) else
                             round(float(res.nearest_distance), 1), clear=round(float(res.clear_distance), 1),
                             ms=round(ms, 1), obs=obs, truth_x=tr.get('x') if tr.get('n', 0) else None, truth_y=tr.get('y'), truth_n=tr.get('n', 0), env=round(float(res.envelope_range), 1),
                             ncand=len(res.candidates)))
    return rows


def summarise(rows):
    for seg in dict.fromkeys(r['seg'] for r in rows):
        R = [r for r in rows if r['seg'] == seg]
        lv = np.array([r['level'] for r in R])
        tx = np.array([r['truth_x'] or np.nan for r in R], float)
        # detection of the person: an obstacle within 6 m of the reference position
        hit = [any(abs(o[0] - r['truth_x']) < 6 for o in r['obs']) if r['truth_x'] else False for r in R]
        stop_hit = [r['level'] == 2 and r['nearest'] is not None and r['truth_x'] and abs(r['nearest'] - r['truth_x']) < 6
                    for r in R]
        other = [any(not (r['truth_x'] and abs(o[0] - r['truth_x']) < 6) for o in r['obs']) for r in R]
        ms = np.array([r['ms'] for r in R])
        print(f"seg {seg:12s} person {np.nanmedian(tx):6.1f} m | detected {np.mean(hit) * 100:5.1f}% STOP-on-person "
              f"{np.mean(stop_hit) * 100:5.1f}% | frames with other obstacles {np.mean(other) * 100:5.1f}% | "
              f"STOP {np.mean(lv == 2) * 100:5.1f}% CAUTION {np.mean(lv == 1) * 100:5.1f}% | clear med "
              f"{np.median([r['clear'] for r in R]):5.0f} m | ms med {np.median(ms):5.1f} p95 {np.percentile(ms, 95):5.1f}")


TRACK_Y = 0.35          # our track centre in the levelled sensor frame (rails at -0.4 / +1.1 m)


def report_full(rows):
    """Continuous run: detection by distance band while the person is on our track, false alarms while beside it."""
    on = lambda r: r['truth_x'] is not None and r['truth_n'] >= 1 and abs(r.get('truth_y', 0) - TRACK_Y) < 1.3
    beside = lambda r: r['truth_x'] is not None and r['truth_n'] >= 1 and abs(r.get('truth_y', 0) - TRACK_Y) >= 1.6
    near_p = lambda r, d: r['truth_x'] is not None and abs(d - r['truth_x']) < 6
    R = [r for r in rows if on(r)]
    print(f'frames {len(rows)}; person on our track {len(R)}, beside it {sum(map(beside, rows))}')
    first = next((r for r in rows if on(r) and any(near_p(r, o[0]) for o in r['obs'])), None)
    first_stop = next((r for r in rows if on(r) and r['level'] == 2 and r['nearest'] is not None and near_p(r, r['nearest'])), None)
    print('first detection of the person:', None if first is None else f"frame {first['k']} at {first['truth_x']:.1f} m")
    print('first STOP on the person     :', None if first_stop is None else f"frame {first_stop['k']} at {first_stop['truth_x']:.1f} m")
    for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 320)):
        B = [r for r in R if lo <= r['truth_x'] < hi]
        if not B:
            continue
        det = np.mean([any(near_p(r, o[0]) for o in r['obs']) for r in B])
        stp = np.mean([r['level'] == 2 and r['nearest'] is not None and near_p(r, r['nearest']) for r in B])
        env = np.mean([r.get('env', r['clear']) >= r['truth_x'] - 5 for r in B])
        print(f'  on track {lo:3d}-{hi:3d} m: frames {len(B):4d} detected {100 * det:5.1f}%  STOP {100 * stp:5.1f}%  '
              f'envelope reaches person {100 * env:5.1f}%')
    Bs = [r for r in rows if beside(r)]
    if Bs:
        fp_b = np.mean([r['level'] == 2 and r['nearest'] is not None and near_p(r, r['nearest']) for r in Bs])
        print(f'  person BESIDE the track ({len(Bs)} frames): STOP on the person {100 * fp_b:.1f}% (should be 0)')
    other = [r for r in rows if any(not near_p(r, o[0]) for o in r['obs'])]
    ostop = [r for r in rows if r['level'] == 2 and not (r['nearest'] is not None and near_p(r, r['nearest']))]
    print(f'frames with an obstacle away from the person: {len(other)} ({100 * len(other) / len(rows):.1f}%); '
          f'STOP not on the person: {len(ostop)} ({100 * len(ostop) / len(rows):.1f}%)')
    cl = np.array([r['clear'] for r in rows]); ev = np.array([r.get('env', 0) for r in rows])
    print(f'clear distance: zero in {100 * np.mean(cl == 0):.1f}%, median {np.median(cl):.0f} m, p90 {np.percentile(cl, 90):.0f} m; '
          f'envelope range median {np.median(ev):.0f} m; frame-to-frame |change| of envelope median {np.median(np.abs(np.diff(ev))):.1f} m')
    ms = np.array([r['ms'] for r in rows])
    print(f'latency ms: median {np.median(ms):.1f}  p95 {np.percentile(ms, 95):.1f}  p99 {np.percentile(ms, 99):.1f}  max {ms.max():.1f}')


if __name__ == '__main__':
    cache, truth, out = sys.argv[1:4]
    rest = sys.argv[4:]
    segs = [a for a in rest if ':' in a and '=' not in a] or ['0:150', '600:750', '2050:2200', '2850:3000', '3850:4000']
    ov = [a for a in rest if '=' in a]
    rows = run(cache, truth, segs, ov)
    json.dump(rows, open(out, 'w'))
    summarise(rows) if len(segs) > 1 else report_full(rows)
