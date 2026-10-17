"""Generalisation to curved track: analytic round tunnel (known centreline) at radii 300 / 500 / 800 m and straight,
dense (Pandar-like) and thinned to Falcon-like density; centreline error vs distance, and detection of a 0.6 m box
standing on the curved track at 40-150 m.
usage: python curve_test.py [out.json]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard')
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, 'test'))
import numpy as np

from tunnel_sim import centreline, make_tunnel
from tunnel_guard.core.detector import DetectorConfig, ObstacleDetector
MODEL = os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json')


def thin(p, rng):
    """Falcon-like: far fewer returns with distance (vertical lines every ~0.1-0.24 deg)."""
    keep = rng.random(len(p)) < np.clip(9.0 / np.maximum(p[:, 0], 1.0), 0.04, 1.0)
    return p[keep]


def main(out=None):
    rows = []
    for radius in (None, 800.0, 500.0, 300.0):
        for sparse in (False, True):
            rng = np.random.default_rng(1)
            det = ObstacleDetector(DetectorConfig(forward_axis='x', mount_calibration=False, scorer_model=MODEL))
            errs = {x: [] for x in (25, 50, 75, 100, 150)}
            for k in range(30):
                p = make_tunnel(radius=radius, seed=k)
                p = thin(p, rng) if sparse else p
                res = det.process(p, 0.1 * k)
                g = res.geometry
                if k >= 20 and g.ok:
                    for x in errs:
                        errs[x].append(abs(float(g.centre(np.array([x]))[0]) - float(centreline(np.array([float(x)]), radius)[0])))
            det_rates = {}
            for d in (40, 80, 120, 150):
                rng2 = np.random.default_rng(d)
                det2 = ObstacleDetector(DetectorConfig(forward_axis='x', mount_calibration=False, scorer_model=MODEL))
                found = 0
                for k in range(35):
                    obs = [(float(d), 0.0, 0.6, 0.6, 1.0)] if k >= 20 else []
                    p = make_tunnel(radius=radius, seed=100 + k, obstacles=obs)
                    p = thin(p, rng2) if sparse else p
                    res = det2.process(p, 0.1 * k)
                    if k >= 25:
                        # to_track uses forward X, not integrated arc length.
                        found += any(abs(o['distance'] - d) < 4 and abs(o['lateral']) < 1 for o in res.obstacles)
                det_rates[d] = found / 10
            row = dict(radius=radius or 'straight', density='falcon-like' if sparse else 'dense',
                       centre_error_cm={x: round(100 * float(np.median(v)), 1) if v else None for x, v in errs.items()},
                       box_detected={d: det_rates[d] for d in det_rates})
            rows.append(row)
            print(row, flush=True)
    if out:
        json.dump(rows, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
