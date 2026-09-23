"""Trace why a far synthetic obstacle is (not) detected: per frame, count points per pipeline stage."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'tunnel_guard'))
import numpy as np
from loader import BagCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.geometry import GeometryEstimator
from tunnel_guard.core.synth import SHAPES, inject
from tunnel_guard.core.gauge import classify

nm, f0, d0, shape = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
b = BagCache(nm)
rng = np.random.default_rng(0)
ref = GeometryEstimator()
det = ObstacleDetector(DetectorConfig(forward_axis='x'))
for f in range(f0 - 12, f0 + 12):
    p, I, _ = b.points(f)
    g_ref = ref.estimate(p[p[:, 0] > 1])
    if f < f0:
        det.process(p, b.t[f], intensity=I)
        continue
    k = f - f0
    dist = d0 - k
    centre = np.array([dist, float(g_ref.centre(dist)), float(g_ref.rail_z(dist))])
    rimg, inten = b.frame(f)
    rimg2, nr = inject(rimg, b.dirs, centre, SHAPES[shape], rng)
    V = rimg2 > 0.5
    pts = b.dirs[V] * rimg2[V][:, None]
    res = det.process(pts, b.t[f], intensity=inten[V])
    g = res.geometry
    P = res.forward_points
    near_obj = (np.abs(P[:, 0] - dist) < 1.0) & (np.abs(P[:, 1] - centre[1]) < 0.6) & (P[:, 2] > centre[2] - 0.2)
    x, l, h = g.to_track(P[near_obj])
    sl, sv = g.sigma_at(x)
    z = classify(x, l, h, det.cfg.gauge, sl, sv)
    cands = [(round(c['s_min'], 1), round(c['l_mean'], 2), c['n'], c['in_gauge']) for c in res.candidates if abs(c['s_min'] - dist) < 4]
    tr = [(t.id, round(t.s, 1), t.hit_count, t.confirmed) for t in det.tracker.tracks if abs(t.s - dist) < 6]
    print(f'k{k} dist {dist:.0f} returns {nr} obj_pts {near_obj.sum()} valid {g.valid_range:.0f} sight {res.timings["sight"]:.0f} '
          f'yc_err {g.centre(dist) - g_ref.centre(dist):+.2f} zr_err {g.rail_z(dist) - g_ref.rail_z(dist):+.2f} '
          f'sig_l {float(g.sigma_at(np.array([dist]))[0][0]):.2f} h {np.round(np.percentile(h, [0, 100]), 2) if len(h) else None} '
          f'l {np.round(np.percentile(l, [0, 100]), 2) if len(l) else None} zones {np.bincount(z, minlength=3)} cands {cands} tracks {tr} level {res.level}')
