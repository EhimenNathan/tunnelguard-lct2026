"""Physics-grounded feature vector of an obstacle candidate (input of the learned second-stage scorer).

All features are expressed in quantities that transfer between tunnels and sensors: distances in track coordinates,
extents in lidar beam spacings, point counts normalised for range, geometry uncertainty and validity margins, and the
outcomes/evidence of the physical plausibility tests.  Deliberately excluded (they would encode how training data was
generated rather than what an obstacle is): intensity, ring count, track age, closing speed, candidates per frame.
"""
import numpy as np

from .gauge import half_width

BEAM_V = 0.00218      # Pandar128 vertical resolution in the central band (0.125 deg)
BEAM_H = 0.00175      # Pandar128 horizontal resolution at 10 Hz (0.1 deg)

FEATURE_NAMES = [
    's', 'log_s', 'n', 'log_n', 'n_norm', 's_ext',
    'l_mean', 'abs_l', 'l_ext', 'l_absmax', 'l_absmin',
    'h_min', 'h_max', 'h_ext', 'h_mid', 'h_ext_beams', 'l_ext_beams', 'n_per_vbeam',
    'in_gauge', 'in_frac', 'sigma_l', 'sigma_v', 'depth_lat', 'depth_top', 'depth_rel',
    'valid_margin', 'rail_margin', 'trusted_margin', 'sight_margin',
    'wall_left', 'wall_right', 'contained', 'shell_pts', 'gravity_fail', 'shape_fail',
    'hits', 'l_std', 'curv_abs',
]


def candidate_features(c, g, gauge_cfg, sight, track=None):
    s = max(c['s_min'], 1.0)
    n = c['n']
    l_ext = c['l_max'] - c['l_min']
    h_ext = c['h_max'] - c['h_min']
    same_side = c['l_min'] * c['l_max'] > 0
    l_absmax = max(abs(c['l_min']), abs(c['l_max']))
    l_absmin = min(abs(c['l_min']), abs(c['l_max'])) if same_side else 0.0
    sl, sv = g.sigma_at(np.array([s]))
    sl, sv = float(sl[0]), float(sv[0])
    top = max(v[0] for v in gauge_cfg.profile)
    h_mid = 0.5 * (c['h_min'] + c['h_max'])
    w_mid = float(half_width(np.array([min(h_mid, top - 0.01)]), gauge_cfg)[0])
    depth_lat = w_mid - l_absmin
    depth_top = top - c['h_min']
    ls = getattr(track, 'ls', None) if track is not None else None
    return [
        s, np.log(s), n, np.log(n), n * (s / 50.0) ** 2, c['s_max'] - c['s_min'],
        c['l_mean'], abs(c['l_mean']), l_ext, l_absmax, l_absmin,
        c['h_min'], c['h_max'], h_ext, h_mid, h_ext / (BEAM_V * s), l_ext / (BEAM_H * s), n / max(h_ext / (BEAM_V * s), 1.0),
        c.get('in_gauge_raw', c['in_gauge']), c.get('in_gauge_raw', c['in_gauge']) / n, sl, sv,
        depth_lat, depth_top, depth_lat / max(sl, 0.02),
        g.valid_range - s, g.rail_range - s, g.trusted_range - s, sight - s,
        c.get('wall_left', 0), c.get('wall_right', 0), float(c.get('contained', True)), c.get('shell_pts', 0),
        float(c.get('gravity_fail', False)), float(c.get('shape_fail', False)),
        float(track.hit_count) if track is not None else 1.0,
        float(np.std(ls[-5:])) if ls is not None and len(ls) > 1 else 0.0,
        abs(g.curvature(s)),
    ]
