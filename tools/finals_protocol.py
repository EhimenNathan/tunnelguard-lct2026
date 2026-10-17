"""Shared evaluation definitions; imports no ROS, cache decoder or detector."""
import numpy as np

DISTANCES = [60, 90, 100, 120, 150, 175, 180, 190, 200, 215, 220, 230, 250, 260, 270, 280, 300]
OBJECTS = {
    'person': dict(kind='person', size=(0.5, 0.3, 1.75), rho=0.10),
    'box_0.5': dict(kind='box', size=(0.5, 0.5, 0.5), rho=0.20, z0=-1.80),       # standing on the rails
    'box_0.3': dict(kind='box', size=(0.3, 0.3, 0.3), rho=0.20, z0=-1.80),
    'dust': dict(kind='dust', size=(2.4, 3.0, 2.5), sigma=0.6, rho=0.05, z0=-1.9),
}
WARM, ON = 30, 50


def segment_for(d, truth):
    """First frame of an 80-frame stretch where the real person stays >= 25 m from distance d."""
    xs = np.array([truth.get(k, {}).get('x', np.nan) if truth.get(k, {}).get('n') else np.nan for k in range(4443)])
    for start in range(100, 4443 - WARM - ON, 20):
        seg = xs[start:start + WARM + ON]
        if np.all(np.isnan(seg) | (np.abs(seg - d) > 25)):
            return start
    return 100


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

