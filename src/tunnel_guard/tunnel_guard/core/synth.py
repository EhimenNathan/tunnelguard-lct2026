"""Physically-motivated synthetic obstacle injection into real lidar frames (for evaluation only).

Obstacles are ray-cast into the sensor's actual beam directions, so they inherit the real angular sampling
(128 channels, azimuth step), occlude the tunnel behind them and lose returns with range according to a
reflectivity-dependent maximum range model.  This gives detection-range curves without any labelled data
and without touching the held-out obstacle recording.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Shape:
    kind: str          # 'box' | 'cylinder'
    size: tuple        # box: (length, width, height); cylinder: (radius, height)
    reflectivity: float = 0.15
    name: str = ''


SHAPES = {
    'person': Shape('cylinder', (0.25, 1.75), 0.15, 'person'),
    'crouching_person': Shape('cylinder', (0.30, 1.00), 0.15, 'crouching person'),
    'box_50cm': Shape('box', (0.5, 0.5, 0.5), 0.20, 'box 0.5 m'),
    'box_30cm': Shape('box', (0.3, 0.3, 0.3), 0.20, 'box 0.3 m'),
    'dark_box_50cm': Shape('box', (0.5, 0.5, 0.5), 0.05, 'dark box 0.5 m'),
    'trolley': Shape('box', (1.0, 0.8, 1.0), 0.25, 'trolley 1 m'),
}


def max_range(reflectivity, r10=200.0, cap=230.0):
    """Lidar link budget for extended targets: range ~ sqrt(reflectivity); Pandar128E3X spec: 200 m at 10 %,
    ranging accuracy +-2 cm, 0.1 deg horizontal / 0.125 deg vertical (channels 26-90) resolution at 10 Hz."""
    return min(cap, r10 * np.sqrt(reflectivity / 0.10))


def channel_reach(reflectivity, ring_index, n_rings):
    """Per-ray reach (70 % detection) and hard cap from the Pandar128E3X channel table (user manual, Appendix A):
    range at 10 % reflectivity per channel scaled with sqrt(reflectivity), capped at the instrumented range (+5 %: the
    recordings contain returns up to 209 m on 200 m channels).  Falls back to the uniform model for other sensors."""
    if n_rings == 128:
        try:
            from .pandar import model
            m = model()
            reach = m.range_10pct[ring_index] * np.sqrt(reflectivity / 0.10)
            cap = m.max_instrumented[ring_index] * 1.05
            return np.minimum(reach, cap), cap
        except FileNotFoundError:
            pass
    r = max_range(reflectivity)
    return np.full(len(ring_index), r), np.full(len(ring_index), np.inf)


def ray_hits(dirs, centre, shape: Shape):
    """dirs (M,3) unit rays from origin; centre = bottom-centre of object (forward frame). Returns t (M,), inf=no hit."""
    if shape.kind == 'box':
        L, W, H = shape.size
        lo = np.array([centre[0] - L / 2, centre[1] - W / 2, centre[2]])
        hi = np.array([centre[0] + L / 2, centre[1] + W / 2, centre[2] + H])
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / dirs
            t1 = lo * inv
            t2 = hi * inv
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmax >= tmin) & (tmax > 0)
        return np.where(hit, np.maximum(tmin, 0), np.inf)
    rad, H = shape.size
    dx, dy = dirs[:, 0], dirs[:, 1]
    a = dx * dx + dy * dy
    b = -2 * (dx * centre[0] + dy * centre[1])
    c = centre[0] ** 2 + centre[1] ** 2 - rad ** 2
    disc = b * b - 4 * a * c
    with np.errstate(invalid='ignore', divide='ignore'):
        t = (-b - np.sqrt(disc)) / (2 * a)
    z = t * dirs[:, 2]
    hit = (disc >= 0) & (t > 0) & (z >= centre[2]) & (z <= centre[2] + H)
    return np.where(hit, t, np.inf)


def inject(rng_img, dirs, centre, shape: Shape, rng: np.random.Generator, noise=0.02):
    """Modify an organised range image (W,R) in place-copy. dirs (W,R,3) unit vectors in the forward frame.
    Returns (new range image, number of object returns)."""
    out = rng_img.copy()
    dist = np.hypot(centre[0], centre[1])
    ext = max(shape.size) + 0.5
    # angular pre-selection
    az = np.arctan2(dirs[..., 1], dirs[..., 0])
    az0 = np.arctan2(centre[1], centre[0])
    sel = np.abs(az - az0) < np.arctan2(ext, max(dist - ext, 0.5))
    wi, ri = np.nonzero(sel)
    if len(wi) == 0:
        return out, 0
    t = ray_hits(dirs[wi, ri], centre, shape)
    rmax, rcap = channel_reach(shape.reflectivity, ri, dirs.shape[1])
    # return probability ramps down over the last 30 % of the reachable range; nothing beyond the instrumented range
    p_ret = np.clip((rmax - t) / (0.3 * rmax), 0.0, 1.0) * (t <= rcap)
    got = np.isfinite(t) & (rng.random(len(t)) < p_ret)
    cur = out[wi, ri]
    closer = got & ((cur < 0.5) | (t < cur))
    out[wi[closer], ri[closer]] = t[closer] + rng.normal(0, noise, closer.sum())
    # rays that hit the object but produced no return are occluded (no echo from behind either)
    blocked = np.isfinite(t) & ~got & ((cur < 0.5) | (t < cur))
    out[wi[blocked], ri[blocked]] = 0.0
    return out, int(closer.sum())
