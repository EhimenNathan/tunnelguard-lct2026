"""Analytic metro-tunnel point cloud generator for unit tests (known ground-truth geometry)."""
import numpy as np

GAUGE_CC = 1.595


def centreline(s, radius=None, offset=0.0):
    if radius is None:
        return np.full_like(s, offset, dtype=float)
    return offset + s * s / (2.0 * radius)


def make_tunnel(radius=None, lidar_height=1.15, grade=0.0, s_max=220.0, tunnel_r=2.6, centre_h=1.9, offset=0.0,
                obstacles=(), seed=0):
    """Returns (N,3) forward-frame points (X forward, Y left, Z up) of a round tunnel with rails.
    obstacles: iterable of (s, l, width, length, height) boxes standing on the rail plane."""
    rng = np.random.default_rng(seed)
    pts = []
    s = 2.0
    ang = np.arange(np.radians(-70), np.radians(250), 0.03)
    while s < s_max:
        ds = max(0.08, 0.004 * s)
        l_floor = np.arange(-2.2, 2.2, 0.02 + 0.0004 * s)
        h_floor = np.full_like(l_floor, -0.17)
        for c in (-GAUGE_CC / 2, GAUGE_CC / 2):
            h_floor[np.abs(l_floor - c) < 0.036] = 0.0            # rail heads
        wl = tunnel_r * np.cos(ang)
        wh = centre_h + tunnel_r * np.sin(ang)
        keep = wh > -0.1
        l = np.r_[l_floor, wl[keep]]
        h = np.r_[h_floor, wh[keep]]
        for (os_, ol, ow, olen, oh) in obstacles:
            if os_ <= s < os_ + olen:
                lo = np.arange(ol - ow / 2, ol + ow / 2, 0.03)
                ho = np.arange(0.0, oh, 0.03)
                L, Hh = np.meshgrid(lo, ho)
                l = np.r_[l, L.ravel()]
                h = np.r_[h, Hh.ravel()]
        y = centreline(np.array([s]), radius, offset)[0] + l
        z = -lidar_height + grade * s + h
        pts.append(np.stack([np.full_like(l, s), y, z], 1))
        s += ds
    p = np.concatenate(pts)
    p += rng.normal(0, 0.005, p.shape)
    return p.astype(np.float32)
