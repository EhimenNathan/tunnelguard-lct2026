"""PointCloud2 decoding (no ROS dependency) and sensor-to-forward-frame axis handling."""
import numpy as np

_DT = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
_AXES = {'x': (0, 1), '-x': (0, -1), 'y': (1, 1), '-y': (1, -1), 'z': (2, 1), '-z': (2, -1)}


def cloud_to_arrays(fields, data, point_step, is_bigendian=False):
    """fields: iterable of (name, offset, datatype, count). Returns dict name -> 1D array (zero-copy view)."""
    names, fmts, offs = [], [], []
    for name, off, dt, cnt in fields:
        if cnt != 1 or dt not in _DT:
            continue
        names.append(name)
        fmts.append(('>' if is_bigendian else '<') + _DT[dt])
        offs.append(off)
    dtype = np.dtype({'names': names, 'formats': fmts, 'offsets': offs, 'itemsize': point_step})
    n = len(data) // point_step
    arr = np.frombuffer(data, dtype=dtype, count=n)
    return {k: arr[k] for k in names}


def forward_rotation(axis):
    """3x3 matrix mapping sensor xyz to forward frame (X forward, Y left, Z up), for a horizontal forward axis."""
    # Falcon native convention (manual section 1.4): x up, y right, z forward.
    # ROS drivers may already convert to x forward / z up; never apply twice.
    if axis in ('z', '-z', 'falcon'):
        sign = -1 if axis == '-z' else 1
        return np.array([[0., 0., sign], [0., -sign, 0.], [1., 0., 0.]])
    i, sgn = _AXES[axis]
    fwd = np.zeros(3)
    fwd[i] = sgn
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, fwd)
    return np.stack([fwd, left, up])


class AxisDetector:
    """Detects which horizontal sensor axis looks down the tunnel: the direction with most far returns."""

    def __init__(self, frames=5, min_range=40.0):
        self.frames = frames
        self.min_range = min_range
        self.votes = {k: 0 for k in _AXES}
        self.seen = 0
        self.axis = None

    def update(self, xyz):
        if self.axis is not None:
            return self.axis
        far = xyz[np.einsum('ij,ij->i', xyz, xyz) > self.min_range ** 2]
        for k, (i, sgn) in _AXES.items():
            j = 1 if i == 2 else 1 - i
            self.votes[k] += int(np.count_nonzero((sgn * far[:, i] > 0) & (np.abs(far[:, j]) < 0.3 * np.abs(far[:, i]))))
        self.seen += 1
        if self.seen >= self.frames:
            self.axis = max(self.votes, key=self.votes.get)
        return max(self.votes, key=self.votes.get)


class MountCalibrator:
    """Self-calibration of the lidar mounting: pitch and roll of the sensor relative to the track bed.

    The track bed (floor, sleepers, rails) is the lowest extended surface ahead of the train.  Over the first frames the
    lower envelope of the near field (5-40 m ahead, +-3 m lateral, per 1 m x 0.5 m cell) is fitted with a plane by
    iteratively reweighted least squares; the rotation that makes this plane level is applied to every later frame.
    A tilted or rolled lidar (e.g. the finals' Falcon is pitched ~2.4 deg up) then looks like a level one to the rest of
    the pipeline; track grade and cant are still measured by the geometry estimator relative to this frame.
    """

    def __init__(self, frames=8, max_tilt_deg=15.0):
        self.frames = frames
        self.max_tilt = np.radians(max_tilt_deg)
        self.cells = []
        self.R = None

    def update(self, p):
        """p: points in the forward frame (x ahead, y left, z up). Returns the 3x3 levelling rotation or None."""
        if self.R is not None:
            return self.R
        m = (p[:, 0] > 5) & (p[:, 0] < 40) & (np.abs(p[:, 1]) < 3.0)
        q = p[m]
        if len(q) > 200:
            key = np.floor(q[:, 0]).astype(np.int64) * 64 + np.floor((q[:, 1] + 3.0) / 0.5).astype(np.int64)
            o = np.argsort(key, kind='stable')
            ks, qs = key[o], q[o]
            starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
            ends = np.r_[starts[1:], len(ks)]
            for a, b in zip(starts, ends):
                if b - a >= 3:
                    zc = qs[a:b, 2]
                    i = a + int(np.argsort(zc)[max(0, (b - a) // 10)])
                    self.cells.append(qs[i])
        self.frames -= 1
        if self.frames > 0:
            return None
        c = np.array(self.cells, float) if self.cells else np.zeros((0, 3))
        self.R = np.eye(3)
        if len(c) >= 30:
            A = np.column_stack([np.ones(len(c)), c[:, 0], c[:, 1]])
            w = np.ones(len(c))
            for _ in range(8):
                coef = np.linalg.lstsq(A * w[:, None], c[:, 2] * w, rcond=None)[0]
                r = c[:, 2] - A @ coef
                s = 1.4826 * np.median(np.abs(r)) + 1e-3
                w = 1.0 / np.maximum(1.0, np.abs(r) / (1.5 * s))      # Huber weights
                w[r > 3 * s] = 0.0                                     # objects / walls above the bed do not count
            n = np.array([-coef[1], -coef[2], 1.0])
            n /= np.linalg.norm(n)
            ang = np.arccos(np.clip(n[2], -1, 1))
            if 1e-4 < ang < self.max_tilt:
                ax = np.cross(n, [0.0, 0.0, 1.0])
                ax /= np.linalg.norm(ax)
                K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                self.R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)    # Rodrigues: n -> z
        self.cells = []
        return self.R
