"""PointCloud2 decoding (no ROS dependency) and sensor-to-forward-frame axis handling."""
import numpy as np

_DT = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
_AXES = {'x': (0, 1), '-x': (0, -1), 'y': (1, 1), '-y': (1, -1)}


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
        far = xyz[np.einsum('ij,ij->i', xyz[:, :2], xyz[:, :2]) > self.min_range ** 2]
        for k, (i, sgn) in _AXES.items():
            j = 1 - i
            self.votes[k] += int(np.count_nonzero((sgn * far[:, i] > 0) & (np.abs(far[:, j]) < 0.3 * np.abs(far[:, i]))))
        self.seen += 1
        if self.seen >= self.frames:
            self.axis = max(self.votes, key=self.votes.get)
        return max(self.votes, key=self.votes.get)
