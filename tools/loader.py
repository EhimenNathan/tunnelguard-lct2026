import numpy as np, zlib, json
class BagCache:
    """Reader for the compact range-image cache. Points are returned in the lidar frame
    rotated so that X = forward (-y_lidar), Y = left (-x_lidar), Z = up."""
    def __init__(self, name, root='cache'):
        m = np.load(f'{root}/{name}_meta.npz'); self.meta = json.load(open(f'{root}/{name}_meta.json'))
        self.W = self.meta['W']; self.index = m['index']; self.stamps = m['stamps']; self.col_dt = m['col_dt']
        d = self._complete_dirs(m['dirs'].astype(np.float64), m['dircnt'] > 0).astype(np.float32)
        self.dirs = np.stack([-d[..., 1], -d[..., 0], d[..., 2]], -1)
        self.n = len(self.index); self.f = open(f'{root}/{name}.bin', 'rb'); self.name = name
        self.t = (self.index[:, 2] - self.index[0, 2]) / 1e9
    @staticmethod
    def _complete_dirs(d, v):
        """Rays that never returned have no calibrated direction; the Hesai scan is separable
        (azimuth = column azimuth + per-channel offset, elevation = per-channel), fit exactly (<0.001 deg) and fill."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            el = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
            az = np.degrees(np.arctan2(d[..., 0], -d[..., 1]))
            elr = np.nanmedian(np.where(v, el, np.nan), axis=0)
            off = np.zeros(d.shape[1])
            for _ in range(4):
                azc = np.nanmedian(np.where(v, az - off[None, :], np.nan), axis=1)
                off = np.nanmedian(np.where(v, az - azc[:, None], np.nan), axis=0)
        cols = np.arange(d.shape[0]); ok = np.isfinite(azc)
        azc = np.interp(cols, cols[ok], azc[ok])
        off = np.nan_to_num(off); elr = np.interp(np.arange(d.shape[1]), np.nonzero(np.isfinite(elr))[0], elr[np.isfinite(elr)])
        A = np.radians(azc[:, None] + off[None, :]); E = np.radians(np.broadcast_to(elr[None, :], A.shape))
        full = np.stack([np.cos(E) * np.sin(A), -np.cos(E) * np.cos(A), np.sin(E)], -1)
        return np.where(v[..., None], d, full)

    def frame(self, i):
        off, ln = self.index[i][:2]; self.f.seek(off); raw = zlib.decompress(self.f.read(ln))
        n = self.W * 128
        rng = np.frombuffer(raw[:2 * n], np.uint16).reshape(self.W, 128).astype(np.float32) / 100
        inten = (np.frombuffer(raw[2 * n:], np.uint8).reshape(self.W, 128) if len(raw) > 2 * n
                 else np.zeros((self.W, 128), np.uint8))            # range-only caches (dataset2)
        return rng, inten
    def points(self, i, rmin=0.5):
        rng, inten = self.frame(i); v = rng > rmin
        return self.dirs[v] * rng[v][:, None], inten[v], v
