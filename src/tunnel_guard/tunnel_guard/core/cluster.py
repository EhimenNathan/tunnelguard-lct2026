"""Clustering of envelope points on a range-adaptive track-aligned grid.

Along-track cells grow with distance (lidar sampling becomes sparser), lateral cells are fixed.  Connected
components (8-neighbourhood) of occupied cells form clusters; per-cluster statistics feed the tracker.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as cc_label


@dataclass
class ClusterConfig:
    s_max: float = 300.0
    s_cell_min: float = 0.4
    s_cell_rel: float = 0.012        # along-track cell = max(s_cell_min, s_cell_rel * s)
    l_range: float = 2.2
    l_cell: float = 0.25
    min_points: int = 3
    h_gap_min: float = 0.6            # minimum vertical gap that separates two objects [m]
    h_gap_beams: float = 2.5          # ... or this many vertical beam spacings at the object's range
    beam_vertical_rad: float = 0.00218  # Pandar128 fine vertical resolution 0.125 deg


def _s_edges(cfg):
    edges = [0.0]
    while edges[-1] < cfg.s_max:
        edges.append(edges[-1] + max(cfg.s_cell_min, cfg.s_cell_rel * edges[-1]))
    return np.array(edges)


class Clusterer:
    def __init__(self, cfg: ClusterConfig = None):
        self.cfg = cfg or ClusterConfig()
        self.s_edges = _s_edges(self.cfg)
        self.nl = int(round(2 * self.cfg.l_range / self.cfg.l_cell))

    def run(self, s, l, h, zone, ring=None, intensity=None, xyz=None):
        """Returns list of dict clusters.  All inputs are per-point arrays of envelope (zone>0) points."""
        cfg = self.cfg
        if len(s) == 0:
            return []
        si = np.searchsorted(self.s_edges, s, side='right') - 1
        li = ((l + cfg.l_range) / cfg.l_cell).astype(int)
        ok = (si >= 0) & (si < len(self.s_edges) - 1) & (li >= 0) & (li < self.nl)
        idx = np.nonzero(ok)[0]
        grid = np.zeros((len(self.s_edges) - 1, self.nl), bool)
        grid[si[idx], li[idx]] = True
        lab, n = cc_label(grid, structure=np.ones((3, 3), bool))
        if n == 0:
            return []
        pl = lab[si[idx], li[idx]]
        order = np.argsort(pl, kind='stable')
        pl = pl[order]
        pidx = idx[order]
        starts = np.r_[0, np.nonzero(np.diff(pl))[0] + 1, len(pl)]
        out = []
        groups = []
        for a, b in zip(starts[:-1], starts[1:]):
            m = pidx[a:b]
            if len(m) < cfg.min_points:
                continue
            # 3D: split at vertical gaps larger than the lidar's vertical sampling at that range
            order_h = m[np.argsort(h[m])]
            gap = max(cfg.h_gap_min, cfg.h_gap_beams * cfg.beam_vertical_rad * float(s[m].mean()))
            cuts = np.nonzero(np.diff(h[order_h]) > gap)[0] + 1
            for part in np.split(order_h, cuts):
                if len(part) >= cfg.min_points:
                    groups.append(part)
        for m in groups:
            c = dict(
                n=len(m),
                s_min=float(s[m].min()), s_max=float(s[m].max()),
                l_min=float(l[m].min()), l_max=float(l[m].max()), l_mean=float(l[m].mean()),
                h_min=float(h[m].min()), h_max=float(h[m].max()),
                in_gauge=int((zone[m] == 2).sum()),
                rings=int(len(np.unique(ring[m]))) if ring is not None else 0,
                intensity=float(intensity[m].mean()) if intensity is not None else 0.0,
                idx=m,
            )
            if xyz is not None:
                p = xyz[m]
                c['centroid'] = p.mean(axis=0)
                c['nearest'] = p[np.argmin(np.linalg.norm(p, axis=1))]
                c['range'] = float(np.linalg.norm(c['nearest']))
            out.append(c)
        return out
