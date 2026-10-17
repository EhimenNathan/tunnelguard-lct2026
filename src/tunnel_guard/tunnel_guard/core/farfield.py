"""Far-field obstacle detection by temporal evidence accumulation (track-before-detect).

Beyond the range where a single frame can confirm an object, a person returns only a handful of points per frame
(Falcon Prime at 300 m: 2-5 points; Pandar128 at 160 m: ~10).  The tunnel core ahead, however, is almost empty: in the
finals' Falcon recording the conservative envelope core beyond 120 m receives 0.1-0.35 stray returns per frame in total,
spread over 200 m.  A physical object returns from the same place frame after frame; noise, dust and aerosol do not.

Evidence is kept in 1 m along-track cells for the last `window` frames (fixed arrays), shifted by the train's travel
(lidar odometry) so that anything standing still in the tunnel stays in its cell; a slowly walking person drifts by a
few centimetres per frame and is caught by the +-1 cell neighbourhood.  A cell (with neighbours) hit in at least
`min_frames` of the last `window` frames, with enough points and with floor support, is a far obstacle: CAUTION at
first, STOP once it has persisted for `stop_frames` frames.
"""
from dataclasses import dataclass

import numpy as np

from .gauge import IN_GAUGE, NEAR_GAUGE


@dataclass
class FarFieldConfig:
    enabled: bool = True
    s_min: float = 60.0             # never below this distance (the frame-by-frame pipeline owns the near field)
    s_max: float = 520.0            # evidence grid length [m]
    half_width: float = 1.0         # lateral core |l| [m] (envelope half width 1.1-1.35 m minus margin)
    centre_half_width: float = 0.9  # an object's mean lateral offset must be inside this
    h_lo: float = 0.3               # core height band above the rail plane [m]
    h_hi: float = 2.6
    cell: float = 1.0               # along-track cell [m]
    window: int = 15                # frames of evidence
    min_frames: int = 6             # frames with hits (cell +-1) for a detection
    min_points: int = 8             # points in the window
    stop_frames: int = 9            # frames with hits for STOP
    floor_support: float = 1.1      # lowest return of the object must be below this height [m]
    sigma_l_max: float = 0.6        # geometry must be at least this certain laterally at the object


class FarField:
    def __init__(self, cfg: FarFieldConfig = None):
        self.cfg = cfg or FarFieldConfig()
        c = self.cfg
        self.nc = int(np.ceil(c.s_max / c.cell)) + 2
        shape = (c.window, self.nc)
        self.cnt = np.zeros(shape, np.int32)
        self.hmin = np.full(shape, 9.0, np.float32)
        self.hmax = np.full(shape, -9.0, np.float32)
        self.lsum = np.zeros(shape, np.float32)
        self.ssum = np.zeros(shape, np.float32)
        self.scnt = np.zeros(shape, np.int32)       # scatter (dust/aerosol) returns per cell
        self.k = 0                    # frames seen
        self.offset = 0.0
        self.ids = np.zeros(self.nc, np.int64)
        self.next_id = 100000

    def reset(self):
        self.__init__(self.cfg)

    def _shift(self, n):
        """Objects move n cells towards the train."""
        for a, fill in ((self.cnt, 0), (self.hmin, 9.0), (self.hmax, -9.0), (self.lsum, 0.0), (self.ssum, 0.0),
                        (self.scnt, 0)):
            a[:, :-n] = a[:, n:]
            a[:, -n:] = fill
        self.ssum[:, :-n] -= n * self.cfg.cell * (self.cnt[:, :-n] > 0) * self.cnt[:, :-n]
        self.ids[:-n] = self.ids[n:]
        self.ids[-n:] = 0

    def update(self, p, g, s_from, s_to, travel=0.0, exclude=None):
        """p: forward-frame points; g: TrackGeometry; evaluate s in [s_from, s_to]. Returns far obstacle dicts."""
        cfg = self.cfg
        if not cfg.enabled or g is None or not g.ok:
            self.reset()
            return []
        self.offset += travel
        n = int(np.floor(self.offset / cfg.cell))
        if 0 < n < self.nc:
            self.offset -= n * cfg.cell
            self._shift(n)
        s_from = max(s_from, cfg.s_min)
        s_to = min(s_to, cfg.s_max - 2 * cfg.cell)
        row = self.k % cfg.window
        for a, fill in ((self.cnt, 0), (self.hmin, 9.0), (self.hmax, -9.0), (self.lsum, 0.0), (self.ssum, 0.0),
                        (self.scnt, 0)):
            a[row] = fill
        sel = np.flatnonzero((p[:, 0] > s_from - 2) & (p[:, 0] < s_to + 2))
        cells_now = np.zeros(0, np.int64)
        if len(sel):
            s, l, h = g.to_track(p[sel])
            core = (s >= s_from) & (s <= s_to) & (np.abs(l) < cfg.half_width) & (h > cfg.h_lo) & (h < cfg.h_hi)
            if core.any():
                sel, s, l, h = sel[core], s[core], l[core], h[core]
                if exclude is not None:
                    # scatter returns (dust, aerosol) never count as evidence, but are counted to recognise a cloud
                    sc = exclude[sel]
                    if sc.any():
                        np.add.at(self.scnt[row], np.floor(s[sc] / cfg.cell).astype(np.int64), 1)
                    sel, s, l, h = sel[~sc], s[~sc], l[~sc], h[~sc]
                cells_now = np.floor(s / cfg.cell).astype(np.int64)
                np.add.at(self.cnt[row], cells_now, 1)
                np.minimum.at(self.hmin[row], cells_now, h.astype(np.float32))
                np.maximum.at(self.hmax[row], cells_now, h.astype(np.float32))
                np.add.at(self.lsum[row], cells_now, l.astype(np.float32))
                np.add.at(self.ssum[row], cells_now, s.astype(np.float32))
        self.k += 1
        if self.k < cfg.min_frames:
            return []
        # cell +-1 neighbourhood over the window
        def nb(a, op):
            out = a.copy()
            if op == 'sum':
                out[:, 1:] += a[:, :-1]; out[:, :-1] += a[:, 1:]
            elif op == 'min':
                np.minimum(out[:, 1:], a[:, :-1], out=out[:, 1:]); np.minimum(out[:, :-1], a[:, 1:], out=out[:, :-1])
            else:
                np.maximum(out[:, 1:], a[:, :-1], out=out[:, 1:]); np.maximum(out[:, :-1], a[:, 1:], out=out[:, :-1])
            return out
        c3 = nb(self.cnt, 'sum')
        fh = (c3 > 0).sum(0)
        npts = c3.sum(0)
        cand = np.flatnonzero((fh >= cfg.min_frames) & (npts >= cfg.min_points))
        if not len(cand):
            self.ids[:] = 0
            return []
        sc3 = nb(self.scnt, 'sum').sum(0)
        hmin = nb(self.hmin, 'min').min(0)
        hmax = nb(self.hmax, 'max').max(0)
        lsum = nb(self.lsum, 'sum').sum(0)
        ssum = nb(self.ssum, 'sum').sum(0)
        out, used = [], np.zeros(self.nc, bool)
        order = cand[np.lexsort((-npts[cand], -fh[cand]))]
        keep_ids = np.zeros(self.nc, np.int64)
        for c in order:
            if used[max(c - 1, 0):c + 2].any():
                continue
            lmean = lsum[c] / max(npts[c], 1)
            smean = ssum[c] / max(npts[c], 1)
            if hmin[c] > cfg.floor_support or abs(lmean) > cfg.centre_half_width:
                continue
            if sc3[c] >= 0.3 * (sc3[c] + npts[c]):
                continue                                  # mostly scatter returns around it: a dust / aerosol cloud
            sl, _ = g.sigma_at(np.array([smean]))
            if sl[0] > cfg.sigma_l_max:
                continue
            used[max(c - 1, 0):c + 2] = True
            prev = self.ids[max(c - 1, 0):c + 2]
            oid = int(prev[prev > 0][0]) if (prev > 0).any() else self.next_id
            if oid == self.next_id:
                self.next_id += 1
            keep_ids[c] = oid
            idx = sel[np.abs(cells_now - c) <= 1] if len(cells_now) else np.zeros(0, np.int64)
            out.append(dict(id=oid, zone=IN_GAUGE if fh[c] >= cfg.stop_frames else NEAR_GAUGE, distance=float(smean),
                            lateral=float(lmean), height=float(hmax[c]), h_min=float(hmin[c]), n=int(npts[c]),
                            confidence=float(fh[c] / cfg.window), frames=int(fh[c]), idx=idx, far=True))
        self.ids = keep_ids
        return sorted(out, key=lambda o: o['distance'])
