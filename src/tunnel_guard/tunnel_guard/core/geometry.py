"""Track geometry estimation for a metro tunnel.

Output is a TrackGeometry: along the forward axis x (metres; sensor frame rotated so X is forward, Y left, Z up)
it gives the running-track centreline y_c(x), the rail-head height z_r(x), the cross-level (roll) slope and the
range over which the geometry is supported by measurements.

Pipeline
1. Rails (near field, ~3-40 m).  A BEV max-height map, a lateral white top-hat -> rail-head ridges, gauge-pair
   evidence and a dynamic-programming search for the globally best smooth centre path.  Scan-pattern independent;
   gives centreline, rail height and roll at centimetre level.
2. Cross-section template.  A tunnel keeps its cross-section along the track, so the near-field structure in track
   coordinates (lateral l, height h) - accumulated with temporal memory - is a model of "the normal tunnel".
   Its truncated distance fields are the alignment targets; the corridor core carries an extra free-space
   penalty (trains drive through it, so observed structure cannot belong there).
3. Dense direct profile alignment.  The centreline and rail-height profiles (1 m grid, up to the lidar horizon)
   are optimised by Gauss-Newton so that every lidar point falls onto the template surfaces.  Each point constrains
   the profile exactly at its own distance (oblique walls in curves included); the Jacobian is banded, so each
   iteration is one banded linear solve.  Coarse-to-fine truncated distances give a wide basin and robustness:
   points far from any template surface exert no pull.
4. Priors: rail measurements, a clothoid-like smoothness prior (penalising curvature change - railway transition
   curves - plus a small curvature term) and the previous frame's profile.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solveh_banded
from scipy.ndimage import distance_transform_edt, gaussian_filter, grey_opening, maximum_filter1d

GAUGE_CC = 1.595          # rail-head centre-to-centre distance for 1520 mm gauge [m]


@dataclass
class GeometryConfig:
    x_max: float = 300.0
    grid_step: float = 1.0
    rail_x_max: float = 45.0
    rail_dy: float = 0.03
    rail_min_evidence: float = 0.05
    rail_sigma: float = 0.03
    rail_trust_x: float = 15.0       # rail detections beyond this are gated by temporal consistency
    template_x: tuple = (4.0, 30.0)
    template_l: tuple = (-8.0, 8.0)
    template_h: tuple = (-2.5, 7.5)
    template_res: float = 0.05
    template_memory: float = 0.85
    template_every: int = 2          # template refresh period [frames]
    align_x_min: float = 12.0
    align_points: int = 12000
    align_sigma: float = 0.05        # nominal point-to-surface noise [m]
    align_node_points: float = 30.0  # points per 1 m node counted at full weight (density normalisation)
    align_iters: tuple = ((1.0, 2), (0.5, 2), (0.25, 2))   # (distance truncation [m], Gauss-Newton iterations)
    lateral_max_trunc: float = 0.75  # lateral profile is refined only on stages finer than this
    align_min_info: float = 800.0    # lateral information per node to count as measured (~2 wall points)
    vertical_ref_max_h: float = 0.8  # only structure below this height above the rail plane constrains the grade
    vertical_ref_min_l: float = 1.2  # ... outside the envelope footprint (obstacles cannot pull the rail plane)
    vertical_bed_h: float = 0.3      # ... or at track-bed level anywhere (only the feet of an obstacle are this low)
    core_half_width: float = 1.1
    core_h: tuple = (0.3, 3.2)
    core_penalty: float = 0.0        # DP free-space term. 0: observed points must never steer the track away (obstacles!)
    dp_range: float = 12.0           # lateral hypotheses +- [m] around the rail-based reference
    dp_step: float = 0.25
    dp_trunc: float = 0.6
    dp_max_heading_change: float = 0.06   # max lateral drift per metre between slabs
    dp_smooth: float = 50.0          # transition cost per (lateral jump^2 / slab spacing)
    dp_points: int = 200
    dp_local_range: float = 3.0      # search +- around the temporal prior on ordinary frames
    dp_full_every: int = 10          # full-range global search every N frames (recovery)
    smooth_lat: tuple = (1.0e3, 1.0e8)   # (curvature, curvature-change) penalties, lateral
    smooth_vert: tuple = (1.0e3, 1.0e8)  # (curvature, curvature-change) penalties, vertical
    temporal_weight: float = 1.0     # scale of the Kalman-style temporal prior
    temporal_process: tuple = (0.05, 0.03)  # per-frame process noise of the profile (lateral, vertical) [m]
    temporal_max_weight: float = 400.0
    roll_alpha: float = 0.2
    sample_seed: int = 0             # far-field subsampling seed (combined with the frame content)
    sigma_sys_l: tuple = (0.04, 0.10)   # systematic lateral error: (at 0 m, per 100 m)
    sigma_sys_v: tuple = (0.03, 0.05)   # systematic vertical error: (at 0 m, per 100 m)
    sigma_unmeasured_per_m: float = 0.02  # error growth per metre beyond the measured range
    trusted_sigma: float = 0.45      # detection horizon: lateral 1-sigma must stay below this


@dataclass
class TrackGeometry:
    xs: np.ndarray
    yc: np.ndarray
    zr: np.ndarray
    roll: float = 0.0              # dz/dl cross-level slope
    rail_range: float = 0.0        # farthest x with rail evidence
    valid_range: float = 0.0       # farthest x supported by rails or dense alignment
    ok: bool = False
    meas: dict = field(default_factory=dict)
    sigma_l: np.ndarray = None     # 1-sigma lateral centreline uncertainty per grid node [m]
    sigma_v: np.ndarray = None     # 1-sigma rail-height uncertainty per grid node [m]
    rand_l: np.ndarray = None      # random (temporally fusable) part of sigma_l
    rand_v: np.ndarray = None
    trusted_range: float = 0.0     # farthest x with lateral uncertainty below trusted_sigma

    def __post_init__(self):
        self._head = np.gradient(self.yc, self.xs)

    def _lerp(self, arr, x):
        step = self.xs[1] - self.xs[0]
        u = np.clip(np.asarray(x, dtype=np.float64) / step, 0.0, len(self.xs) - 1.000001)
        i = u.astype(np.int64)
        f = u - i
        return arr[i] * (1.0 - f) + arr[i + 1] * f

    def centre(self, x):
        return self._lerp(self.yc, x)

    def sigma_at(self, x):
        if self.sigma_l is None:
            z = np.zeros_like(np.asarray(x, dtype=float))
            return z, z
        return self._lerp(self.sigma_l, x), self._lerp(self.sigma_v, x)

    def rail_z(self, x):
        return self._lerp(self.zr, x)

    def heading(self, x):
        return self._lerp(self._head, x)

    def curvature(self, x=10.0):
        return float(self._lerp(np.gradient(self._head, self.xs), x))

    def sight_distance(self, clearance=1.8, x_max=None):
        """Farthest along-track distance whose centreline point is visible from the sensor through a curved tunnel:
        the straight line of sight must stay within `clearance` of the centreline everywhere before it."""
        n = len(self.xs) if x_max is None else int(min(len(self.xs), x_max / (self.xs[1] - self.xs[0]) + 1))
        x = self.xs[:n:2]
        y = self.yc[:n:2]
        z = self.zr[:n:2]
        u = x[None, :] / np.maximum(x[:, None], 1e-6)
        dev_y = y[None, :] - (y[0] + (y[:, None] - y[0]) * u)
        dev_z = z[None, :] - (z[0] + (z[:, None] - z[0]) * u)
        before = x[None, :] <= x[:, None]
        worst = np.max(np.where(before, np.maximum(np.abs(dev_y), np.abs(dev_z)), 0.0), axis=1)
        blocked = np.nonzero(worst > clearance)[0]
        return float(x[blocked[0]]) if len(blocked) else float(x[-1])

    def to_track(self, pts):
        """(N,3) forward-frame points -> (s, l, h): along-track, lateral (+left), height above the rail plane."""
        x = pts[:, 0]
        step = self.xs[1] - self.xs[0]
        u = np.clip(x / step, 0.0, len(self.xs) - 1.000001)
        i = u.astype(np.int64)
        f = u - i
        g = 1.0 - f
        yc = self.yc[i] * g + self.yc[i + 1] * f
        zr = self.zr[i] * g + self.zr[i + 1] * f
        head = self._head[i] * g + self._head[i + 1] * f
        l = (pts[:, 1] - yc) / np.sqrt(1.0 + head * head)
        h = pts[:, 2] - zr - self.roll * l
        return x, l, h


# ----------------------------------------------------------------------------------------------- rails
def _ridge_map(pts, x_edges, z_ref, dy, y_lim=4.0):
    nb = int(round(2 * y_lim / dy))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    s = (x >= x_edges[0]) & (x < x_edges[-1]) & (np.abs(y) < y_lim) & (z > z_ref - 0.8) & (z < z_ref + 0.7)
    x, y, z = x[s], y[s], z[s]
    nr = len(x_edges) - 1
    key = (np.searchsorted(x_edges, x, side='right') - 1) * nb + ((y + y_lim) / dy).astype(np.int64)
    H = np.full(nr * nb, -np.inf)
    if len(key):
        order = np.argsort(key)
        ks = key[order]
        starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
        H[ks[starts]] = np.maximum.reduceat(z[order], starts)
    H = H.reshape(nr, nb)
    occ = np.isfinite(H)
    # fill empty bins by linear interpolation along each row (vectorised forward/backward fill + blend)
    cols = np.arange(nb)[None, :].repeat(nr, 0)
    fwd = np.where(occ, cols, -1)
    np.maximum.accumulate(fwd, axis=1, out=fwd)
    bwd = np.minimum.accumulate(np.where(occ, cols, nb)[:, ::-1], axis=1)[:, ::-1]
    rows_i = np.arange(nr)[:, None]
    hf = np.where(fwd >= 0, H[rows_i, np.clip(fwd, 0, nb - 1)], np.nan)
    hb = np.where(bwd < nb, H[rows_i, np.clip(bwd, 0, nb - 1)], np.nan)
    wgt = np.clip((cols - fwd) / np.maximum(bwd - fwd, 1), 0, 1)
    Hf = np.where(np.isnan(hf), hb, np.where(np.isnan(hb), hf, hf + (hb - hf) * wgt))
    rich = occ.sum(axis=1) >= 15
    Hf = np.where(rich[:, None] & np.isfinite(Hf), Hf, -50.0)
    k = int(round(0.36 / dy)) | 1
    w = int(round(0.09 / dy))
    th = Hf - grey_opening(Hf, size=(1, k))
    side = np.maximum(np.roll(th, w, axis=1), np.roll(th, -w, axis=1))
    ok = occ & rich[:, None] & (th > 0.06) & (th < 0.30) & (side < 0.6 * th)
    return np.where(ok, np.minimum(th, 0.2), 0.0), H


try:
    from numba import njit

    @njit(cache=True)
    def _dp_kernel(S, cost0, pen, k, prior_c, ys, has_prior):
        nr, nb = S.shape
        back = np.zeros((nr, nb), np.int8)
        cost = cost0.copy()
        new = np.empty(nb)
        for r in range(1, nr):
            for i in range(nb):
                best = 1e18
                arg = 0
                for dd in range(-k, k + 1):
                    src = i + dd
                    if 0 <= src < nb:
                        c = cost[src] + pen[dd + k]
                        if c < best:
                            best = c
                            arg = dd
                v = best - S[r, i]
                if has_prior:
                    v += 0.008 * (ys[i] - prior_c[r]) ** 2
                new[i] = v
                back[r, i] = arg
            cost[:] = new
        return cost, back
    @njit(cache=True)
    def _coarse_cost_kernel(li, hi, qs, oc, F, trunc, nsl):
        noff = oc.shape[0]
        nl = F.shape[0]
        C = np.zeros((nsl, noff))
        cnt = np.zeros(nsl)
        for i in range(li.shape[0]):
            cnt[qs[i]] += 1.0
            for j in range(noff):
                L = li[i] - oc[j]
                if 0 <= L < nl:
                    C[qs[i], j] += F[L, hi[i]]
                else:
                    C[qs[i], j] += trunc
        for k in range(nsl):
            if cnt[k] > 0:
                for j in range(noff):
                    C[k, j] = C[k, j] / cnt[k] / trunc
        return C, cnt

    @njit(cache=True)
    def _coarse_dp_kernel(C, acc0, xc, max_hc, step, smooth):
        nsl, noff = C.shape
        back = np.zeros((nsl, noff), np.int32)
        acc = acc0.copy()
        new = np.empty(noff)
        for k in range(1, nsl):
            dx = xc[k] - xc[k - 1]
            m = max(1, int(np.ceil(max_hc * dx / step)))
            for i in range(noff):
                best = 1e18
                arg = i
                for d in range(-m, m + 1):
                    src = i - d
                    if 0 <= src < noff:
                        c = acc[src] + smooth * (d * step) ** 2 / dx
                    else:
                        c = 1e9 + smooth * (d * step) ** 2 / dx
                    if c < best:
                        best = c
                        arg = src
                new[i] = best + C[k, i]
                back[k, i] = arg
            acc[:] = new
        return acc, back
    @njit(cache=True)
    def _gn_kernel(px, py, pz, k, f, sw, y, z, head, roll, F, trunc, l0, h0, res, core_w, core_h0, core_h1,
                   vmax_h, vmin_l, vbed_h, n):
        nl, nh = F.shape
        jy0 = np.zeros(n); jy1 = np.zeros(n); ry = np.zeros(n)
        jz0 = np.zeros(n); jz1 = np.zeros(n); rz = np.zeros(n)
        for q in range(px.shape[0]):
            kk = k[q]; ff = f[q]; gg = 1.0 - ff
            hd = head[kk]
            cos = 1.0 / np.sqrt(1.0 + hd * hd)
            yc = y[kk] * gg + y[kk + 1] * ff
            zc = z[kk] * gg + z[kk + 1] * ff
            l = (py[q] - yc) * cos
            h = pz[q] - zc - roll * l
            u = (l - l0) / res - 0.5
            v = (h - h0) / res - 0.5
            if u >= 0 and u < nl - 1 and v >= 0 and v < nh - 1:
                i = int(u); j = int(v)
                a = u - i; b = v - j
                f00 = F[i, j]; f10 = F[i + 1, j]; f01 = F[i, j + 1]; f11 = F[i + 1, j + 1]
                val = f00 * (1 - a) * (1 - b) + f10 * a * (1 - b) + f01 * (1 - a) * b + f11 * a * b
                du = ((f10 - f00) * (1 - b) + (f11 - f01) * b) / res
                dv = ((f01 - f00) * (1 - a) + (f11 - f10) * a) / res
            else:
                val = trunc; du = 0.0; dv = 0.0
            if trunc >= 1.0:
                w = sw[q]
            else:
                t = min(val / trunc, 1.0)
                w = sw[q] * (1.0 - t * t)
            al = abs(l)
            if al < core_w and h > core_h0 - 0.3 and h < core_h1:
                w = 0.0
            r = w * val
            gy = w * (-cos) * (du - roll * dv)
            vm = h < vmax_h and (al > vmin_l or h < vbed_h)
            gz = w * (-dv) if vm else 0.0
            J0 = gy * gg; J1 = gy * ff
            jy0[kk] += J0 * J0; jy0[kk + 1] += J1 * J1; jy1[kk] += J0 * J1
            ry[kk] += J0 * r; ry[kk + 1] += J1 * r
            if vm:
                J0 = gz * gg; J1 = gz * ff
                jz0[kk] += J0 * J0; jz0[kk + 1] += J1 * J1; jz1[kk] += J0 * J1
                rz[kk] += J0 * r; rz[kk + 1] += J1 * r
        return jy0, jy1, ry, jz0, jz1, rz
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is optional
    HAVE_NUMBA = False


def warmup():
    """Compile the numba kernels once (called at node start so the first lidar frame is not delayed)."""
    if not HAVE_NUMBA:
        return False
    z1, i1 = np.zeros(4), np.zeros(4, np.int64)
    _dp_kernel(np.zeros((3, 4)), z1, np.zeros(3), 1, np.zeros(3), z1, True)
    _dp_kernel(np.zeros((3, 4)), z1, np.zeros(3), 1, np.zeros(3), z1, False)
    _coarse_cost_kernel(i1, i1, i1, i1, np.zeros((6, 6)), 0.6, 4)
    _coarse_dp_kernel(np.zeros((3, 4)), z1, np.arange(3.0), 0.06, 0.25, 50.0)
    f4 = np.zeros(4)
    _gn_kernel(f4, f4, f4, i1, f4, f4, np.zeros(8), np.zeros(8), np.zeros(8), 0.0, np.zeros((6, 6)), 0.5,
               -8.0, -2.5, 0.05, 1.1, 0.3, 3.2, 0.8, 1.2, 0.3, 8)
    return True


def _track_dp(R, dy, y_lim, prior_c=None, lam=40.0, max_step=0.10):
    """Globally optimal smooth centre path through gauge-pair evidence (Viterbi over x rows)."""
    nr, nb = R.shape
    half = int(round(GAUGE_CC / 2 / dy))
    Rm = maximum_filter1d(R, size=3, axis=1)
    S = np.zeros_like(R)
    S[:, half:nb - half] = np.minimum(Rm[:, :nb - 2 * half], Rm[:, 2 * half:])
    ys = -y_lim + (np.arange(nb) + 0.5) * dy
    k = max(1, int(round(max_step / dy)))
    d = np.arange(-k, k + 1)
    pen = lam * (d * dy) ** 2
    cost = (0.02 * ys ** 2 if prior_c is None else 0.2 * (ys - prior_c[0]) ** 2) - S[0]
    if HAVE_NUMBA:
        pc = np.zeros(nr) if prior_c is None else np.asarray(prior_c, np.float64)
        cost, back = _dp_kernel(S.astype(np.float64), cost.astype(np.float64), pen.astype(np.float64), k, pc,
                                ys.astype(np.float64), prior_c is not None)
        path = np.zeros(nr, np.int64)
        path[-1] = int(np.argmin(cost))
        for r in range(nr - 1, 0, -1):
            path[r - 1] = path[r] + back[r, path[r]]
        return ys[path], S[np.arange(nr), path], path
    back = np.zeros((nr, nb), np.int8)
    ar = np.arange(nb)
    for r in range(1, nr):
        cp = np.pad(cost, k, constant_values=np.inf)
        C = np.lib.stride_tricks.sliding_window_view(cp, 2 * k + 1) + pen[None, :]
        j = np.argmin(C, axis=1)
        cost = C[ar, j] - S[r]
        if prior_c is not None:
            cost = cost + 0.008 * (ys - prior_c[r]) ** 2
        back[r] = d[j]
    path = np.zeros(nr, int)
    path[-1] = int(np.argmin(cost))
    for r in range(nr - 1, 0, -1):
        path[r - 1] = path[r] + back[r, path[r]]
    return ys[path], S[np.arange(nr), path], path


def extract_rails(pts, cfg: GeometryConfig, prior=None):
    """Returns dict with row centres xc, centre yc, evidence ev, rail heights zl, zr (NaN when unseen)."""
    y_lim = 3.0
    if prior is not None and prior.ok:
        z_ref = float(prior.rail_z(10.0))
    else:
        near = pts[(pts[:, 0] > 3) & (pts[:, 0] < 15) & (np.abs(pts[:, 1]) < 1.5)]
        z_ref = float(np.percentile(near[:, 2], 40)) if len(near) > 50 else -1.3
    x_edges = np.r_[np.arange(2.5, 20, 0.5), np.arange(20, cfg.rail_x_max + 1e-6, 1.0)]
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    R, H = _ridge_map(pts, x_edges, z_ref, cfg.rail_dy, y_lim)
    pc = None if prior is None or not prior.ok else prior.centre(xc)
    yc, ev, path = _track_dp(R, cfg.rail_dy, y_lim, prior_c=pc)
    half = int(round(GAUGE_CC / 2 / cfg.rail_dy))
    rows = np.arange(len(xc))
    nb = H.shape[1]

    def rail_h(idx):
        best = np.full(len(rows), -np.inf)
        for o in (-1, 0, 1):
            best = np.maximum(best, H[rows, np.clip(idx + o, 0, nb - 1)])
        return np.where(np.isfinite(best), best, np.nan)

    return dict(xc=xc, yc=yc, ev=ev, zl=rail_h(path + half), zr=rail_h(path - half))


# ------------------------------------------------------------------------------- cross-section template
class CrossSectionTemplate:
    """Occupancy of the tunnel cross-section in track coordinates and its truncated distance fields."""

    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg
        self.nl = int(round((cfg.template_l[1] - cfg.template_l[0]) / cfg.template_res))
        self.nh = int(round((cfg.template_h[1] - cfg.template_h[0]) / cfg.template_res))
        self.occ = np.zeros((self.nl, self.nh), np.float32)
        self.fields = {}
        lc = cfg.template_l[0] + (np.arange(self.nl) + 0.5) * cfg.template_res
        hc = cfg.template_h[0] + (np.arange(self.nh) + 0.5) * cfg.template_res
        core = (np.abs(lc)[:, None] < cfg.core_half_width) & (hc[None, :] > cfg.core_h[0]) & (hc[None, :] < cfg.core_h[1])
        self.core = core

    def update(self, l, h):
        c = self.cfg
        il = np.floor((l - c.template_l[0]) / c.template_res).astype(np.int64)
        ih = np.floor((h - c.template_h[0]) / c.template_res).astype(np.int64)
        m = (il >= 0) & (il < self.nl) & (ih >= 0) & (ih < self.nh)
        cur = np.zeros_like(self.occ)
        cur[il[m], ih[m]] = 1.0
        self.occ = np.maximum(self.occ * c.template_memory, cur)
        d = distance_transform_edt(self.occ < 0.3) * c.template_res
        self.fields = {t: np.minimum(d, t).astype(np.float32) for t, _ in c.align_iters}
        self.fields[c.dp_trunc] = np.minimum(d, c.dp_trunc).astype(np.float32)
        # squared lateral component of the surface normal (information a point on that surface gives about dy)
        B = gaussian_filter(self.occ, 2.0)
        gl, gh = np.gradient(B)
        self.normal_l2 = (gl * gl / (gl * gl + gh * gh + 1e-12)).astype(np.float32)
        self.normal_h2 = (1.0 - self.normal_l2) * (gl * gl + gh * gh > 1e-10)

    def normals2(self, l, h):
        c = self.cfg
        i = np.clip(((l - c.template_l[0]) / c.template_res).astype(np.int64), 0, self.nl - 1)
        j = np.clip(((h - c.template_h[0]) / c.template_res).astype(np.int64), 0, self.nh - 1)
        return self.normal_l2[i, j], self.normal_h2[i, j]

    def sample(self, trunc, l, h):
        """Bilinear value and gradient (per metre) of the truncated distance field at points (l, h)."""
        c = self.cfg
        F = self.fields[trunc]
        u = (l - c.template_l[0]) / c.template_res - 0.5
        v = (h - c.template_h[0]) / c.template_res - 0.5
        inside = (u >= 0) & (u < self.nl - 1) & (v >= 0) & (v < self.nh - 1)
        i = np.clip(u.astype(np.int64), 0, self.nl - 2)
        j = np.clip(v.astype(np.int64), 0, self.nh - 2)
        a = np.clip(u - i, 0, 1)
        b = np.clip(v - j, 0, 1)
        f00, f10, f01, f11 = F[i, j], F[i + 1, j], F[i, j + 1], F[i + 1, j + 1]
        val = f00 * (1 - a) * (1 - b) + f10 * a * (1 - b) + f01 * (1 - a) * b + f11 * a * b
        du = ((f10 - f00) * (1 - b) + (f11 - f01) * b) / c.template_res
        dv = ((f01 - f00) * (1 - a) + (f11 - f10) * a) / c.template_res
        val = np.where(inside, val, trunc)
        du = np.where(inside, du, 0.0)
        dv = np.where(inside, dv, 0.0)
        return val, du, dv


# ----------------------------------------------------------------------------------------- estimator
_BANDS = {}


def _penalty_bands(n, lam2, lam3):
    """Upper banded form (u=3) of lam2*D2'D2 + lam3*D3'D3 (difference operators on a uniform grid)."""
    key = (n, lam2, lam3)
    if key not in _BANDS:
        H = np.zeros((n, n))
        for order, lam in ((2, lam2), (3, lam3)):
            if lam > 0:
                D = np.diff(np.eye(n), n=order, axis=0)
                H += lam * D.T @ D
        ab = np.zeros((4, n))
        for k in range(4):
            ab[3 - k, k:] = np.diagonal(H, k)
        _BANDS[key] = (ab, H)
    return _BANDS[key]


def _banded_mul(H, x):
    return H @ x


def _solve_profile(n, lam, data_idx, data_val, data_w, prior=None, prior_w=0.0, jtj=None, jtr=None, x0=None):
    """One (Gauss-Newton) step of
        E(c) = sum_i w_i (c[idx_i] - v_i)^2 + |pen^(1/2) c|^2 + prior_w |c - prior|^2 + sum_points r(c)^2
    around x0 with the point term linearised as (jtj tridiagonal: diag, off), jtr."""
    ab_pen, H_pen = _penalty_bands(n, lam[0], lam[1])
    ab = ab_pen.copy()
    x0 = np.zeros(n) if x0 is None else x0
    ab[3] += 1e-9 + prior_w
    np.add.at(ab[3], data_idx, data_w)
    grad = H_pen @ x0 + prior_w * (x0 - (prior if prior is not None else x0))
    np.add.at(grad, data_idx, data_w * (x0[data_idx] - data_val))
    if jtj is not None:
        ab[3] += jtj[0]
        ab[2, 1:] += jtj[1]
        grad += jtr
    return x0 - solveh_banded(ab, grad)


def _temporal_weights(prior, cfg):
    """Kalman-style temporal prior: weight of the previous profile per node = 1 / (its uncertainty^2 + process noise^2)."""
    if prior is None:
        return 0.0, 0.0
    if prior.sigma_l is None:
        return cfg.temporal_weight, cfg.temporal_weight
    wy = cfg.temporal_weight / (prior.sigma_l ** 2 + cfg.temporal_process[0] ** 2)
    wz = cfg.temporal_weight / (prior.sigma_v ** 2 + cfg.temporal_process[1] ** 2)
    return np.minimum(wy, cfg.temporal_max_weight), np.minimum(wz, cfg.temporal_max_weight)


class GeometryEstimator:
    def __init__(self, cfg: GeometryConfig = None):
        self.cfg = cfg or GeometryConfig()
        self.xs = np.arange(0.0, self.cfg.x_max + 1e-6, self.cfg.grid_step)
        self.template = CrossSectionTemplate(self.cfg)
        self.prev = None
        self.roll = 0.0
        self.frame_count = 0

    def reset(self):
        self.__init__(self.cfg)

    def _coarse_dp(self, far, y_ref, z_ref, x_start, search=None):
        """Globally optimal coarse lateral profile: DP over offset hypotheses per slab (template cost + free-space
        core occupancy), with bounded heading change between slabs.  Returns (slab centres, offsets, informative)."""
        cfg = self.cfg
        edges = [x_start]
        while edges[-1] < cfg.x_max:
            edges.append(edges[-1] + (3.0 if edges[-1] < 100 else 5.0))
        edges = np.array(edges)
        rng = cfg.dp_range if search is None else search
        offs = np.arange(-rng, rng + 1e-6, cfg.dp_step)
        nsl = len(edges) - 1
        sid = np.searchsorted(edges, far[:, 0], side='right') - 1
        keep_rng = (sid >= 0) & (sid < nsl)
        far, sid = far[keep_rng], sid[keep_rng]
        order = np.argsort(sid, kind='stable')
        fs, sid = far[order], sid[order]
        raw_cnt = np.bincount(sid, minlength=nsl)
        starts = np.r_[0, np.cumsum(raw_cnt)[:-1]]
        # per-slab uniform subsampling to at most dp_points (same selection as slab-by-slab processing)
        step = np.maximum(1, np.ceil(raw_cnt / cfg.dp_points)).astype(np.int64)
        pos = np.arange(len(sid)) - starts[sid]
        sel = (pos % step[sid] == 0) & (raw_cnt[sid] >= 15)
        q, qs = fs[sel], sid[sel]
        xc = 0.5 * (edges[:-1] + edges[1:])
        ref = TrackGeometry(self.xs, y_ref, z_ref, roll=self.roll)
        C = np.zeros((nsl, len(offs)))
        informative = np.zeros(nsl, bool)
        if len(q):
            _, l0, h = ref.to_track(q)
            T = self.template
            res = cfg.template_res
            li = np.round((l0 - cfg.template_l[0]) / res).astype(np.int64)
            hi = np.round((h - cfg.template_h[0]) / res).astype(np.int64)
            okh = (hi >= 0) & (hi < T.nh)
            li, hi, qs = li[okh], hi[okh], qs[okh]
            cnt = np.bincount(qs, minlength=nsl).astype(float)
            valid_slab = cnt >= 10
            F = T.fields[cfg.dp_trunc]
            oc = np.round(offs / res).astype(np.int64)
            if HAVE_NUMBA and cfg.core_penalty == 0:
                Ck, _ = _coarse_cost_kernel(li, hi, qs.astype(np.int64), oc, F.astype(np.float64), float(cfg.dp_trunc), nsl)
                C = np.where(valid_slab[:, None], Ck, 0.0)
            for j, o in enumerate(oc if not (HAVE_NUMBA and cfg.core_penalty == 0) else []):
                L = li - o
                inside = (L >= 0) & (L < T.nl)
                v = np.where(inside, F[np.clip(L, 0, T.nl - 1), hi], cfg.dp_trunc)
                cost = np.bincount(qs, v, minlength=nsl) / np.maximum(cnt, 1) / cfg.dp_trunc
                if cfg.core_penalty > 0:
                    cv = np.where(inside, T.core[np.clip(L, 0, T.nl - 1), hi], False)
                    cost = cost + cfg.core_penalty * np.bincount(qs, cv, minlength=nsl) / np.maximum(cnt, 1)
                C[:, j] = np.where(valid_slab, cost, 0.0)
            informative = valid_slab & ((np.median(C, axis=1) - C.min(axis=1)) > 0.15)
        # first-order DP with bounded drift and stiff quadratic continuity cost
        big = 1e9
        acc = C[0] + 0.02 * (offs / 0.5) ** 2
        back = np.zeros((nsl, len(offs)), np.int32)
        idx = np.arange(len(offs))
        if HAVE_NUMBA:
            acc, back = _coarse_dp_kernel(C, acc, xc, float(cfg.dp_max_heading_change), float(cfg.dp_step), float(cfg.dp_smooth))
        for k in range(1, nsl if not HAVE_NUMBA else 1):
            dx = xc[k] - xc[k - 1]
            m = max(1, int(np.ceil(cfg.dp_max_heading_change * dx / cfg.dp_step)))
            ds = np.arange(-m, m + 1)
            pad = np.r_[np.full(m, big), acc, np.full(m, big)]
            W = np.lib.stride_tricks.sliding_window_view(pad, 2 * m + 1)[:, ::-1]   # column j <-> d = ds[j]
            cand = W + (cfg.dp_smooth * (ds * cfg.dp_step) ** 2 / dx)[None, :]
            j = np.argmin(cand, axis=1)
            acc = cand[idx, j] + C[k]
            back[k] = idx - ds[j]
        path = np.zeros(nsl, np.int32)
        path[-1] = int(np.argmin(acc))
        for k in range(nsl - 1, 0, -1):
            path[k - 1] = back[k, path[k]]
        return xc, offs[path], informative

    def _align(self, pts, y, z, prior, rails_y, rails_z):
        """Dense Gauss-Newton alignment of the profiles to the template.  Returns y, z, lateral info per node."""
        cfg = self.cfg
        n = len(self.xs)
        step = cfg.grid_step
        x = pts[:, 0] / step
        k = np.clip(x.astype(np.int64), 0, n - 2)
        f = np.clip(x - k, 0.0, 1.0)
        cnt = np.bincount(k, minlength=n).astype(float)
        w_pt = (1.0 / cfg.align_sigma ** 2) / np.maximum(1.0, cnt[k] / cfg.align_node_points)
        sw = np.sqrt(w_pt)
        ri, ry, rw = rails_y
        zi, zv, zw = rails_z
        pwy, pwz = _temporal_weights(prior, cfg)
        info = np.zeros(n)
        for trunc, iters in cfg.align_iters:
            for _ in range(int(iters)):
                if HAVE_NUMBA:
                    head = np.gradient(y, self.xs)
                    jy0, jy1, ryv, jz0, jz1, rzv = _gn_kernel(
                        pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64), pts[:, 2].astype(np.float64), k, f, sw,
                        y, z, head, float(self.roll), self.template.fields[trunc].astype(np.float64), float(trunc),
                        float(cfg.template_l[0]), float(cfg.template_h[0]), float(cfg.template_res),
                        float(cfg.core_half_width), float(cfg.core_h[0]), float(cfg.core_h[1]),
                        float(cfg.vertical_ref_max_h), float(cfg.vertical_ref_min_l), float(cfg.vertical_bed_h), n)
                    if trunc < cfg.lateral_max_trunc:
                        y = _solve_profile(n, cfg.smooth_lat, ri, ry, rw, None if prior is None else prior.yc, pwy,
                                           (jy0, jy1[:n - 1]), ryv, y)
                    z = _solve_profile(n, cfg.smooth_vert, zi, zv, zw, None if prior is None else prior.zr, pwz,
                                       (jz0, jz1[:n - 1]), rzv, z)
                    continue
                head = np.gradient(y, self.xs)
                cos = 1.0 / np.sqrt(1.0 + head[k] ** 2)
                yc = y[k] * (1 - f) + y[k + 1] * f
                zc = z[k] * (1 - f) + z[k + 1] * f
                l = (pts[:, 1] - yc) * cos
                h = pts[:, 2] - zc - self.roll * l
                val, dl, dh = self.template.sample(trunc, l, h)
                # robust (Tukey) weights on the finer stages: only consistent surfaces steer the profile
                rw_pt = sw if trunc >= 1.0 else sw * (1.0 - np.minimum(val / trunc, 1.0) ** 2)
                # invariance: geometry is estimated only from structure outside the space where obstacles can be
                al = np.abs(l)
                rw_pt = rw_pt * ~((al < cfg.core_half_width) & (h > cfg.core_h[0] - 0.3) & (h < cfg.core_h[1]))
                r = rw_pt * val
                # d r / d y_node = w * (dl * dl/dy + dh * dh/dy), dl/dy = -cos*wgt, dh/dy = +roll*cos*wgt
                gy = rw_pt * (-cos) * (dl - self.roll * dh)
                # the rail plane is referenced to floor-level structure only (ceiling height varies with tunnel type)
                vmask = (h < cfg.vertical_ref_max_h) & ((al > cfg.vertical_ref_min_l) | (h < cfg.vertical_bed_h))
                gz = rw_pt * (-dh) * vmask
                J0y, J1y = gy * (1 - f), gy * f
                J0z, J1z = gz * (1 - f), gz * f
                jtj_y = (np.bincount(k, J0y * J0y, n) + np.bincount(k + 1, J1y * J1y, n), np.bincount(k, J0y * J1y, n)[:n - 1])
                jtr_y = np.bincount(k, J0y * r, n) + np.bincount(k + 1, J1y * r, n)
                jtj_z = (np.bincount(k, J0z * J0z, n) + np.bincount(k + 1, J1z * J1z, n), np.bincount(k, J0z * J1z, n)[:n - 1])
                rz = r * vmask
                jtr_z = np.bincount(k, J0z * rz, n) + np.bincount(k + 1, J1z * rz, n)
                if trunc < cfg.lateral_max_trunc:   # wide stages refine only the grade (lateral starts from the global DP)
                    y = _solve_profile(n, cfg.smooth_lat, ri, ry, rw, None if prior is None else prior.yc, pwy, jtj_y, jtr_y, y)
                z = _solve_profile(n, cfg.smooth_vert, zi, zv, zw, None if prior is None else prior.zr, pwz, jtj_z, jtr_z, z)
        # lateral information per node: on-surface points weighted by the lateral component of the surface normal
        head = np.gradient(y, self.xs)
        cos = 1.0 / np.sqrt(1.0 + head[k] ** 2)
        l = (pts[:, 1] - (y[k] * (1 - f) + y[k + 1] * f)) * cos
        h = pts[:, 2] - (z[k] * (1 - f) + z[k + 1] * f) - self.roll * l
        val, _, _ = self.template.sample(cfg.align_iters[-1][0], l, h)
        on = (val < 0.1).astype(float)
        nl2, nh2 = self.template.normals2(l, h)
        info = np.bincount(k, w_pt * on * nl2, n)
        info_v = np.bincount(k, w_pt * on * nh2 * (h < cfg.vertical_ref_max_h) *
                             ((np.abs(l) > cfg.vertical_ref_min_l) | (h < cfg.vertical_bed_h)), n)
        # left/right consistency: signed lateral residuals of wall points on each side, per 5 m bin.  A centreline
        # bias shifts both sides equally; a cross-section that does not match the template (junctions, openings)
        # makes the sides disagree -> measured systematic lateral uncertainty.
        v1, dl1, _ = self.template.sample(max(t for t, _ in cfg.align_iters), l, h)
        wall = (v1 < 0.6) & (nl2 > 0.5)
        signed = v1 * np.sign(dl1)
        b = np.clip((pts[:, 0] / 5.0).astype(np.int64), 0, n // 5 + 1)
        nb = n // 5 + 2
        side_sigma = np.zeros(n)
        stats = []
        for side in (l > 0.8, l < -0.8):
            m = wall & side
            c = np.bincount(b[m], minlength=nb)
            mean = np.bincount(b[m], signed[m], minlength=nb) / np.maximum(c, 1)
            stats.append((c, mean))
        both = (stats[0][0] >= 5) & (stats[1][0] >= 5)
        dis = np.where(both, np.abs(stats[0][1] - stats[1][1]), 0.0)
        side_sigma = np.repeat(dis, 5)[:n]
        return y, z, (info, info_v, side_sigma)

    def estimate(self, pts):
        cfg = self.cfg
        xs = self.xs
        n = len(xs)
        prior = self.prev if (self.prev is not None and self.prev.ok) else None
        rails = extract_rails(pts, cfg, prior)
        good = rails['ev'] > cfg.rail_min_evidence
        if good.sum() < 5:
            # rails not visible (occlusion, glitch frame): keep the previous geometry
            if self.prev is not None:
                self.prev.meas = dict(rails=rails)
                return self.prev
            return TrackGeometry(xs, np.zeros(n), np.full(n, -1.2), ok=False, meas=dict(rails=rails))
        xr, yr = rails['xc'][good], rails['yc'][good]
        zl, zrr = rails['zl'][good], rails['zr'][good]
        both = np.isfinite(zl) & np.isfinite(zrr)
        zmean = np.where(both, 0.5 * (zl + zrr), np.nan)
        if both.sum() >= 3:
            roll_meas = float(np.median((zl - zrr)[both])) / GAUGE_CC
            self.roll = roll_meas if prior is None else (1 - cfg.roll_alpha) * self.roll + cfg.roll_alpha * roll_meas
        rail_range = float(xr.max())
        ri = np.clip(np.round(xr / cfg.grid_step).astype(int), 0, n - 1)
        zf = np.isfinite(zmean)
        rw = 1.0 / cfg.rail_sigma ** 2
        rwy = np.full(len(ri), rw)
        if prior is not None:
            # far rail detections must agree with the previous geometry (junction/gate guide rails can mislead DP)
            dev = np.abs(yr - prior.centre(xr))
            rwy = np.where((xr > cfg.rail_trust_x) & (dev > 0.10 + 0.005 * xr), rw * 0.01, rwy)
        rails_y = (ri, yr, rwy)
        rails_z = (ri[zf], zmean[zf], np.full(zf.sum(), rw))
        pwy, pwz = _temporal_weights(prior, cfg)

        # ---- initial profile: rails (+ previous frame)
        y = _solve_profile(n, cfg.smooth_lat, *rails_y, None if prior is None else prior.yc, pwy,
                           x0=None if prior is None else prior.yc.copy())
        z = _solve_profile(n, cfg.smooth_vert, *rails_z, None if prior is None else prior.zr, pwz,
                           x0=None if prior is None else prior.zr.copy())
        if prior is None:
            # first frame: linear extrapolation of the rails trend
            z = z if zf.sum() >= 3 else np.full(n, -1.2)

        # ---- template from the near field
        ref = TrackGeometry(xs, y, z, roll=self.roll)
        near = pts[(pts[:, 0] > cfg.template_x[0]) & (pts[:, 0] < min(cfg.template_x[1], max(12.0, rail_range)))]
        if self.frame_count % cfg.template_every == 0 or not self.template.fields:
            near = near[::max(1, len(near) // 15000)]
            _, lt, ht = ref.to_track(near)
            self.template.update(lt, ht)

        # ---- dense alignment of the far field
        far = pts[pts[:, 0] >= cfg.align_x_min]
        if len(far) > cfg.align_points:
            # keep far (sparse, valuable) points preferentially: probability ~ 1/density
            xb = np.clip((far[:, 0] / 5).astype(int), 0, 100)
            dens = np.bincount(xb, minlength=101)[xb].astype(float)
            p = 1.0 / dens
            # seeded by the frame content: the same frame always gives the same geometry, independent of history
            rng = np.random.default_rng([cfg.sample_seed, len(far), int(np.abs(far[:, 0]).sum() * 100) % (1 << 32)])
            keep = rng.random(len(far)) < p * (cfg.align_points / p.sum())
            far = far[keep]
        info = (np.zeros(n), np.zeros(n), np.zeros(n))
        if len(far) >= 50:
            # global coarse lateral profile (robust to local minima), then dense refinement
            self.frame_count += 1
            full = prior is None or self.frame_count % cfg.dp_full_every == 0
            xc, off, informative = self._coarse_dp(far, y, z, max(cfg.align_x_min, min(rail_range, 30.0) - 5.0),
                                                   None if full else cfg.dp_local_range)
            if informative.sum() >= 3:
                gi = np.clip(np.round(xc[informative] / cfg.grid_step).astype(int), 0, n - 1)
                yi = np.interp(xc[informative], xs, y) + off[informative]
                y = _solve_profile(n, cfg.smooth_lat, np.r_[ri, gi], np.r_[yr, yi],
                                   np.r_[rwy, np.full(len(gi), 1.0 / 0.15 ** 2)])
            y, z, info = self._align(far, y, z, prior, rails_y, rails_z)
        # ---- validity: contiguous run of measured nodes beyond the rails (gaps up to 12 m tolerated)
        info_l, info_v, side_sigma = info
        measured = info_l > cfg.align_min_info
        valid_range = rail_range
        gap = 0
        for kk in range(int(rail_range), n):
            if measured[kk]:
                valid_range = xs[kk]
                gap = 0
            else:
                gap += 1
                if gap > 12:
                    break
        # ---- uncertainty.  Random part: information from on-surface points (+ rails); only MEASURED nodes carry
        #      information and fuse (Kalman-style) with the previous frame; extrapolation adds no information and
        #      grows linearly from the last measured node.  Systematic part (distance-growing, side disagreement)
        #      is never reduced by fusion.
        rail_info = np.zeros(n)
        np.add.at(rail_info, ri, rw)
        sys_l = cfg.sigma_sys_l[0] + cfg.sigma_sys_l[1] * xs / 100.0
        sys_v = cfg.sigma_sys_v[0] + cfg.sigma_sys_v[1] * xs / 100.0
        side = np.maximum.accumulate(np.convolve(side_sigma, np.ones(11) / 11, 'same'))   # persists beyond a junction

        def random_part(info, prev_rand, q):
            inf5 = np.convolve(info, np.ones(5), 'same')
            meas = np.where(inf5 > cfg.align_min_info, np.minimum(1.0 / np.sqrt(inf5 + 25.0), 0.2), np.inf)
            if prev_rand is not None:
                prev = np.where(np.isfinite(prev_rand), prev_rand ** 2 + q ** 2, np.inf)
                with np.errstate(divide='ignore'):
                    fused = 1.0 / np.sqrt(np.where(np.isfinite(meas), 1.0 / meas ** 2, 0.0) +
                                          np.where(np.isfinite(prev), 1.0 / prev, 0.0))
            else:
                fused = meas
            known = np.isfinite(fused)
            # extrapolate from the last known node (and before the first one from the nearest known node)
            idx = np.where(known, np.arange(n), -1)
            np.maximum.accumulate(idx, out=idx)
            out = np.full(n, 0.2 + cfg.sigma_unmeasured_per_m * xs)
            has = idx >= 0
            out[has] = fused[idx[has]] + cfg.sigma_unmeasured_per_m * (xs[has] - xs[idx[has]])
            return out, np.where(known, fused, np.inf)

        near_l = np.where(xs <= rail_range, rail_info * 4.0, 0.0)          # rails measure the near field
        rand_l, keep_l = random_part(info_l + near_l, None if prior is None else prior.rand_l, cfg.temporal_process[0])
        rand_v, keep_v = random_part(info_v + near_l, None if prior is None else prior.rand_v, cfg.temporal_process[1])
        sigma_l = np.sqrt(sys_l ** 2 + rand_l ** 2 + side ** 2)
        sigma_v = np.sqrt(sys_v ** 2 + rand_v ** 2)
        over = np.nonzero(sigma_l > cfg.trusted_sigma)[0]
        trusted = float(xs[over[0] - 1]) if len(over) and over[0] > 0 else float(xs[-1])
        geo = TrackGeometry(xs, y, z, roll=self.roll, rail_range=rail_range, valid_range=float(valid_range), ok=True,
                            sigma_l=sigma_l, sigma_v=sigma_v, rand_l=keep_l, rand_v=keep_v,
                            trusted_range=max(trusted, rail_range))
        geo.meas = dict(rails=rails, info=info_l, info_v=info_v)
        self.prev = geo
        return geo
