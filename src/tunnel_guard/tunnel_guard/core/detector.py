"""TunnelGuard pipeline: point cloud -> track geometry -> envelope test -> clusters -> confirmed obstacles.

Pure numpy/scipy, no ROS dependency, so the same code runs in the ROS 2 node, in offline evaluation and in tests.
"""
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from .adapt import AdaptConfig, ClutterCalibrator

from .cloud import AxisDetector, MountCalibrator, forward_rotation
from .cluster import ClusterConfig, Clusterer
from .farfield import FarField, FarFieldConfig
from .features import FEATURE_NAMES, candidate_features
from .gauge import IN_GAUGE, NEAR_GAUGE, GaugeConfig, classify
from .geometry import GeometryConfig, GeometryEstimator
from .tracker import Tracker, TrackerConfig

LEVEL_CLEAR, LEVEL_CAUTION, LEVEL_STOP = 0, 1, 2


@dataclass
class DetectorConfig:
    forward_axis: str = 'auto'          # 'auto' | 'x' | '-x' | 'y' | '-y'
    mount_calibration: bool = True      # estimate lidar pitch/roll from the track bed and level the cloud (cloud.py)
    min_range: float = 1.5              # ignore returns from the own train body
    max_range: float = 500.0            # longest range processed (Falcon Prime: 500 m; Pandar128: 200 m instrumented)
    report_range_cap: float = 0.0       # cap of the reported clear distance; 0 = automatic: the reach the lidar actually
                                        # demonstrates (running max of the 99.9th percentile of forward returns)
    half_fov_deg: float = 70.0
    min_distance: float = 3.0           # along-track distance below which returns are ignored (own train, spray)
    horizon_slack: float = 5.0          # evaluate the envelope this far beyond the verified geometry
    cand_min_points: int = 3
    min_in_gauge_points: int = 2        # confident envelope points required for an IN_GAUGE (STOP) object
    cand_min_points_far: int = 3
    near_decimate_x: float = 25.0       # points closer than this are decimated (very dense, little extra information)
    near_decimate: int = 3
    sight_clearance: float = 1.8        # lateral/vertical room of the line of sight inside the tunnel [m]
    containment_check: bool = True      # a foreign object must have tunnel structure beyond it laterally
    containment_gap: float = 0.3
    containment_min_points: int = 3
    containment_sigma_k: float = 3.0    # containment is skipped when a wall displacement would need > k sigma
    wall_min_offset: float = 1.48       # innermost tunnel wall offset measured in all recordings [m]
    gravity_min_s: float = 60.0         # beyond this distance STOP requires a floor-supported object
    gravity_max_base: float = 1.0       # lowest point of a floor-supported object above the rail plane [m]
    extent_min_s: float = 40.0          # shape plausibility applies beyond this distance
    extent_min_beams: float = 1.2       # required vertical extent in vertical beam spacings
    extent_min_points_exempt: int = 15  # well-supported clusters are exempt
    scorer_model: str = ''              # learned second-stage scorer (JSON); '' / 'none' = physics rules only (the ROS node
                                        # and evaluate_bag substitute the packaged model for '')
    scorer_model_hang: str = 'auto'     # scorer for floating / hanging candidates: 'auto' = obstacle_scorer_hang.json next to
                                        # the main scorer, a path, or 'none' (main scorer only)
    route_hang_h: float = 0.5           # candidates whose bottom is at least this high above the rails use it [m]
    scorer_near_override_s: float = 30.0  # below this distance the rule decision stands (scorer cannot suppress)
    scorer_shell_veto: bool = True      # shell-attached clusters can never be promoted by the scorer
    scorer_smooth_k: int = 0            # temporal score smoothing window; 0 = value stored with the model
    two_tier: bool = True               # STOP only within the range measured in this frame (beyond: CAUTION)
    shell_check: bool = True            # tall clusters continuing upward into the tunnel shell are infrastructure
    shell_min_height: float = 2.5       # only clusters reaching this high are tested [m above rail]
    shell_gap: float = 1.2              # search window above the cluster top [m]
    shell_lateral: float = 0.4
    shell_min_points: int = 3
    shell_hanging_h: float = 0.8       # a shell-attached cluster whose bottom is above this is 'hanging', not infrastructure
    shell_touch_min: float = 0.25      # max gap between cluster top and shell for attachment [m]
    shell_touch_beams: float = 3.0     #   or this many vertical beam spacings at the cluster's range
    shell_edge_l: float = 0.75         # only clusters reaching this close to the envelope edge can be infrastructure: fixed
                                       # structure never stands in the core of the kinematic envelope, a hanging cable can
    scatter_check: bool = True          # dust / aerosol / spray: volumetric returns cannot trigger STOP
    scatter_cell_deg: float = 0.3       # angular cell for the depth-spread test
    scatter_depth: float = 0.5          # a solid surface returns at one depth per line of sight; a cloud at many [m]
    scatter_frac: float = 0.5           # ... share of a cluster's returns that are scatter returns
    scatter_depth_max: float = 5.0      # returns further apart along the line of sight are different objects (a person
                                        # behind a dust cloud stays a person)
    scatter_min_cells: int = 3
    sensor_adaptive: bool = True        # measure the lidar's angular sampling and normalise beam-based quantities to it
    hold_frames: int = 10              # a confirmed STOP object that drops out (thin wire between beams, below the field of
                                       # view when very close) is held at its predicted distance for up to this many frames
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    gauge: GaugeConfig = field(default_factory=GaugeConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    farfield: FarFieldConfig = field(default_factory=FarFieldConfig)
    adapt: AdaptConfig = field(default_factory=lambda: AdaptConfig(enabled=False))   # online self-calibration (adapt.py)
    ego_check: bool = True              # ego-motion consistency: a STOP object must close in as the train advances
    ego_min_travel: float = 4.0         # train travel (lidar odometry) over the track history needed to test [m]
    ego_max_slope: float = -0.35        # veto STOP when d(distance)/d(train travel) > this (static object: -1, artefact
                                        # carried along with the train: 0)
    ego_history: int = 10
    ego_pool_min_dx: float = 5.0        # neighbourhood test (young tracks): only frames the train has since advanced this far
    ego_pool_ds: float = 2.5            #   from count; support window along track and across [m]
    ego_pool_dl: float = 0.8
    stop_debounce_frames: int = 1       # >1: beyond stop_debounce_s a track must be STOP-eligible on this many of its last
    stop_debounce_window: int = 3       # stop_debounce_window frames.  Off: on the sealed drive it removed ~1 false event
                                        # per 2.8 km but cost far-field recall (box@120 m 71 -> 57 %) - not worth it
    stop_debounce_s: float = 30.0
    skip_duplicates: bool = True        # re-issue the last decision for a bit-identical repeated cloud
    settle_frames: int = 10             # start-up: the track estimate must change < settle_tol over 20-150 m on this many
    settle_tol: float = 0.15            # consecutive frames before STOP is allowed beyond the directly measured rails [m]


@dataclass
class FrameResult:
    stamp: float
    level: int
    obstacles: list
    nearest_distance: float
    nearest_ttc: float
    clear_distance: float
    curvature: float
    geometry: object
    timings: dict
    candidates: list
    forward_points: np.ndarray = None
    zone: np.ndarray = None
    rotation: np.ndarray = None
    all_candidates: list = None
    features: np.ndarray = None
    envelope_range: float = 0.0     # how far the envelope is evaluated (frame pipeline + far-field evidence)
    dust: list = None               # volumetric scatter zones (dust, aerosol, spray): information only


class ObstacleDetector:
    def __init__(self, cfg: DetectorConfig = None):
        self.cfg = cfg or DetectorConfig()
        self.geo = GeometryEstimator(self.cfg.geometry)
        self.clusterer = Clusterer(self.cfg.cluster)
        self.tracker = Tracker(self.cfg.tracker)
        self.ftracker = Tracker(self.cfg.tracker)       # tracks every candidate (temporal features for the scorer)
        self.scorer = None
        self.score_hist = {}
        if self.cfg.scorer_model and self.cfg.scorer_model.lower() != 'none':
            from .scorer import ObstacleScorer
            self.scorer = ObstacleScorer(self.cfg.scorer_model)
        self.scorer_hang = None
        hang = self.cfg.scorer_model_hang
        if self.scorer is not None and hang.lower() == 'auto':
            # the floating/hanging expert ships next to the main scorer (config/obstacle_scorer_hang.json)
            cand = os.path.join(os.path.dirname(os.path.abspath(self.cfg.scorer_model)), 'obstacle_scorer_hang.json')
            hang = cand if os.path.exists(cand) else ''
        if self.scorer is not None and hang and hang.lower() != 'none':
            from .scorer import ObstacleScorer
            self.scorer_hang = ObstacleScorer(hang)
        self.axis = AxisDetector()
        self.mount = MountCalibrator()
        self.far = FarField(self.cfg.farfield)
        self._settled, self._geo_ref = 0, None
        self._dens = []
        self.sensor_scale = 1.0
        self.calib = ClutterCalibrator(self.cfg.adapt)
        self.recent = deque(maxlen=self.cfg.ego_history)   # per frame: (odometry segment, x, s, l) of candidates
        self._last, self._last_fp = None, None
        self._last_stamp = None
        self._reach = 0.0

    def reset(self):
        self.recent.clear()
        self._last, self._last_fp = None, None
        self._last_stamp = None
        self._reach = 0.0
        self.geo.reset()
        self.tracker.reset()
        self.ftracker.reset()
        self.score_hist.clear()
        self.calib.reset()
        self.axis = AxisDetector()
        self.mount = MountCalibrator()
        self.far = FarField(self.cfg.farfield)
        self._settled, self._geo_ref = 0, None
        self._dens = []
        self.sensor_scale = 1.0

    def _range_cap(self, p):
        """Reported clear distance never exceeds what the lidar demonstrably measures."""
        if self.cfg.report_range_cap > 0:
            return self.cfg.report_range_cap
        if len(p) > 20:
            # demonstrated reach: the 10th farthest return ahead (robust to single spurious points)
            self._reach = max(self._reach, float(np.partition(p[:, 0], len(p) - 10)[len(p) - 10]))
        return min(self._reach + 1.0, self.cfg.max_range)

    def _sense_density(self, p):
        """Angular sampling density of this lidar (rays/deg^2) in the forward cone at 15-80 m, from the fullest cells;
        after 10 frames the beam-based quantities are normalised to it (Pandar128 reference: 80 rays/deg^2)."""
        q = p[(p[:, 0] > 15) & (p[:, 0] < 80)]
        az = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
        el = np.degrees(np.arctan2(q[:, 2], np.hypot(q[:, 0], q[:, 1])))
        m = (np.abs(az) < 12) & (np.abs(el) < 4)
        if np.count_nonzero(m) < 200:
            return
        key = np.floor(az[m] / 0.5).astype(np.int64) * 1000 + np.floor(el[m] / 0.5).astype(np.int64)
        counts = np.bincount(np.unique(key, return_inverse=True)[1])
        self._dens.append(float(np.percentile(counts, 90)) / 0.25)
        if len(self._dens) == 10:
            scale = float(np.sqrt(80.0 / max(np.median(self._dens), 1.0)))
            if abs(scale - 1.0) < 0.15:
                scale = 1.0                      # the training sensor itself (Pandar128): exactly unchanged behaviour
            self.sensor_scale = scale

    def _scatter_flags(self, q):
        """Per-point scatter flag: another return in the same small solid angle lies scatter_depth..scatter_depth_max
        in front of or behind it.  A solid surface returns from one depth per line of sight; a cloud from many."""
        cfg = self.cfg
        flag = np.zeros(len(q), bool)
        if len(q) < 2:
            return flag
        r = np.sqrt(np.einsum('ij,ij->i', q, q))
        az = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
        el = np.degrees(np.arctan2(q[:, 2], np.hypot(q[:, 0], q[:, 1])))
        key = np.floor(az / cfg.scatter_cell_deg).astype(np.int64) * 100003 + \
            np.floor(el / cfg.scatter_cell_deg).astype(np.int64)
        o = np.lexsort((r, key))
        k, rr = key[o], r[o]
        kr = (k - k.min()).astype(np.float64) * 1e5 + rr          # cells far apart on one sorted axis (r < 1e4 m)
        j_up = np.searchsorted(kr, kr + cfg.scatter_depth)        # first return at least scatter_depth deeper
        j_dn = np.searchsorted(kr, kr - cfg.scatter_depth, side='right') - 1
        n = len(kr)
        up_ok = j_up < n
        ju = np.minimum(j_up, n - 1)
        f = up_ok & (k[ju] == k) & (rr[ju] - rr < cfg.scatter_depth_max)
        dn_ok = j_dn >= 0
        jd = np.maximum(j_dn, 0)
        f |= dn_ok & (k[jd] == k) & (rr - rr[jd] < cfg.scatter_depth_max)
        flag[o] = f
        return flag

    def _scatter(self, q):
        """Volumetric scatter (dust, aerosol, spray, smoke) versus a solid surface.  Within one small solid angle a
        solid object returns from a single depth; a cloud returns from many.  True when the depth spread inside the
        angular cells that hold several returns is large in most of them."""
        cfg = self.cfg
        if len(q) < 6:
            return False
        r = np.sqrt(np.einsum('ij,ij->i', q, q))
        az = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
        el = np.degrees(np.arctan2(q[:, 2], np.hypot(q[:, 0], q[:, 1])))
        key = np.floor(az / cfg.scatter_cell_deg).astype(np.int64) * 100003 + \
            np.floor(el / cfg.scatter_cell_deg).astype(np.int64)
        o = np.argsort(key, kind='stable')
        k, rr = key[o], r[o]
        starts = np.r_[0, np.nonzero(np.diff(k))[0] + 1]
        ends = np.r_[starts[1:], len(k)]
        multi = (ends - starts) >= 2
        if np.count_nonzero(multi) < cfg.scatter_min_cells:
            return False
        spread = np.maximum.reduceat(rr, starts)[multi] - np.minimum.reduceat(rr, starts)[multi]
        return bool(np.mean(spread > cfg.scatter_depth) >= cfg.scatter_frac)

    def _near(self, lo, hi):
        nmax = len(self._nb_starts) - 1
        a = self._nb_starts[int(np.clip(np.floor(lo), 0, nmax))]
        b = self._nb_starts[int(np.clip(np.floor(hi) + 1, 0, nmax))]
        return self._nb_order[a:b]

    def _wall_counts(self, c, s, l, h):
        cfg = self.cfg
        ds = 2.0 + 0.03 * c['s_min']
        i = self._near(c['s_min'] - ds, c['s_max'] + ds)
        si, li, hi = s[i], l[i], h[i]
        m = (si > c['s_min'] - ds) & (si < c['s_max'] + ds) & (hi > c['h_min'] - 0.6) & (hi < c['h_max'] + 0.6)
        return (int(np.count_nonzero(m & (li > c['l_max'] + cfg.containment_gap))),
                int(np.count_nonzero(m & (li < c['l_min'] - cfg.containment_gap))))

    def _shell_points(self, c, s, l, h):
        cfg = self.cfg
        ds = 1.0 + 0.01 * c['s_min']
        i = self._near(c['s_min'] - ds, c['s_max'] + ds)
        si, li, hi = s[i], l[i], h[i]
        m = ((si > c['s_min'] - ds) & (si < c['s_max'] + ds) & (hi > c['h_max']) & (hi < c['h_max'] + cfg.shell_gap)
             & (li > c['l_min'] - cfg.shell_lateral) & (li < c['l_max'] + cfg.shell_lateral))
        if not m.any():
            return 0
        # attachment needs continuity: the shell must start right above the cluster top (a few beam spacings); an object
        # floating below the roof with a clear gap is not infrastructure
        touch = max(cfg.shell_touch_min, cfg.shell_touch_beams * cfg.cluster.beam_vertical_rad * c['s_min'])
        return int(np.count_nonzero(m)) if float(hi[m].min()) - c['h_max'] <= touch else 0

    def _carried_along(self, tr):
        """True when the track keeps its distance while the train demonstrably advances (lidar odometry): a solid
        object on the track closes in at the train's own speed (slope -1), an artefact of the train's own view
        (geometry model error travelling with the train, spray, reflections) does not (slope ~0).  Robust
        Theil-Sen slope over the current unbroken odometry segment; no decision without enough travel."""
        cfg = self.cfg
        o = [(x, s) for seg, x, s in tr.obs if seg == self.calib.segment]
        if len(o) >= 4:
            x, s = np.array(o).T
            if x.max() - x.min() >= cfg.ego_min_travel:
                i, j = np.triu_indices(len(x), 1)
                dx = x[j] - x[i]
                ok = np.abs(dx) > 0.5
                if ok.sum() >= 3:
                    return float(np.median((s[j] - s[i])[ok] / dx[ok])) > cfg.ego_max_slope
        # young track (an artefact often fragments into short-lived tracks): test its neighbourhood instead.  In the
        # frames since which the train advanced >= ego_pool_min_dx, was there something at the same distance ahead
        # (carried along) or at distance + travel (static)?  Veto only with carried-along support and no static support.
        return self._pooled_carried(float(tr.last['s_min']), float(tr.last['l_mean']))

    def _pooled_carried(self, s_now, l_now):
        cfg = self.cfg
        co = st = 0
        for seg, x, ss, ll in self.recent:
            dx = self.calib.x - x
            if seg != self.calib.segment or dx < cfg.ego_pool_min_dx:
                continue
            near = ss[np.abs(ll - l_now) < cfg.ego_pool_dl]
            co += bool(np.any(np.abs(near - s_now) < cfg.ego_pool_ds))
            st += bool(np.any(np.abs(near - (s_now + dx)) < cfg.ego_pool_ds))
        return co >= 2 and st == 0

    def process(self, xyz, stamp, intensity=None, ring=None):
        cfg = self.cfg
        t0 = time.perf_counter()
        tm = {}
        xyz = np.asarray(xyz, dtype=np.float32)
        # a bit-identical repeat of the previous cloud (recorders and simulators re-send frames) carries no new
        # information: re-issue the last decision instead of feeding the trackers a frame with zero motion
        fp = (len(xyz), hash(xyz[::max(1, len(xyz) // 4096)].tobytes()))
        if cfg.skip_duplicates and self._last is not None and fp == self._last_fp:
            return self._last
        self._last_fp = fp
        r2 = np.einsum('ij,ij->i', xyz, xyz)
        valid = np.isfinite(r2) & (r2 > cfg.min_range ** 2) & (r2 < cfg.max_range ** 2)
        if cfg.forward_axis != 'auto':
            axis = cfg.forward_axis
        elif self.axis.axis is not None:
            axis = self.axis.axis
        else:
            axis = self.axis.update(xyz[valid])
        Rm = forward_rotation(axis).astype(np.float32)
        # the rotation is a signed axis permutation: select columns instead of a matrix product (identical values)
        cols = []
        for k in range(3):
            j = int(np.argmax(np.abs(Rm[k])))
            cols.append(xyz[:, j] if Rm[k, j] > 0 else -xyz[:, j])
        X, Y, Z = cols
        if cfg.mount_calibration:
            Rl = self.mount.R
            if Rl is None:
                Rl = self.mount.update(np.stack([X[valid], Y[valid], Z[valid]], 1))
                if Rl is not None:
                    self.geo.reset()                       # geometry restarts in the levelled frame
            if Rl is not None and not np.allclose(Rl, np.eye(3)):
                Rl32 = Rl.astype(np.float32)
                X, Y, Z = (Rl32[0, 0] * X + Rl32[0, 1] * Y + Rl32[0, 2] * Z,
                           Rl32[1, 0] * X + Rl32[1, 1] * Y + Rl32[1, 2] * Z,
                           Rl32[2, 0] * X + Rl32[2, 1] * Y + Rl32[2, 2] * Z)
                Rm = (Rl32 @ Rm).astype(np.float32)
        fov = valid & (X > 1.0) & (np.abs(Y) < np.tan(np.radians(cfg.half_fov_deg)) * X + 3.0) & (np.abs(Z) < 12)
        sel = np.flatnonzero(fov)
        if cfg.near_decimate > 1:
            rank = np.cumsum(valid) - 1                     # position among valid points (decimation pattern)
            sel = sel[(X[sel] >= cfg.near_decimate_x) | (rank[sel] % cfg.near_decimate == 0)]
        p = np.stack([X[sel], Y[sel], Z[sel]], axis=1)
        inten = np.asarray(intensity)[sel] if intensity is not None else None
        rg = np.asarray(ring)[sel] if ring is not None else None
        tm['prep'] = time.perf_counter() - t0

        t1 = time.perf_counter()
        v = self.calib.ego_speed() if (cfg.adapt.enabled or cfg.ego_check) else None
        dt = 0.0 if self._last_stamp is None else float(np.clip(stamp - self._last_stamp, 0.0, 1.0))
        self._last_stamp = stamp
        travel = (v or 0.0) * dt
        g = self.geo.estimate(p, travel=travel)
        if g.ok:
            xq = np.arange(20.0, 151.0, 10.0)
            prof = np.r_[g.centre(xq), g.rail_z(xq)]
            stable = self._geo_ref is not None and np.max(np.abs(prof - self._geo_ref)) < cfg.settle_tol
            self._settled = self._settled + 1 if stable else 0
            self._geo_ref = prof
        else:
            self._settled, self._geo_ref = 0, None
        settled = self._settled >= cfg.settle_frames
        if cfg.sensor_adaptive and len(self._dens) < 10:
            self._sense_density(p)
        tm['geometry'] = time.perf_counter() - t1

        t2 = time.perf_counter()
        sight = g.sight_distance(cfg.sight_clearance, cfg.max_range) if g.ok else 0.0
        horizon = min(cfg.max_range, max(g.valid_range, g.rail_range, g.trusted_range) + cfg.horizon_slack, sight) if g.ok else 0.0
        near_env = p[:, 0] < horizon
        s, l, h = g.to_track(p[near_env])
        if cfg.adapt.enabled or cfg.ego_check:
            self.calib.update_motion(s, l, h, stamp)
        # 1 m distance buckets: evidence queries per candidate touch only nearby points
        nbk = int(cfg.max_range) + 1
        sb = np.clip(s.astype(np.int64), 0, nbk - 1)
        self._nb_order = np.argsort(sb.astype(np.int16), kind='stable')
        self._nb_starts = np.r_[0, np.cumsum(np.bincount(sb, minlength=nbk))]
        top = max(v[0] for v in cfg.gauge.profile)
        pre = (h > cfg.gauge.h_min) & (h < top) & (np.abs(l) < max(v[1] for v in cfg.gauge.profile) + cfg.gauge.near_margin)
        zone_sub = np.zeros(len(s), np.uint8)
        pre &= s > cfg.min_distance
        sl, sv = g.sigma_at(s[pre])
        zone_sub[pre] = classify(s[pre], l[pre], h[pre], cfg.gauge, sl, sv)
        # two-tier decision: STOP only where this frame measured the track geometry; beyond it (temporally trusted /
        # extrapolated geometry) an in-envelope object is an early warning (CAUTION)
        measured = max(g.valid_range, g.rail_range) + cfg.horizon_slack
        if cfg.two_tier:
            zone_sub[(zone_sub == IN_GAUGE) & (s > measured)] = NEAR_GAUGE
        zone = np.zeros(len(p), np.uint8)
        zone[np.nonzero(near_env)[0]] = zone_sub
        # scatter returns (dust, aerosol, spray) among envelope points and the far corridor
        scat = np.zeros(len(p), bool)
        if cfg.scatter_check:
            sc_idx = np.flatnonzero((zone > 0) | ((p[:, 0] > cfg.farfield.s_min) & (np.abs(p[:, 1]) < 6.0)))
            scat[sc_idx] = self._scatter_flags(p[sc_idx])
        e = zone_sub > 0
        eidx = np.nonzero(near_env)[0][e]
        clusters = self.clusterer.run(s[e], l[e], h[e], zone_sub[e],
                                      ring=rg[eidx] if rg is not None else None,
                                      intensity=inten[eidx] if inten is not None else None,
                                      xyz=p[eidx])
        for c in clusters:
            c['idx'] = eidx[c['idx']]
        cands_all = [c for c in clusters if c['n'] >= (cfg.cand_min_points if c['s_min'] < 100 else cfg.cand_min_points_far)]
        # physical plausibility tests, evaluated on every candidate; evidence is kept for the learned scorer
        for c in cands_all:
            c['in_gauge_raw'] = c['in_gauge']
            c['gravity_fail'] = c['s_min'] > cfg.gravity_min_s and c['h_min'] > cfg.gravity_max_base
            c['scatter'] = cfg.scatter_check and len(c['idx']) >= 3 and \
                float(np.mean(scat[c['idx']])) >= cfg.scatter_frac
            c['shape_fail'] = (c['s_min'] > cfg.extent_min_s and c['n'] < cfg.extent_min_points_exempt and
                               c['h_max'] - c['h_min'] < cfg.extent_min_beams * cfg.cluster.beam_vertical_rad * c['s_min'])
            left, right = self._wall_counts(c, s, l, h)
            c['wall_left'], c['wall_right'] = left, right
            need = cfg.containment_min_points
            ok = (left >= need) if c['l_mean'] > 0.3 else ((right >= need) if c['l_mean'] < -0.3 else (left >= need or right >= need))
            sig_l = float(g.sigma_at(np.array([c['s_min']]))[0][0])
            depth = cfg.wall_min_offset - max(abs(c['l_min']), abs(c['l_max'])) if c['l_min'] * c['l_max'] > 0 else cfg.wall_min_offset
            c['contained'] = (not cfg.containment_check) or depth > cfg.containment_sigma_k * sig_l or ok
            c['shell_pts'] = self._shell_points(c, s, l, h)
            # infrastructure = a floor-to-roof slice attached to the shell (tunnel wall at a curve, pole); an object hanging
            # from the roof whose bottom is clear of the floor is a hazard and is judged by the scorer instead
            c['shell'] = (cfg.shell_check and c['h_max'] >= cfg.shell_min_height and c['shell_pts'] >= cfg.shell_min_points
                          and c['h_min'] < cfg.shell_hanging_h
                          and max(abs(c['l_min']), abs(c['l_max'])) >= cfg.shell_edge_l)
        # rule-based decision path: suspended / sliver clusters cannot trigger STOP; uncontained or shell-attached
        # clusters are infrastructure
        cands = []
        for c in cands_all:
            if c['gravity_fail'] or c['shape_fail'] or c['scatter']:
                c['in_gauge'] = 0
            if c['contained'] and not c['shell'] and not c['scatter']:
                cands.append(c)
        tm['envelope'] = time.perf_counter() - t2
        tm['sight'] = sight

        t3 = time.perf_counter()
        # features of every candidate (temporal context from a tracker over all candidates)
        feats = []
        for c in cands_all:
            c['ml_view'] = dict(c)
        self.ftracker.update([c['ml_view'] for c in cands_all], stamp,
                             ego_speed=self.calib.ego_speed() if (cfg.adapt.enabled or cfg.ego_check) else None)
        for c in cands_all:
            feats.append(candidate_features(c['ml_view'], g, cfg.gauge, sight, c['ml_view'].get('track'), self.sensor_scale))
        feats = np.asarray(feats, np.float32).reshape(-1, len(FEATURE_NAMES))
        if self.scorer is not None and len(cands_all):
            # hybrid decision (validated leave-one-recording-out): beyond the near field a temporally smoothed learned
            # score decides STOP eligibility (it can veto rule false alarms and rescue objects the rules rejected);
            # in the near field the physics rules always keep their STOP decision
            probs = self.scorer.prob(feats)
            tau = self.calib.threshold(self.scorer.threshold)
            tm['tau'] = tau
            # mixture of experts by physical state: floor-supported candidates are judged by the main scorer, floating /
            # hanging ones (bottom clear of the rails) by the scorer trained on the full hazard space.  Each score is
            # compared with its own out-of-fold threshold, so the smoothed quantity is the margin p - tau
            margins = probs - tau
            if self.scorer_hang is not None:
                ph = self.scorer_hang.prob(feats)
                hang = np.array([c['h_min'] >= cfg.route_hang_h for c in cands_all])
                margins = np.where(hang, ph - self.scorer_hang.threshold, margins)
                probs = np.where(hang, ph, probs)
            live = {t.id for t in self.ftracker.tracks}
            decided = []
            for c, pr, mg in zip(cands_all, probs, margins):
                tr = c['ml_view'].get('track')
                key = tr.id if tr is not None else -1
                hist = self.score_hist.setdefault(key, [])
                hist.append(float(mg))
                del hist[:-(cfg.scorer_smooth_k or self.scorer.smooth_k)]
                c['score'] = tau + float(np.mean(hist))      # = smoothed probability when a single scorer is used
                rule_kept = c['contained'] and not c['shell'] and not c['scatter']
                rule_stop = rule_kept and c['in_gauge'] >= cfg.min_in_gauge_points
                # physical evidence of fixed infrastructure (attachment to the tunnel shell) is a hard veto
                ml_stop = (c['in_gauge_raw'] >= cfg.min_in_gauge_points and c['score'] >= tau
                           and not (cfg.scorer_shell_veto and c['shell']) and not c['scatter'])
                eligible = ml_stop or (rule_stop and c['s_min'] < cfg.scorer_near_override_s)
                if eligible or rule_kept:
                    c['in_gauge'] = c['in_gauge_raw'] if eligible else 0
                    decided.append(c)
            cands = decided
            self.score_hist = {k: v for k, v in self.score_hist.items() if k in live}
            if cfg.adapt.enabled:
                self.calib.observe([(c['s_min'], c['score']) for c in cands_all if 'score' in c], stamp)
        ego_v = self.calib.ego_speed() if (cfg.adapt.enabled or cfg.ego_check) else None
        tracks = self.tracker.update(cands, stamp, ego_speed=ego_v)
        for tr in tracks:
            if tr.last is not None:
                tr.obs.append((self.calib.segment, self.calib.x, float(tr.last['s_min'])))
                del tr.obs[:-cfg.ego_history]
        obstacles = []
        for tr in tracks:
            if tr.confirmed and tr.last is None and tr.prev is not None and tr.zone_last == IN_GAUGE \
                    and tr.misses <= cfg.hold_frames and tr.s > 0.5:
                # object permanence: keep a STOP object that briefly drops out, at its predicted distance
                c = tr.prev
                vv = max(tr.v, 0.0)
                obstacles.append(dict(id=tr.id, zone=IN_GAUGE, distance=float(tr.s), range=float(tr.s),
                                      centroid=c['centroid'] @ Rm, nearest=c['nearest'] @ Rm, centroid_fwd=c['centroid'],
                                      size=(c['s_max'] - c['s_min'], c['l_max'] - c['l_min'], c['h_max'] - c['h_min']),
                                      lateral=c['l_mean'], height=c['h_max'], n=0, confidence=0.5, closing_speed=vv,
                                      ttc=tr.s / vv if vv > 0.3 else float('inf'), age=tr.age,
                                      idx=np.zeros(0, np.int64), ego_veto=False, held=True))
                continue
            if not tr.confirmed or tr.last is None:
                tr.stop_hist = (tr.stop_hist + [0])[-cfg.stop_debounce_window:]
                continue
            c = tr.last
            zone_c = IN_GAUGE if c['in_gauge'] >= cfg.min_in_gauge_points else NEAR_GAUGE
            ego_veto = zone_c == IN_GAUGE and cfg.ego_check and self._carried_along(tr)
            if ego_veto:
                zone_c = NEAR_GAUGE
            tr.stop_hist = (tr.stop_hist + [int(zone_c == IN_GAUGE)])[-cfg.stop_debounce_window:]
            if zone_c == IN_GAUGE and c['s_min'] >= cfg.stop_debounce_s and sum(tr.stop_hist) < cfg.stop_debounce_frames:
                zone_c = NEAR_GAUGE
            tr.zone_last = zone_c
            v = tr.v
            ttc = tr.s / v if v > 0.3 else float('inf')
            conf = float(c['score']) if 'score' in c else float(np.clip(0.25 * tr.hit_count + 0.05 * min(c['n'], 10), 0.0, 1.0))
            obstacles.append(dict(id=tr.id, zone=zone_c, distance=c['s_min'], range=c.get('range', c['s_min']),
                                  centroid=c['centroid'] @ Rm, nearest=c['nearest'] @ Rm, centroid_fwd=c['centroid'],
                                  size=(c['s_max'] - c['s_min'], c['l_max'] - c['l_min'], c['h_max'] - c['h_min']),
                                  lateral=c['l_mean'], height=c['h_max'], n=c['n'], confidence=conf,
                                  closing_speed=v, ttc=ttc, age=tr.age, idx=c['idx'], ego_veto=ego_veto))
        # dust / aerosol zones (volumetric scatter): reported for information, never as obstacles
        dust = [dict(distance=float(c['s_min']), n=int(c['n']), far=False) for c in cands_all if c.get('scatter')]
        # far field: temporal evidence beyond the range this frame can confirm on its own
        env_range = horizon
        if cfg.farfield.enabled and g.ok and settled:
            far_to = min(self._range_cap(p), float(g.xs[-1]))
            far_from = cfg.farfield.s_min
            if far_to > far_from:
                env_range = max(env_range, far_to)
                for f in self.far.update(p, g, far_from, far_to, travel, exclude=scat):
                    if any(abs(f['distance'] - d['distance']) < 5.0 for d in dust) or                             (cfg.scatter_check and len(f['idx']) >= 6 and self._scatter(p[f['idx']])):
                        dust.append(dict(distance=f['distance'], n=int(len(f['idx'])), far=True))
                        continue
                    dup = [o for o in obstacles if abs(f['distance'] - o['distance']) < 6.0]
                    if dup:
                        # the same object seen by the frame pipeline: temporal evidence can only raise its zone
                        if f['zone'] == IN_GAUGE:
                            for o in dup:
                                o['zone'] = IN_GAUGE
                                o['confidence'] = max(o['confidence'], f['confidence'])
                        continue
                    sx = np.array([f['distance']])
                    hm = 0.5 * (f['h_min'] + f['height'])
                    c_fwd = np.array([f['distance'], float(g.centre(sx)[0]) + f['lateral'],
                                      float(g.rail_z(sx)[0]) + hm], np.float32)
                    vv = ego_v or 0.0
                    obstacles.append(dict(id=f['id'], zone=f['zone'], distance=f['distance'], range=f['distance'],
                                          centroid=c_fwd @ Rm, nearest=c_fwd @ Rm, centroid_fwd=c_fwd,
                                          size=(3 * cfg.farfield.cell, 0.6, f['height'] - f['h_min']),
                                          lateral=f['lateral'], height=f['height'], n=f['n'],
                                          confidence=f['confidence'], closing_speed=vv,
                                          ttc=f['distance'] / vv if vv > 0.3 else float('inf'), age=f['frames'],
                                          idx=f['idx'], ego_veto=False, far=True))
        self.recent.append((self.calib.segment, self.calib.x, np.array([c['s_min'] for c in cands_all], float),
                            np.array([c['l_mean'] for c in cands_all], float)))
        if not settled:
            for o in obstacles:
                if o['zone'] == IN_GAUGE and o['distance'] > g.rail_range + cfg.horizon_slack:
                    o['zone'] = NEAR_GAUGE        # geometry still converging: early warning only
        in_g = [o for o in obstacles if o['zone'] == IN_GAUGE]
        if in_g:
            level = LEVEL_STOP
        elif obstacles or any(tr.hit_count >= 2 for tr in tracks
                              if tr.last is not None and tr.last['in_gauge'] >= cfg.min_in_gauge_points):
            level = LEVEL_CAUTION
        else:
            level = LEVEL_CLEAR
        nearest = min((o['distance'] for o in in_g), default=float('nan'))
        ttc = min((o['ttc'] for o in in_g), default=float('inf'))
        clear = min(float(np.nanmin([horizon, nearest])) if in_g else float(horizon), self._range_cap(p))
        tm['tracking'] = time.perf_counter() - t3
        tm['total'] = time.perf_counter() - t0
        self._last = FrameResult(stamp=stamp, level=level, obstacles=obstacles, nearest_distance=nearest, nearest_ttc=ttc,
                           clear_distance=clear, curvature=g.curvature(20.0) if g.ok else 0.0, geometry=g,
                           timings=tm, candidates=cands, forward_points=p, zone=zone, rotation=Rm,
                           all_candidates=cands_all, features=feats, envelope_range=float(env_range), dust=dust)
        return self._last
