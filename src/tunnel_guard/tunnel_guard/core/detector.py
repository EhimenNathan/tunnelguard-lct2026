"""TunnelGuard pipeline: point cloud -> track geometry -> envelope test -> clusters -> confirmed obstacles.

Pure numpy/scipy, no ROS dependency, so the same code runs in the ROS 2 node, in offline evaluation and in tests.
"""
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from .adapt import AdaptConfig, ClutterCalibrator

from .cloud import AxisDetector, forward_rotation
from .cluster import ClusterConfig, Clusterer
from .features import FEATURE_NAMES, candidate_features
from .gauge import IN_GAUGE, NEAR_GAUGE, GaugeConfig, classify
from .geometry import GeometryConfig, GeometryEstimator
from .tracker import Tracker, TrackerConfig

LEVEL_CLEAR, LEVEL_CAUTION, LEVEL_STOP = 0, 1, 2


@dataclass
class DetectorConfig:
    forward_axis: str = 'auto'          # 'auto' | 'x' | '-x' | 'y' | '-y'
    min_range: float = 1.5              # ignore returns from the own train body
    max_range: float = 300.0
    report_range_cap: float = 210.0     # Pandar128E3X instrumented range 200 m (recordings: max 209 m): the reported clear
                                        # distance never exceeds what the lidar can actually measure
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
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    gauge: GaugeConfig = field(default_factory=GaugeConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
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
        self.calib = ClutterCalibrator(self.cfg.adapt)
        self.recent = deque(maxlen=self.cfg.ego_history)   # per frame: (odometry segment, x, s, l) of candidates
        self._last, self._last_fp = None, None

    def reset(self):
        self.recent.clear()
        self._last, self._last_fp = None, None
        self.geo.reset()
        self.tracker.reset()
        self.calib.reset()
        self.axis = AxisDetector()

    def _near(self, lo, hi):
        a = self._nb_starts[int(np.clip(np.floor(lo), 0, 400))]
        b = self._nb_starts[int(np.clip(np.floor(hi) + 1, 0, 400))]
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
        g = self.geo.estimate(p)
        tm['geometry'] = time.perf_counter() - t1

        t2 = time.perf_counter()
        sight = g.sight_distance(cfg.sight_clearance, cfg.max_range) if g.ok else 0.0
        horizon = min(cfg.max_range, max(g.valid_range, g.rail_range, g.trusted_range) + cfg.horizon_slack, sight) if g.ok else 0.0
        near_env = p[:, 0] < horizon
        s, l, h = g.to_track(p[near_env])
        if cfg.adapt.enabled or cfg.ego_check:
            self.calib.update_motion(s, l, h, stamp)
        # 1 m distance buckets: evidence queries per candidate touch only nearby points
        sb = np.clip(s.astype(np.int64), 0, 399)
        self._nb_order = np.argsort(sb.astype(np.int16), kind='stable')
        self._nb_starts = np.r_[0, np.cumsum(np.bincount(sb, minlength=400))]
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
                          and c['h_min'] < cfg.shell_hanging_h)
        # rule-based decision path: suspended / sliver clusters cannot trigger STOP; uncontained or shell-attached
        # clusters are infrastructure
        cands = []
        for c in cands_all:
            if c['gravity_fail'] or c['shape_fail']:
                c['in_gauge'] = 0
            if c['contained'] and not c['shell']:
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
            feats.append(candidate_features(c['ml_view'], g, cfg.gauge, sight, c['ml_view'].get('track')))
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
                rule_kept = c['contained'] and not c['shell']
                rule_stop = rule_kept and c['in_gauge'] >= cfg.min_in_gauge_points
                # physical evidence of fixed infrastructure (attachment to the tunnel shell) is a hard veto
                ml_stop = (c['in_gauge_raw'] >= cfg.min_in_gauge_points and c['score'] >= tau
                           and not (cfg.scorer_shell_veto and c['shell']))
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
            v = tr.v
            ttc = tr.s / v if v > 0.3 else float('inf')
            conf = float(c['score']) if 'score' in c else float(np.clip(0.25 * tr.hit_count + 0.05 * min(c['n'], 10), 0.0, 1.0))
            obstacles.append(dict(id=tr.id, zone=zone_c, distance=c['s_min'], range=c.get('range', c['s_min']),
                                  centroid=c['centroid'] @ Rm, nearest=c['nearest'] @ Rm, centroid_fwd=c['centroid'],
                                  size=(c['s_max'] - c['s_min'], c['l_max'] - c['l_min'], c['h_max'] - c['h_min']),
                                  lateral=c['l_mean'], height=c['h_max'], n=c['n'], confidence=conf,
                                  closing_speed=v, ttc=ttc, age=tr.age, idx=c['idx'], ego_veto=ego_veto))
        self.recent.append((self.calib.segment, self.calib.x, np.array([c['s_min'] for c in cands_all], float),
                            np.array([c['l_mean'] for c in cands_all], float)))
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
        clear = min(float(np.nanmin([horizon, nearest])) if in_g else float(horizon), cfg.report_range_cap)
        tm['tracking'] = time.perf_counter() - t3
        tm['total'] = time.perf_counter() - t0
        self._last = FrameResult(stamp=stamp, level=level, obstacles=obstacles, nearest_distance=nearest, nearest_ttc=ttc,
                           clear_distance=clear, curvature=g.curvature(20.0) if g.ok else 0.0, geometry=g,
                           timings=tm, candidates=cands, forward_points=p, zone=zone, rotation=Rm,
                           all_candidates=cands_all, features=feats)
        return self._last
