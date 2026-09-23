"""Unit and end-to-end tests of the detector core (no ROS needed):  python -m pytest src/tunnel_guard/test"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from tunnel_guard.core.cloud import AxisDetector, cloud_to_arrays, forward_rotation  # noqa: E402
from tunnel_guard.core.detector import DetectorConfig, ObstacleDetector, LEVEL_CLEAR, LEVEL_STOP  # noqa: E402
from tunnel_guard.core.gauge import GaugeConfig, IN_GAUGE, OUTSIDE, classify  # noqa: E402
from tunnel_guard.core.geometry import GeometryEstimator  # noqa: E402
from tunnel_guard.core.synth import SHAPES, ray_hits  # noqa: E402
from tunnel_guard.core.tracker import Tracker  # noqa: E402
from tunnel_guard import params as P  # noqa: E402
from tunnel_sim import make_tunnel, centreline  # noqa: E402


def test_cloud_decode_roundtrip():
    n = 100
    dt = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring', 'timestamp'],
                   'formats': ['<f4', '<f4', '<f4', '<f4', '<u2', '<f8'],
                   'offsets': [0, 4, 8, 12, 16, 18], 'itemsize': 26})
    a = np.zeros(n, dt)
    a['x'] = np.arange(n)
    a['ring'] = np.arange(n) % 128
    fields = [('x', 0, 7, 1), ('y', 4, 7, 1), ('z', 8, 7, 1), ('intensity', 12, 7, 1), ('ring', 16, 4, 1), ('timestamp', 18, 8, 1)]
    out = cloud_to_arrays(fields, a.tobytes(), 26)
    assert np.allclose(out['x'], np.arange(n))
    assert np.array_equal(out['ring'], np.arange(n) % 128)


def test_axis_detection_and_rotation():
    fwd = make_tunnel()
    sensor = fwd @ forward_rotation('-y')           # forward frame -> sensor frame with forward = -y
    det = AxisDetector(frames=1)
    assert det.update(sensor) == '-y'
    back = sensor @ forward_rotation('-y').T
    assert np.allclose(back, fwd, atol=1e-4)


def test_geometry_straight_and_curved():
    for radius, tol in ((None, 0.10), (600.0, 0.35)):
        est = GeometryEstimator()
        for _ in range(3):
            g = est.estimate(make_tunnel(radius=radius, grade=0.01))
        for x in (10.0, 60.0, 120.0):
            truth = centreline(np.array([x]), radius)[0]
            assert abs(g.centre(x) - truth) < tol, (radius, x, g.centre(x), truth)
            assert abs(g.rail_z(x) - (-1.15 + 0.01 * x)) < 0.15, (x, g.rail_z(x))
        assert g.valid_range > 120


def test_gauge_zones():
    cfg = GaugeConfig()
    x = np.array([50.0, 50.0, 50.0, 50.0])
    l = np.array([0.0, 3.0, 0.0, 1.0])
    h = np.array([1.0, 1.0, 0.02, 0.3])
    z = classify(x, l, h, cfg)
    assert z[0] == IN_GAUGE and z[1] == OUTSIDE and z[2] == OUTSIDE and z[3] == IN_GAUGE


def test_tracker_confirms_persistent_and_drops_flicker():
    tr = Tracker()
    for k in range(4):
        c = dict(s_min=100.0 - 1.5 * k, l_mean=0.0)
        tracks = tr.update([c], 0.1 * k)
    assert any(t.confirmed for t in tracks)
    tr = Tracker()
    tr.update([dict(s_min=80.0, l_mean=0.0)], 0.0)
    for k in range(1, 4):
        tracks = tr.update([], 0.1 * k)
    assert not any(t.confirmed for t in tracks)


def test_ray_hits_box():
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t = ray_hits(dirs, np.array([10.0, 0.0, -0.5]), SHAPES['box_50cm'])
    assert abs(t[0] - 9.75) < 1e-6 and np.isinf(t[1])


def test_end_to_end_clear_and_obstacle():
    cfg = DetectorConfig(forward_axis='x')
    det = ObstacleDetector(cfg)
    for k in range(5):
        res = det.process(make_tunnel(radius=800.0, seed=k), 0.1 * k)
    assert res.level == LEVEL_CLEAR, [(o['distance'], o['lateral'], o['height']) for o in res.obstacles]
    for k in range(5, 10):
        res = det.process(make_tunnel(radius=800.0, seed=k, obstacles=[(90.0 - 1.0 * (k - 5), 0.0, 0.6, 0.6, 1.0)]), 0.1 * k)
    assert res.level == LEVEL_STOP
    assert abs(res.nearest_distance - 86.0) < 2.0


def test_params_roundtrip():
    cfg = P.default_config()
    flat = P.flatten(cfg)
    assert 'gauge.profile' in flat and 'geometry.x_max' in flat
    P.apply(cfg, 'gauge.h_min', 0.2)
    P.apply(cfg, 'gauge.profile', [0.0, 1.0, 1.0, 1.0, 3.0, 1.4])
    assert cfg.gauge.h_min == 0.2 and cfg.gauge.profile == ((0.0, 1.0), (1.0, 1.0), (3.0, 1.4))


def test_detector_from_ros_style_parameters():
    """Parameters arrive from ROS flattened (lists of doubles, ints as given); the detector must run with them."""
    cfg = P.default_config()
    for name, value in P.flatten(cfg).items():
        if isinstance(value, list):
            value = [float(v) for v in value]
        P.apply(cfg, name, value)
    det = ObstacleDetector(cfg)
    cfg.forward_axis = 'x'
    res = det.process(make_tunnel(radius=None, seed=1), 0.0)
    assert res.geometry.ok


def test_deployed_scorer_loads_and_is_sane():
    from tunnel_guard.core.scorer import ObstacleScorer
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'obstacle_scorer.json')
    sc = ObstacleScorer(path)
    names = sc.features
    assert 0.0 < sc.threshold < 1.0
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, len(names)))
    p = sc.prob(X)
    assert p.shape == (64,) and np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    # physics sanity: more in-gauge evidence (all else equal) must not lower the score (monotone members)
    base = np.median(X, 0)
    lo, hi = base.copy(), base.copy()
    lo[names.index('in_gauge')], hi[names.index('in_gauge')] = 0.0, 40.0
    assert sc.logit(hi[None])[0] >= sc.logit(lo[None])[0] - 1e-6


def test_pandar128_channel_table_and_far_field_coverage():
    from tunnel_guard.core.pandar import Pandar128
    m = Pandar128(os.path.join(os.path.dirname(__file__), '..', 'config', 'pandar128_channels.csv'))
    assert len(m.channel) == 128 and m.far_field.sum() == 32 and m.high_res.sum() == 65
    assert abs(m.elevation_deg[41]) < 1e-6                       # channel 42 is the horizontal beam
    far = m.elevation_deg[m.far_field]
    # from 80 m to 200 m the corridor (rail plane .. 2 m above it) seen from 1.1-1.7 m above the rail lies in the
    # 200 m far-field channels
    for h in (1.1, 1.7):
        for s in (80.0, 200.0):
            lo, hi = np.degrees(np.arctan2(-h, s)), np.degrees(np.arctan2(2.0 - h, s))
            assert far.min() <= lo and hi <= far.max()
    assert np.allclose(m.vertical_spacing_deg(-1.0), 0.125, atol=0.01)
    assert m.max_instrumented.max() == 200.0


def test_synthetic_returns_respect_instrumented_range():
    from tunnel_guard.core.synth import SHAPES, channel_reach
    reach, cap = channel_reach(0.9, np.arange(128), 128)          # very bright target
    assert np.all(reach <= cap) and cap.max() <= 210.0 + 1e-9


def test_ego_motion_consistency_veto():
    """A solid object closes in at the train's speed; an artefact travelling with the train keeps its distance."""
    from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
    from tunnel_guard.core.tracker import Track
    det = ObstacleDetector(DetectorConfig(scorer_model='none'))
    rng = np.random.default_rng(1)

    def track(xs, ss):
        t = Track(1, ss[-1], 0.0, last=dict(s_min=ss[-1], l_mean=0.0))
        t.obs = [(det.calib.segment, x, s + rng.normal(0, 0.5)) for x, s in zip(xs, ss)]
        return t

    x = 1.2 * np.arange(8)                                     # train advancing 12 m/s at 10 Hz
    assert not det._carried_along(track(x, 60.0 - x))          # static obstacle: slope -1 -> kept
    assert det._carried_along(track(x, 25.0 + 0 * x))          # carried along: slope 0 -> vetoed
    assert not det._carried_along(track(x, 60.0 - 0.6 * x))    # object approaching slower than the train: kept
    x0 = np.zeros(8)                                           # train standing: no evidence, never vetoed
    assert not det._carried_along(track(x0, 25.0 + 0 * x0))
    t = track(x, 25.0 + 0 * x)
    det.calib.segment += 1                                     # odometry chain broken: history unusable
    assert not det._carried_along(t)


def test_ego_motion_neighbourhood_test_for_young_tracks():
    """A fragmenting artefact spawns new tracks; its neighbourhood shows it travelling with the train."""
    from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
    from tunnel_guard.core.tracker import Track
    det = ObstacleDetector(DetectorConfig(scorer_model='none'))
    young = Track(1, 25.0, 0.8, last=dict(s_min=25.0, l_mean=0.8))
    young.obs = [(0, 12.0, 25.0)]
    seg = det.calib.segment

    def history(s_of_k):
        det.recent.clear()
        for k in range(10):                                   # train at 1.2 m/frame, now at x = 12 m
            det.recent.append((seg, 1.2 * k, np.array([s_of_k(k)]), np.array([0.8])))
        det.calib.x = 12.0

    history(lambda k: 25.0)                                   # something always 25 m ahead: carried along
    assert det._carried_along(young)
    history(lambda k: 25.0 + 12.0 - 1.2 * k)                  # a static object seen earlier further away
    assert not det._carried_along(young)
    history(lambda k: 90.0)                                   # nothing nearby earlier (object just appeared)
    assert not det._carried_along(young)
