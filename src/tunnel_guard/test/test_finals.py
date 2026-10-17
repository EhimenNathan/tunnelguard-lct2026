"""Регрессии финала: изоляция сенсоров, ложные рельсы, эквивалентность C++."""
import numpy as np
import pytest

from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.geometry import _track_dp, GAUGE_CC


def test_native_falcon_axes_and_already_converted_ros_axes():
    from tunnel_guard.core.cloud import AxisDetector, forward_rotation
    rng = np.random.default_rng(4)
    ros = np.column_stack([rng.uniform(50, 300, 2000), rng.uniform(-3, 3, 2000),
                           rng.uniform(-2, 4, 2000)])
    R = forward_rotation('falcon')
    raw = ros @ R
    np.testing.assert_allclose(raw @ R.T, ros)
    assert np.linalg.det(R) == pytest.approx(1.)
    for points, expected in ((ros, 'x'), (raw, 'z')):
        axis = AxisDetector(frames=2)
        axis.update(points)
        assert axis.update(points) == expected


def test_sensor_density_is_per_detector_and_reset_preserves_config():
    a = ObstacleDetector(DetectorConfig(scorer_model='none'))
    a.sensor_scale = 0.7
    a.cfg.cluster.beam_vertical_rad = 0.003
    a.score_hist[99] = [0.99]
    b = ObstacleDetector(DetectorConfig(scorer_model='none'))
    assert a.sensor_scale == 0.7 and b.sensor_scale == 1.0
    a.reset()
    assert a.sensor_scale == 1.0
    assert a.cfg.cluster.beam_vertical_rad == 0.003
    assert a.score_hist == {}


def test_rail_continuity_rejects_stronger_isolated_gauge_pair():
    dy, lim = 0.03, 3.0
    R = np.zeros((60, 200))
    half = int(round(GAUGE_CC / 2 / dy))
    ys = -lim + (np.arange(200) + 0.5) * dy
    # Continuous curved true track; stronger rail-like artefact only occupies four rows.
    true = 0.00012 * np.arange(60) ** 2
    centres = np.argmin(abs(ys[None] - true[:, None]), axis=1)
    for row, c in enumerate(centres):
        R[row, c-half] = R[row, c+half] = 0.12
    fake = int(np.argmin(abs(ys + 1.25)))
    R[25:29, fake-half] = R[25:29, fake+half] = 0.2
    found, _, _ = _track_dp(R, dy, lim, prior_c=true)
    assert np.max(abs(found - true)) < 0.09


def test_cpp_geometry_matches_reference_and_rejects_bad_indices():
    native = pytest.importorskip('tunnel_guard.core._geometry_native')
    from tunnel_guard.core.geometry import _gn_numba
    rng = np.random.default_rng(2026)
    n, N = 80, 4000
    k = rng.integers(0, n-1, N, dtype=np.int64)
    args = [rng.uniform(2, 200, N), rng.uniform(-6, 6, N), rng.uniform(-3, 6, N), k,
            rng.random(N), rng.random(N), rng.normal(0, .2, n), np.full(n, -1.8),
            rng.normal(0, .05, n), .02, rng.uniform(0, .6, (50, 60)), .5,
            -8., -2.5, .2, 1.1, .3, 3.2, .8, 1.2, .3, 3.7, n]
    ref, cpp = _gn_numba(*args), native.gn(*args)
    for a, b in zip(ref, cpp):
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)
    args[3] = np.full(N, n-1, dtype=np.int64)
    with pytest.raises(ValueError):
        native.gn(*args)
