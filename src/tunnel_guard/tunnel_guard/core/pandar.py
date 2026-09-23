"""Hesai Pandar128E3X channel model from the manufacturer's documentation.

config/pandar128_channels.csv was extracted from the user manual (v4.5, Appendix A "Channel distribution data"):
per channel the elevation, horizontal offset, instrumented range and the range at 10 % reflectivity (probability of
detection 70 %; values in brackets in the manual are capped at the instrumented range).  The angles agree with the
recorded lidar to 0.06 deg in elevation and 0.08 deg in azimuth offset (tools/pandar_calibration_check.py).

Key facts used by the detector and the evaluation:
  * channels 26-90 are high-resolution (0.125 deg vertical spacing, elevation -6.1 .. +2.0 deg);
  * only channels 34-65 (elevation -2.9 .. +1.0 deg) reach 200 m at 10 % reflectivity; the other high-resolution
    channels reach 140 m, the upper channels 100 m and the ground-facing channels 25-100 m;
  * the instrumented (time-of-flight) range of the high-resolution channels is 200 m (recordings: max 209 m) - no
    return beyond that can exist, whatever the reflectivity;
  * firing-time offsets inside a block are at most 55 microseconds (Appendix B.4, unit ns): negligible motion blur.
"""
import csv
import os

import numpy as np

_DEFAULT = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'pandar128_channels.csv')


def _find_table():
    candidates = [_DEFAULT]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(get_package_share_directory('tunnel_guard'), 'config', 'pandar128_channels.csv'))
    except Exception:
        pass
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError('pandar128_channels.csv not found')


class Pandar128:
    def __init__(self, path=None):
        rows = list(csv.DictReader(open(path or _find_table())))
        rows.sort(key=lambda r: int(r['channel']))
        self.channel = np.array([int(r['channel']) for r in rows])
        self.elevation_deg = np.array([float(r['elevation']) for r in rows])
        self.azimuth_offset_deg = np.array([float(r['azimuth_offset']) for r in rows])
        self.range_10pct = np.array([float(r['effective_range_at_10pct']) for r in rows])
        self.max_instrumented = np.array([float(r['max_instrumented']) for r in rows])
        self.far_field = np.array([r['far_field_enhanced'] == 'True' for r in rows])
        self.high_res = np.array([r['high_res'] == 'True' for r in rows])

    def vertical_spacing_deg(self, elevation_deg):
        """Local vertical beam spacing at the given elevation(s)."""
        el = self.elevation_deg[::-1]                     # ascending
        gaps = np.diff(el)
        idx = np.clip(np.searchsorted(el, np.asarray(elevation_deg, float)) - 1, 0, len(gaps) - 1)
        return gaps[idx]

    def max_range(self, reflectivity, channel_index=None):
        """Range at which the probability of detection drops to 70 % (range scales with sqrt(reflectivity))."""
        r10 = self.range_10pct if channel_index is None else self.range_10pct[channel_index]
        return r10 * np.sqrt(np.asarray(reflectivity, float) / 0.10)


_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = Pandar128()
    return _MODEL
