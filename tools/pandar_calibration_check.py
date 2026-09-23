"""Check the recorded lidar against the official Pandar128 calibration files (angle correction, firetime correction).

  python pandar_calibration_check.py "<radar specification dir>"      (run from the scratchpad with the range-image cache)

Reports: elevation / azimuth-offset agreement per channel, the 0.125 deg fine band, its coverage of the track corridor
at 40-200 m, firetime magnitudes and the sweep time of the forward sector (motion-smear bound).
"""
import json
import os
import re
import sys

import numpy as np

SPEC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'radar specification')


def angle_table(path):
    rows = []
    for line in open(path, 'rb').read().split(b'\n')[1:]:
        parts = line.strip().split(b',')
        if len(parts) >= 3:
            rows.append((int(re.match(rb'\d+', parts[0]).group()), float(parts[1]), float(parts[2])))   # "42 (horizontal)"
    return np.array(sorted(rows))


def firetimes(path):
    rows = [l.strip().split(',') for l in open(path, encoding='utf-8-sig')][3:131]
    return np.array([[float(x) for x in r[1:17]] for r in rows])


def wrap(a):
    return (a + 180) % 360 - 180


cal = angle_table(os.path.join(SPEC, 'Pandar128_Angle_Correction_File-2.csv'))
el_cal, az_cal = cal[:, 1], cal[:, 2]
ft = firetimes(os.path.join(SPEC, 'Pandar128_Firetime_Correction_File-1.csv'))
spacing = np.abs(np.diff(el_cal))
fine = el_cal[:-1][np.abs(spacing - 0.125) < 0.01]
out = dict(channels=int(len(cal)), elevation_range=[float(el_cal.min()), float(el_cal.max())],
           fine_band_deg=[float(fine.min()), float(fine.max())], fine_band_channels=int(len(fine)),
           firetime_max=float(ft.max()))
for nm in ['roundT_doubleT', 'doubleT_obstacle']:
    m = np.load(f'cache/{nm}_meta.npz')
    meta = json.load(open(f'cache/{nm}_meta.json'))
    rd, cnt = m['dirs'].astype(float), m['dircnt'] > 0
    el = np.degrees(np.arcsin(np.clip(rd[..., 2], -1, 1)))
    az = np.degrees(np.arctan2(rd[..., 0], -rd[..., 1]))
    el_ring = np.array([np.median(el[:, r][cnt[:, r]]) for r in range(128)])
    ref = np.array([np.angle(np.nanmean(np.where(cnt[c], np.exp(1j * np.radians(az[c])), np.nan)), deg=True)
                    if cnt[c].any() else np.nan for c in range(az.shape[0])])
    off = np.array([np.median(wrap(az[:, r] - ref)[cnt[:, r] & np.isfinite(ref)]) for r in range(128)])
    de = el_ring - el_cal
    da = wrap(-off - np.median(-off) - (az_cal - np.median(az_cal)))        # recorded azimuth runs clockwise
    col_dt = m['col_dt']
    span = float(np.nanmax(col_dt) - np.nanmin(col_dt))
    out[nm] = dict(elevation_offset_deg=float(np.median(de)), elevation_residual_max_deg=float(np.abs(de - np.median(de)).max()),
                   azimuth_residual_median_deg=float(np.median(np.abs(da))), azimuth_residual_max_deg=float(np.abs(da).max()),
                   columns=int(meta['W']), sweep_ms=1e3 * span)
corridor = {}
for hgt in (1.1, 1.7):
    for s in (40, 200):
        corridor[f'h{hgt}_s{s}'] = [float(np.degrees(np.arctan2(-hgt, s))), float(np.degrees(np.arctan2(2 - hgt, s)))]
out['corridor_elevation_deg'] = corridor
print(json.dumps(out, indent=1))
