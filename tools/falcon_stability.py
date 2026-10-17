"""Envelope stability on the whole Falcon recording: per-frame track estimate at fixed distances, envelope range and
clear distance; frame-to-frame jitter, flicker (envelope range changes) and drop-outs (frames without geometry).
usage: python falcon_stability.py <cache.bin> <out.json>"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard'))
import numpy as np

from falcon_cache import frames_cached
from tunnel_guard.core.detector import DetectorConfig, ObstacleDetector

XS = np.array([25.0, 50, 100, 150, 200, 250, 300])


def main(cache, out):
    det = ObstacleDetector(DetectorConfig(scorer_model=os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard',
                                                                    'config', 'obstacle_scorer.json')))
    Y, Z, ENV, CL, OK = [], [], [], [], []
    for k, t, xyz, inten in frames_cached(cache):
        res = det.process(xyz, t, intensity=inten)
        g = res.geometry
        OK.append(bool(g.ok))
        Y.append(g.centre(XS).tolist() if g.ok else [np.nan] * len(XS))
        Z.append(g.rail_z(XS).tolist() if g.ok else [np.nan] * len(XS))
        ENV.append(float(res.envelope_range))
        CL.append(float(res.clear_distance))
    Y, Z, ENV, CL, OK = map(np.array, (Y, Z, ENV, CL, OK))
    s = 30                                  # after start-up (mount calibration + settling)
    rep = dict(frames=len(OK), geometry_dropouts=int((~OK[s:]).sum()),
               envelope_range_median=float(np.median(ENV[s:])), envelope_range_p10=float(np.percentile(ENV[s:], 10)),
               envelope_range_frame_change_median=float(np.median(np.abs(np.diff(ENV[s:])))),
               envelope_range_frames_changed_over_5m=float(np.mean(np.abs(np.diff(ENV[s:])) > 5)),
               clear_median=float(np.median(CL[s:])))
    for i, x in enumerate(XS):
        y, z = Y[s:, i], Z[s:, i]
        dy, dz = np.abs(np.diff(y)), np.abs(np.diff(z))
        rep[f'centre_{int(x)}m'] = dict(jitter_median_cm=float(100 * np.nanmedian(dy)), jitter_p95_cm=float(100 * np.nanpercentile(dy, 95)),
                                        std_cm=float(100 * np.nanstd(y)), mean_offset_m=float(np.nanmean(y)),
                                        rail_jitter_p95_cm=float(100 * np.nanpercentile(dz, 95)), rail_std_cm=float(100 * np.nanstd(z)))
    json.dump(rep, open(out, 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
