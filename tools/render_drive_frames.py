"""Render selected moments of a cached drive with the full HUD (warm detector state).
usage (scratchpad):  python render_drive_frames.py cache/ds2 out_dir t1 t2 ...   (times in seconds from the start)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard', 'tools'))
sys.path.insert(0, os.getcwd())
import numpy as np
import imageio.v2 as imageio

from run_drive import DriveCache, MODEL
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.geometry import warmup
from render_demo import render_frame

root, out = sys.argv[1], sys.argv[2]
times = [float(v) for v in sys.argv[3:]]
os.makedirs(out, exist_ok=True)
drive = DriveCache(root)
warmup()
for tt in times:
    i = int(np.argmin(np.abs(drive.t - tt)))
    det = ObstacleDetector(DetectorConfig(scorer_model=MODEL))
    for j in range(max(0, i - 60), i + 1):
        xyz, inten = drive.cloud(j)
        res = det.process(xyz, drive.t[j], intensity=inten)
    img = render_frame(res, det.cfg.gauge, info=dict(chapter=f't = {drive.t[i]:.1f} s · кадр {i}', subtitle='dataset2',
                                                  clock=f'{drive.names[drive.index[i][0]]}', points=int(len(xyz))))
    imageio.imwrite(os.path.join(out, f't{drive.t[i]:07.1f}.png'), img[::2, ::2])
    cands = [(round(c['s_min'], 1), round(c['l_mean'], 2), round(c['h_max'], 2), c['n'], c['in_gauge_raw'], c.get('contained'),
              c.get('shell'), round(c.get('score', -1), 2)) for c in (res.all_candidates or []) if c['s_min'] < 60]
    print(f't={drive.t[i]:.1f} level={res.level} clear={res.clear_distance:.0f} near candidates (s,l,hmax,n,in_raw,contained,shell,score):',
          sorted(cands)[:8], flush=True)
