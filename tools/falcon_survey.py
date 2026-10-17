"""One pass over the Falcon bag: per-frame stats (stamp, points, forward reach) + sample frames saved as npz.
usage: python falcon_survey.py <falcon_test.zst> <out_dir> [save_every]"""
import json
import os
import sys
import time

import numpy as np

from falcon_stream import decode, frames

src, out = sys.argv[1], sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 500
os.makedirs(out, exist_ok=True)
stats, t0 = [], time.time()
for k, (t, pc) in enumerate(frames(open(src, 'rb'))):
    a = decode(pc)
    xyz = np.stack([a['x'], a['y'], a['z']], 1)
    ok = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)
    x = xyz[ok, 0]
    stats.append(dict(k=k, t=round(t, 3), stamp=pc['stamp'], n=int(ok.sum()), x99=float(np.percentile(x, 99)),
                      n100=int((x > 100).sum()), n200=int((x > 200).sum()), n250=int((x > 250).sum())))
    if k % every == 0:
        np.savez_compressed(os.path.join(out, f'frame_{k:05d}.npz'), xyz=xyz[ok].astype(np.float32),
                            intensity=a['intensity'][ok], scan_id=a['scan_id'][ok], scan_idx=a['scan_idx'][ok],
                            ts=a['timestamp'][ok], flags=a['flags'][ok], elong=a['elongation'][ok],
                            ret2=a['is_2nd_return'][ok], stamp=pc['stamp'])
    if k % 200 == 0:
        print(f'{k} t={t:.1f}s n={stats[-1]["n"]} elapsed {time.time() - t0:.0f}s', flush=True)
json.dump(stats, open(os.path.join(out, 'stats.json'), 'w'))
print('frames', len(stats), 'elapsed', round(time.time() - t0), 's')
