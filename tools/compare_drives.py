"""Compare run_drive.py outputs on the obstacle-free drive: false STOP frames and events per 300 s block
(block 3 = sealed final test), CAUTION share, latency.   usage: python compare_drives.py name=dir [name=dir ...]"""
import json
import sys

import numpy as np

rows = []
for arg in sys.argv[1:]:
    name, d = arg.split('=', 1)
    L = [json.loads(l) for l in open(f'{d}/frames.jsonl')]
    t = np.array([f['t'] for f in L])
    blk = np.minimum((t // 300).astype(int), 3)
    lev = np.array([f['level'] for f in L])
    ms = np.array([f['ms'] for f in L])
    out = dict(name=name)
    for b in range(4):
        m = blk == b
        ids = {o['id'] for f, keep in zip(L, m) if keep for o in f['obs'] if o['zone'] == 2}
        out[b] = (int((lev[m] == 2).sum()), int(m.sum()), len(ids))
    out['caution'] = float((lev == 1).mean())
    out['ms'] = float(np.median(ms))
    rows.append(out)
print(f'{"config":14s}' + ''.join(f'{"block " + str(b) + (" (sealed)" if b == 3 else ""):>28s}' for b in range(4)) +
      f'{"CAUTION":>10s}{"median ms":>11s}')
for r in rows:
    cells = "".join(f"{f'{r[b][0]:4d} fr ({100 * r[b][0] / max(r[b][1], 1):.2f}%) {r[b][2]:3d} ev':>28s}" for b in range(4))
    print(f'{r["name"]:14s}{cells}{100 * r["caution"]:9.1f}%{r["ms"]:10.0f}')
