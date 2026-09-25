"""Stream the organisers' synthetic-obstacle bag (zstd -> tar -> rosbag2 sqlite) and report message layout; optionally
count all readable messages to locate stream corruption.   usage: python peek_ds3.py <archive.zst> [count]"""
import sys
import tarfile
import time

import numpy as np
import zstandard

from convert import iter_messages
from sqlpages import parse_pc2, split_record

path = sys.argv[1]
count_all = len(sys.argv) > 2
raw = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(open(path, 'rb'), read_size=1 << 20)
tf = tarfile.open(fileobj=raw, mode='r|')
t0 = time.time()
for m in tf:
    if not m.name.endswith('.db3'):
        continue
    f = tf.extractfile(m)
    n, last = 0, None
    try:
        for rowid, payload in iter_messages(f):
            vals = split_record(payload)
            if len(vals) < 4 or not isinstance(vals[3], (bytes, bytearray)) or len(vals[3]) < 1000:
                continue
            pc = parse_pc2(vals[3])
            n += 1
            last = (rowid, vals[2], pc['stamp'])
            if n <= 2:
                print('msg', n, 'rowid', rowid, 'stamp', pc['stamp'], 'frame', pc['frame'], 'h x w', pc['height'], pc['width'],
                      'point_step', pc['point_step'], 'fields', pc['fields'], 'dense', pc['dense'], flush=True)
                a = np.frombuffer(pc['data'], np.float32).reshape(-1, pc['point_step'] // 4)
                r = np.linalg.norm(a[:, :3], axis=1)
                print('   points', len(a), 'valid(r>0.3)', int((r > 0.3).sum()), 'max range %.1f' % r.max(),
                      'x %.1f..%.1f y %.1f..%.1f z %.2f..%.2f' % (a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max(),
                                                               a[:, 2].min(), a[:, 2].max()), flush=True)
            if not count_all and n >= 2:
                break
            if n % 200 == 0:
                print(n, 'messages', round(time.time() - t0), 's', flush=True)
    except Exception as e:
        print('STREAM ERROR after', n, 'messages; last', last, ':', repr(e)[:120], flush=True)
    print('readable messages:', n, 'last', last, round(time.time() - t0), 's', flush=True)
    break
