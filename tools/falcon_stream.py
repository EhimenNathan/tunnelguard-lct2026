"""Stream the finals' Falcon (Seyond / Innovusion Falcon Prime) recording straight from Yandex Disk, frame by frame,
without downloading or extracting it: HTTPS -> zstd -> tar -> rosbag2 sqlite pages -> PointCloud2.

usage (inspection): python falcon_stream.py [max_frames] [skip]
"""
import json
import os
import sys
import tarfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import zstandard

from sqlstream import iter_rows
from sqlpages import parse_pc2, split_record

PUBLIC = os.environ.get('FALCON_URL', 'https://disk.yandex.ru/d/DK6H3Lxiyi-M-Q')
API = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key='

_DT = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def open_remote(public=PUBLIC):
    href = json.load(urllib.request.urlopen(API + public))['href']
    return urllib.request.urlopen(href, timeout=120)


def decode(pc):
    """PointCloud2 dict -> dict of named numpy arrays (all fields with count 1)."""
    names, fmts, offs = [], [], []
    for name, off, dt, cnt in pc['fields']:
        if cnt == 1 and dt in _DT:
            names.append(name); fmts.append(('>' if pc['big'] else '<') + _DT[dt]); offs.append(off)
    dtype = np.dtype({'names': names, 'formats': fmts, 'offsets': offs, 'itemsize': pc['point_step']})
    arr = np.frombuffer(pc['data'], dtype=dtype, count=len(pc['data']) // pc['point_step'])
    return {k: arr[k] for k in names}


def frames(source=None, topic_size_min=100000):
    """Yield (t since first frame [s], pc dict) for every PointCloud2 message of the bag."""
    raw = source if source is not None else open_remote()
    z = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(raw, read_size=1 << 20)
    tf = tarfile.open(fileobj=z, mode='r|')
    t0 = None
    for m in tf:
        if not m.name.endswith('.db3'):
            continue
        for _, payload in iter_rows(tf.extractfile(m)):
            vals = split_record(payload)
            if len(vals) < 4 or not isinstance(vals[3], (bytes, bytearray)) or len(vals[3]) < topic_size_min:
                continue
            pc = parse_pc2(vals[3])
            t0 = pc['stamp'] if t0 is None else t0
            yield pc['stamp'] - t0, pc
        return


if __name__ == '__main__':
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    t_start = time.time()
    for k, (t, pc) in enumerate(frames()):
        if k < skip:
            continue
        a = decode(pc)
        xyz = np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64)
        ok = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)
        xyz = xyz[ok]
        r = np.linalg.norm(xyz, axis=1)
        print(f'frame {k} t={t:.2f}s frame_id={pc["frame"]} n={len(r)} h={pc["height"]} w={pc["width"]} '
              f'step={pc["point_step"]} fields={[f[0] + ":" + str(f[2]) for f in pc["fields"]]}')
        for i, ax in enumerate('xyz'):
            q = np.percentile(xyz[:, i], [1, 50, 99])
            print(f'   {ax}: p1={q[0]:8.2f} med={q[1]:8.2f} p99={q[2]:8.2f}  max|.|={np.abs(xyz[:, i]).max():7.1f}')
        print(f'   range p50={np.median(r):.1f} p99={np.percentile(r, 99):.1f} max={r.max():.1f}  '
              f'elapsed {time.time() - t_start:.1f}s', flush=True)
        if k + 1 >= skip + nmax:
            break
