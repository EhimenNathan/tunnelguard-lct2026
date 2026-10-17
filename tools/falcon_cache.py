"""Compact frame cache for fast experiments on the Falcon recording (or any rosbag2 zst of PointCloud2 frames).

build:  python falcon_cache.py build <bag.zst> <cache.bin>
Each frame: xyz as int16 in 2 cm units (+-655 m) + intensity uint8, zstd-compressed; an index (.idx.npy) holds
(offset, size, n, stamp).  read with frames_cached(path) -> (t, xyz float32 (N,3), intensity float32 (N,)).
"""
import os
import sys
import time

import numpy as np
import zstandard

Q = 0.02


def build(src, dst):
    """Resumable: frames already in the cache (index saved every 50 frames) are skipped on a rerun."""
    from falcon_stream import decode, frames
    cz = zstandard.ZstdCompressor(level=3)
    part = dst + '.idx.part.npy'
    idx = [tuple(r) for r in np.load(part)] if os.path.exists(part) and os.path.exists(dst) else []
    off = int(idx[-1][0] + idx[-1][1]) if idx else 0
    t0 = time.time()
    with open(dst, 'r+b' if idx else 'wb') as f:
        f.seek(off)
        f.truncate()
        for k, (t, pc) in enumerate(frames(open(src, 'rb'))):
            if k < len(idx):
                continue
            a = decode(pc)
            xyz = np.stack([a['x'], a['y'], a['z']], 1)
            ok = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0) & (np.abs(xyz).max(1) < 650)
            q = np.round(xyz[ok] / Q).astype(np.int16)
            inten = np.clip(a['intensity'][ok], 0, 255).astype(np.uint8)
            blob = cz.compress(q.tobytes() + inten.tobytes())
            f.write(blob)
            idx.append((off, len(blob), int(ok.sum()), pc['stamp']))
            off += len(blob)
            if k % 50 == 0:
                f.flush()
                np.save(part, np.array(idx, dtype=np.float64))
            if k % 250 == 0:
                print(f'{k} frames, {off / 1e9:.2f} GB, {time.time() - t0:.0f}s', flush=True)
    np.save(dst + '.idx.npy', np.array(idx, dtype=np.float64))
    os.remove(part)
    print('done', len(idx), 'frames', off / 1e9, 'GB')


def frames_cached(path, start=0, stop=None, step=1):
    idx = np.load(path + '.idx.npy')
    dz = zstandard.ZstdDecompressor()
    t0 = idx[0, 3]
    with open(path, 'rb') as f:
        for k in range(start, len(idx) if stop is None else min(stop, len(idx)), step):
            off, size, n, stamp = idx[k]
            f.seek(int(off))
            raw = dz.decompress(f.read(int(size)))
            n = int(n)
            xyz = np.frombuffer(raw, np.int16, count=3 * n).reshape(n, 3).astype(np.float32) * Q
            inten = np.frombuffer(raw, np.uint8, offset=6 * n, count=n).astype(np.float32)
            yield k, stamp - t0, xyz, inten


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(sys.argv[2], sys.argv[3])
