"""Evaluate the deployed detector directly on a (zip ->) zstd -> tar archive of a split rosbag2 recording, without
extracting or caching it (the new 20-minute recording is 90 GB raw).

The pieces of a split bag are stored in random order in the tar, so every 5 s piece is processed independently with a
fresh detector: its first WARM frames rebuild the track geometry and are not scored.  Output (small):
  <out>/frames.jsonl   one line per frame: piece, frame, stamp, level, nearest, clear, latency, obstacles
  <out>/snaps/*.png    rendered view of the first confirmed-STOP frame of every obstacle track (for manual review)
  <out>/summary.json
usage (from the scratchpad):  python stream_eval.py <bags.zip> <out_dir> [max_pieces]
"""
import json
import os
import shutil
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard', 'tools'))
import numpy as np

from convert import iter_messages, DT
from sqlpages import split_record, parse_pc2
from list_archive import open_zst_stream
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.geometry import warmup

MODEL = os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json')
WARM = 8
MAX_SNAPS = 80


def wait_for_disk(min_free=400e6):
    while shutil.disk_usage(os.path.abspath(os.sep)).free < min_free:
        print('low disk space - waiting', flush=True)
        time.sleep(30)


def cloud(payload):
    vals = split_record(payload)
    if len(vals) < 4 or not isinstance(vals[3], (bytes, bytearray)) or len(vals[3]) < 1000:
        return None
    pc = parse_pc2(vals[3])
    dt = np.dtype({'names': [f[0] for f in pc['fields']], 'formats': [DT[f[2]] for f in pc['fields']],
                   'offsets': [f[1] for f in pc['fields']], 'itemsize': pc['point_step']})
    a = np.frombuffer(pc['data'], dtype=dt)
    xyz = np.stack([a['x'], a['y'], a['z']], 1).astype(np.float32)
    ok = np.isfinite(xyz).all(1) & (np.einsum('ij,ij->i', xyz, xyz) > 0.25)
    inten = a['intensity'][ok].astype(np.float32) if 'intensity' in a.dtype.names else None
    return pc['stamp'], xyz[ok], inten


def main():
    src, out = sys.argv[1], sys.argv[2]
    max_pieces = int(sys.argv[3]) if len(sys.argv) > 3 else None
    os.makedirs(os.path.join(out, 'snaps'), exist_ok=True)
    warmup()
    render = None
    fout = open(os.path.join(out, 'frames.jsonl'), 'a')
    done = set()
    if os.path.exists(os.path.join(out, 'pieces_done.txt')):
        done = set(open(os.path.join(out, 'pieces_done.txt')).read().split())
    snaps = len(os.listdir(os.path.join(out, 'snaps')))
    t0 = time.time()
    n_pieces = 0
    tf = tarfile.open(fileobj=open_zst_stream(src), mode='r|')
    for m in tf:
        if not m.name.endswith('.db3'):
            continue
        piece = os.path.basename(m.name)[:-4]
        if piece in done:
            continue
        wait_for_disk()
        det = ObstacleDetector(DetectorConfig(scorer_model=MODEL))
        stopped_tracks = set()
        k = 0
        for rowid, payload in iter_messages(tf.extractfile(m)):
            c = cloud(payload)
            if c is None:
                continue
            stamp, xyz, inten = c
            res = det.process(xyz, stamp, intensity=inten)
            if k >= WARM:
                obs = [dict(id=int(o['id']), zone=int(o['zone']), d=round(float(o['distance']), 1),
                            lat=round(float(o['lateral']), 2), h=round(float(o['height']), 2), n=int(o['n']),
                            conf=round(float(o['confidence']), 2)) for o in res.obstacles]
                fout.write(json.dumps(dict(piece=piece, k=k, t=round(stamp, 3), level=int(res.level),
                                           nearest=None if np.isnan(res.nearest_distance) else round(float(res.nearest_distance), 1),
                                           clear=round(float(res.clear_distance), 1), ms=round(1e3 * res.timings['total'], 1),
                                           npts=int(len(xyz)), obs=obs)) + '\n')
                for o in res.obstacles:
                    if o['zone'] == 2 and o['id'] not in stopped_tracks and snaps < MAX_SNAPS:
                        stopped_tracks.add(o['id'])
                        if render is None:
                            from render_demo import render_frame
                            import imageio.v2 as imageio
                            render = (render_frame, imageio)
                        img = render[0](res, det.cfg.gauge, info=dict(
                            chapter=f'{piece} · кадр {k} · трек {o["id"]}', subtitle='новая запись (dataset2)',
                            clock=f't = {stamp:.1f}', points=int(len(xyz))))
                        render[1].imwrite(os.path.join(out, 'snaps', f'{piece}_k{k:02d}_id{o["id"]}.png'), img[::2, ::2])
                        snaps += 1
            k += 1
        fout.flush()
        with open(os.path.join(out, 'pieces_done.txt'), 'a') as f:
            f.write(piece + '\n')
        n_pieces += 1
        print(f'{piece}: {k} frames  [{n_pieces} pieces, {time.time() - t0:.0f}s]', flush=True)
        if max_pieces and n_pieces >= max_pieces:
            break
    fout.close()
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
