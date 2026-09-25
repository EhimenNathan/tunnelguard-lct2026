"""Run the detector directly on the organisers' synthetic-obstacle bag (zstd -> tar -> rosbag2 sqlite), streaming.
Per frame: decision, obstacles, every candidate with its evidence, lidar odometry; plus a compact near-track point cache
(int16 cm, |lateral| < 4 m, forward < 210 m) for ground-truth localisation and visualisation.
usage (scratchpad):  python stream_eval_ds3.py <archive.zst> <out_dir> [max_frames] [k=v config overrides]"""
import json
import os
import sys
import tarfile
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard'))
import numpy as np
import zstandard

from convert import iter_messages
from sqlpages import parse_pc2, split_record
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.geometry import warmup

MODEL = os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json')


def main():
    path, out = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    max_frames = int(rest[0]) if rest and rest[0].isdigit() else None
    cfg = DetectorConfig(scorer_model=MODEL)
    for kv in [a for a in rest if '=' in a]:
        k, v = kv.split('=', 1)
        obj = cfg
        for p_ in k.split('.')[:-1]:
            obj = getattr(obj, p_)
        cur = getattr(obj, k.split('.')[-1])
        setattr(obj, k.split('.')[-1], v.lower() in ('1', 'true') if isinstance(cur, bool) else type(cur)(v))
    os.makedirs(out, exist_ok=True)
    warmup()
    det = ObstacleDetector(cfg)
    raw = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(open(path, 'rb'), read_size=1 << 20)
    tf = tarfile.open(fileobj=raw, mode='r|')
    fj = open(os.path.join(out, 'frames.jsonl'), 'w')
    fc = open(os.path.join(out, 'cloud.bin'), 'wb')
    index = []
    t0 = time.time()
    n = 0
    err = None
    for m in tf:
        if not m.name.endswith('.db3'):
            continue
        try:
            for rowid, payload in iter_messages(tf.extractfile(m)):
                vals = split_record(payload)
                if len(vals) < 4 or not isinstance(vals[3], (bytes, bytearray)) or len(vals[3]) < 1000:
                    continue
                pc = parse_pc2(vals[3])
                a = np.frombuffer(pc['data'], np.float32).reshape(-1, pc['point_step'] // 4)
                xyz = a[:, :3]
                ok = np.isfinite(xyz).all(1)
                xyz = xyz[ok]
                res = det.process(xyz, pc['stamp'])
                obs = [dict(id=int(o['id']), zone=int(o['zone']), s=round(float(o['distance']), 2),
                            l=round(float(o['lateral']), 2), h=round(float(o['height']), 2), n=int(o['n']),
                            conf=round(float(o['confidence']), 3), size=[round(float(v), 2) for v in o['size']],
                            ego=bool(o.get('ego_veto', False))) for o in res.obstacles]
                cands = [dict(s=round(float(c['s_min']), 2), l=round(float(c['l_mean']), 2), h0=round(float(c['h_min']), 2),
                              h1=round(float(c['h_max']), 2), n=int(c['n']), ig=int(c.get('in_gauge_raw', c['in_gauge'])),
                              sc=round(float(c.get('score', -1)), 3), shell=bool(c['shell']), cont=bool(c['contained']),
                              grav=bool(c['gravity_fail']), shape=bool(c['shape_fail']))
                         for c in (res.all_candidates or [])]
                fj.write(json.dumps(dict(i=n, t=round(pc['stamp'], 3), level=int(res.level),
                                         clear=round(float(res.clear_distance), 1), ms=round(1e3 * res.timings['total'], 1),
                                         x=round(float(det.calib.x), 2), obs=obs, cands=cands)) + '\n')
                # compact near-track cloud in the detector's forward frame (X fwd, Y left, Z up)
                P = res.forward_points
                keep = (np.abs(P[:, 1]) < 4.0) & (P[:, 0] < 210) & (P[:, 2] > -3) & (P[:, 2] < 6)
                blob = zlib.compress(np.round(P[keep] * 100).astype('<i2').tobytes(), 6)
                index.append((fc.tell(), len(blob), int(keep.sum())))
                fc.write(blob)
                n += 1
                if n % 100 == 0:
                    fj.flush()
                    print(f'{n} frames  {time.time() - t0:.0f}s  level={res.level} x={det.calib.x:.1f}', flush=True)
                if max_frames and n >= max_frames:
                    break
        except Exception as e:
            err = repr(e)[:200]
            print('STREAM ERROR after', n, 'frames:', err, flush=True)
        break
    fj.close()
    fc.close()
    np.save(os.path.join(out, 'cloud_index.npy'), np.array(index, np.int64))
    json.dump(dict(frames=n, error=err, sec=time.time() - t0), open(os.path.join(out, 'summary.json'), 'w'))
    print('done', n, 'frames', err, round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
