"""Stream the zstd tar of rosbag2 sqlite bags -> compact per-bag range-image caches.
No sqlite needed: pages are parsed forward-only (overflow chains are contiguous in rosbag2 files)."""
import sys, os, struct, zlib, json, time
import numpy as np
from sqlpages import parse_leaf, split_record, parse_pc2, U

def iter_messages(stream):
    """Yield (rowid, payload bytes) from a forward-only stream of a sqlite db file.
    Overflow pages are kept in a rolling store (pieces of contiguous page runs) and
    pruned once a later chain has been consumed (rosbag2 chains are allocated in order)."""
    pno = 0; pieces = []; pending = b''
    LOCK = (1 << 30) // U + 1   # sqlite lock-byte page: never holds data, chains skip it
    def gather(ov, need):
        npages = -(-need // (U - 4)); end = ov + npages + (1 if ov <= LOCK < ov + npages + 1 else 0)
        out = []; p = ov
        for st, arr in pieces:
            if p >= st + len(arr) or p < st: continue
            k0 = p - st; take = min(len(arr) - k0, end - p)
            seg = arr[k0:k0 + take]
            if st + k0 <= LOCK < st + k0 + take: seg = np.delete(seg, LOCK - (st + k0), axis=0)
            out.append(seg); p += take
            if p >= end: break
        if p < end: return None
        body = np.concatenate(out).ravel() if len(out) > 1 else out[0].ravel()
        return body[:need].tobytes()
    while True:
        chunk = stream.read(1 << 24)
        if not chunk: break
        chunk = pending + chunk; npg = len(chunk) // U; pending = chunk[npg * U:]
        first = np.frombuffer(chunk, np.uint8, count=npg * U)[::U].copy()
        if pno == 0: first[0] = chunk[100]
        pages = np.frombuffer(chunk, np.uint8, count=npg * U).reshape(npg, U)
        leafs = np.nonzero(first == 0x0d)[0]; s = 0
        for li in list(leafs) + [npg]:
            if li > s: pieces.append((pno + s + 1, pages[s:li, 4:]))
            if li == npg: break
            for rowid, P, loc, ov in parse_leaf(memoryview(chunk)[li * U:(li + 1) * U], pno + li + 1):
                if ov == 0: yield rowid, loc; continue
                body = gather(ov, P - len(loc))
                if body is None: print('WARN unmatched chain', rowid, ov, file=sys.stderr); continue
                yield rowid, loc + body
                pieces = [(st, arr) for st, arr in pieces if st + len(arr) > ov]
            s = li + 1
        pno += npg

DT = {1:'i1',2:'u1',3:'i2',4:'u2',5:'i4',6:'u4',7:'f4',8:'f8'}

def convert_bag(stream, out_prefix, max_frames=None, keep_intensity=True):
    fbin = open(out_prefix + '.bin', 'wb'); index = []; dirs_sum = None; col_dt = None; meta = {}
    t0 = time.time()
    for rowid, payload in iter_messages(stream):
        vals = split_record(payload)
        if len(vals) < 4 or not isinstance(vals[3], (bytes, bytearray)) or len(vals[3]) < 1000: continue
        pc = parse_pc2(vals[3])
        dt = np.dtype({'names': [f[0] for f in pc['fields']], 'formats': [DT[f[2]] for f in pc['fields']],
                       'offsets': [f[1] for f in pc['fields']], 'itemsize': pc['point_step']})
        a = np.frombuffer(pc['data'], dtype=dt); N = len(a); W = N // 128
        x = a['x'].reshape(W, 128); y = a['y'].reshape(W, 128); z = a['z'].reshape(W, 128)
        r = np.sqrt(x.astype(np.float32)**2 + y**2 + z**2)
        rng = np.clip(np.round(r * 100), 0, 65535).astype(np.uint16)
        inten = np.clip(a['intensity'].reshape(W, 128), 0, 255).astype(np.uint8)
        if dirs_sum is None:
            dirs_sum = np.zeros((W, 128, 3), np.float64); cnt = np.zeros((W, 128))
            meta = dict(W=W, frame_id=pc['frame'], fields=[f[0] for f in pc['fields']])
            ts = a['timestamp'].reshape(W, 128)
            col_dt = (ts[:, 0] - pc['stamp']).astype(np.float32)
        v = (r > 0.3) & (cnt < 3)
        if v.any():
            dirs_sum[v] += np.stack([x[v] / r[v], y[v] / r[v], z[v] / r[v]], 1); cnt[v] += 1
        blob = zlib.compress(rng.tobytes() + (inten.tobytes() if keep_intensity else b''), 6)   # intensity is not used by any decision
        index.append((fbin.tell(), len(blob), vals[2], pc['stamp'], rowid))
        fbin.write(blob)
        if max_frames and len(index) >= max_frames: break
    fbin.close()
    with np.errstate(invalid='ignore'):
        dirs = (dirs_sum / np.maximum(cnt, 1)[..., None])
        dirs /= np.maximum(np.linalg.norm(dirs, axis=2, keepdims=True), 1e-9)
    np.savez(out_prefix + '_meta.npz', dirs=dirs.astype(np.float32), dircnt=cnt.astype(np.uint16), col_dt=col_dt,
             index=np.array([(i[0], i[1], i[2], i[4]) for i in index], np.int64), stamps=np.array([i[3] for i in index]))
    meta['n'] = len(index); meta['sec'] = time.time() - t0
    json.dump(meta, open(out_prefix + '_meta.json', 'w'))
    return meta

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        with open(sys.argv[2], 'rb') as f: print(convert_bag(f, sys.argv[3], max_frames=int(sys.argv[4])))
    else:
        import zstandard, tarfile
        p = os.environ.get('HACKATHON_ZST', 'for_hackathon.zst')   # path to the organisers' archive
        with open(p, 'rb') as fh:
            rd = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh, read_size=1 << 20)
            tf = tarfile.open(fileobj=rd, mode='r|')
            for m in tf:
                if m.name.endswith('.db3'):
                    name = m.name.split('/')[1]
                    print('converting', name, flush=True)
                    print(name, convert_bag(tf.extractfile(m), 'cache2/' + name), flush=True)
