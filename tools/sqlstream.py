"""Forward-only reader of rosbag2 sqlite rows from a stream (no sqlite, no extraction, bounded memory).

Rows are read from table-leaf pages; large payloads continue in overflow chains, which are followed by their real page
pointers.  A leaf may come before its overflow pages in the file, so such rows wait until their chain has arrived.
Recent pages live in a fixed ring buffer (window pages), so memory stays bounded however large the database is.
Rows are emitted in rowid order (= recording order); a row whose chain fell out of the window is reported and skipped.
"""
import struct
import sys

import numpy as np

from sqlpages import U, parse_leaf


def iter_rows(stream, window=16384, chunk_pages=1024):
    ring = np.zeros((window, U), np.uint8)
    have = np.full(window, -1, np.int64)
    pending = []            # (rowid, P, loc, ov)
    ready = {}              # rowid -> payload
    lost = 0
    pno = 0                 # pages read so far (page numbers are 1-based)
    tail = b''
    while True:
        data = stream.read(chunk_pages * U - len(tail))
        data = tail + data if data else tail
        if not data:
            break
        npg = len(data) // U
        tail = data[npg * U:]
        if npg == 0:
            break
        pages = np.frombuffer(data, np.uint8, count=npg * U).reshape(npg, U)
        nums = np.arange(pno + 1, pno + npg + 1)
        ring[nums % window] = pages
        have[nums % window] = nums
        heads = pages[:, 0].copy()
        if pno == 0:
            heads[0] = pages[0, 100]
        for k in np.nonzero(heads == 0x0D)[0]:
            p = int(nums[k])
            for rowid, P, loc, ov in parse_leaf(memoryview(ring[p % window]), p):
                if ov == 0:
                    ready[rowid] = loc
                else:
                    pending.append((rowid, P, loc, ov))
        pno += npg
        still = []
        for item in pending:
            rowid, P, loc, ov = item
            parts, need, cur, ok = [loc], P - len(loc), ov, True
            while need > 0:
                if cur > pno:
                    ok = None
                    break
                if have[cur % window] != cur:
                    ok = False
                    break
                pg = ring[cur % window]
                take = min(U - 4, need)
                parts.append(pg[4:4 + take].tobytes())
                need -= take
                cur = struct.unpack('>I', pg[:4].tobytes())[0]
            if ok is None:
                still.append(item)
            elif ok:
                ready[rowid] = b''.join(parts)
            else:
                lost += 1
                print(f'WARN row {rowid}: overflow chain left the window', file=sys.stderr)
        pending = still
        limit = min((r for r, *_ in pending), default=None)
        for r in sorted(ready):
            if limit is not None and r > limit:
                break
            yield r, ready.pop(r)
    for r in sorted(ready):
        yield r, ready.pop(r)
    if pending or lost:
        print(f'WARN {len(pending)} rows incomplete at end, {lost} lost', file=sys.stderr)
