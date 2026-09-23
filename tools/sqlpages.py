import struct
U = 4096
def varint(b, i):
    v = 0
    for k in range(8):
        c = b[i]; i += 1
        v = (v << 7) | (c & 0x7f)
        if c < 0x80: return v, i
    v = (v << 8) | b[i]; return v, i + 1

def parse_leaf(page, pno):
    """Yield table-leaf cells: (rowid, payload_size, local_bytes, first_overflow)"""
    off = 100 if pno == 1 else 0
    if page[off] != 0x0d: return
    ncell = struct.unpack('>H', page[off+3:off+5])[0]
    X = U - 35; M = ((U - 12) * 32 // 255) - 23
    for c in range(ncell):
        cp = struct.unpack('>H', page[off+8+2*c:off+10+2*c])[0]
        P, i = varint(page, cp); rowid, i = varint(page, i)
        if P <= X:
            yield rowid, P, bytes(page[i:i+P]), 0
        else:
            K = M + ((P - M) % (U - 4)); L = K if K <= X else M
            yield rowid, P, bytes(page[i:i+L]), struct.unpack('>I', page[i+L:i+L+4])[0]

def record_header(payload):
    hl, i = varint(payload, 0); types = []
    while i < hl:
        t, i = varint(payload, i); types.append(t)
    return hl, types

def read_msg_from_file(data, leafpage):
    pg = data[(leafpage-1)*U:leafpage*U]
    for rowid, P, loc, ov in parse_leaf(pg, leafpage):
        buf = bytearray(loc); cur = ov
        while cur and len(buf) < P:
            o = (cur-1)*U; buf += data[o+4:o+U]; cur = struct.unpack('>I', data[o:o+4])[0]
        return rowid, bytes(buf[:P])

def split_record(payload):
    hl, types = record_header(payload)
    i = hl; vals = []
    for t in types:
        if t == 0: vals.append(None)
        elif 1 <= t <= 6:
            n = {1:1,2:2,3:3,4:4,5:6,6:8}[t]; vals.append(int.from_bytes(payload[i:i+n],'big',signed=True)); i += n
        elif t >= 12 and t % 2 == 0:
            n = (t-12)//2; vals.append(payload[i:i+n]); i += n
        elif t >= 13:
            n = (t-13)//2; vals.append(payload[i:i+n].decode()); i += n
        else: vals.append(t)
    return vals

def parse_pc2(cdr):
    """Minimal CDR (little-endian) PointCloud2 parser -> dict"""
    b = memoryview(cdr); i = 4
    def al(n):
        nonlocal i
        i += (-(i-4)) % n
    def u32():
        nonlocal i
        al(4); v = struct.unpack_from('<I', b, i)[0]; i += 4; return v
    def s():
        nonlocal i
        n = u32(); v = bytes(b[i:i+n-1]).decode(); i += n; return v
    sec = u32(); nsec = u32(); frame = s()
    h = u32(); w = u32(); nf = u32(); fields = []
    for _ in range(nf):
        name = s(); off = u32(); dt = b[i]; i += 1; cnt = u32(); fields.append((name, off, dt, cnt))
    big = b[i]; i += 1
    ps = u32(); rs = u32(); n = u32(); data = b[i:i+n]; i += n
    dense = b[i] if i < len(b) else None
    return dict(stamp=sec+nsec*1e-9, frame=frame, height=h, width=w, fields=fields, big=big, point_step=ps, row_step=rs, data=data, dense=dense)
