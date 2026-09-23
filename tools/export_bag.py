"""Write a standard ROS 2 Humble bag (sqlite3 storage, CDR-serialised sensor_msgs/PointCloud2) from the range-image
cache, so the detector can be tested end to end (docker run ... ros2 bag play) without the original 4 GB archive.

Points are the cached returns (range quantised to 1 cm) in the original lidar frame, fields x, y, z, intensity (float32);
invalid returns are dropped (unorganised cloud).  No ROS installation is needed to write the bag.
usage (scratchpad):  python export_bag.py <cache name> <out dir> [first last] [--topic T] [--frame F]"""
import argparse
import os
import sqlite3
import struct
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from loader import BagCache


class Cdr:
    """Little-endian CDR (XCDR1) writer; alignment is relative to the end of the 4-byte encapsulation header."""

    def __init__(self):
        self.b = bytearray(b'\x00\x01\x00\x00')

    def align(self, n):
        pad = (-(len(self.b) - 4)) % n
        self.b += b'\x00' * pad

    def u8(self, v):
        self.b += struct.pack('<B', v)

    def u32(self, v):
        self.align(4)
        self.b += struct.pack('<I', v)

    def i32(self, v):
        self.align(4)
        self.b += struct.pack('<i', v)

    def string(self, s):
        e = s.encode() + b'\x00'
        self.u32(len(e))
        self.b += e

    def blob(self, data):
        self.u32(len(data))
        self.b += data


def pointcloud2(stamp_ns, frame_id, pts):
    c = Cdr()
    c.i32(int(stamp_ns // 1_000_000_000))
    c.u32(int(stamp_ns % 1_000_000_000))
    c.string(frame_id)
    c.u32(1)                     # height
    c.u32(len(pts))              # width
    c.u32(4)                     # fields
    for i, name in enumerate(('x', 'y', 'z', 'intensity')):
        c.string(name)
        c.u32(4 * i)
        c.u8(7)                  # FLOAT32
        c.u32(1)
    c.u8(0)                      # is_bigendian
    c.u32(16)                    # point_step
    c.u32(16 * len(pts))         # row_step
    c.blob(np.ascontiguousarray(pts, '<f4').tobytes())
    c.u8(1)                      # is_dense
    return bytes(c.b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('out')
    ap.add_argument('span', nargs='*', type=int)
    ap.add_argument('--topic', default='/lidar_points')
    ap.add_argument('--frame', default='hesai_lidar')
    a = ap.parse_args()
    b = BagCache(a.name)
    first, last = (a.span + [0, b.n - 1])[:2] if a.span else (0, b.n - 1)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.basename(os.path.normpath(a.out))
    db_name = f'{base}_0.db3'
    db_path = os.path.join(a.out, db_name)
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    db.executescript("""
        CREATE TABLE schema(schema_version INTEGER PRIMARY KEY, ros_distro TEXT NOT NULL);
        INSERT INTO schema VALUES (3, 'humble');
        CREATE TABLE metadata(id INTEGER PRIMARY KEY, metadata_version INTEGER NOT NULL, metadata TEXT NOT NULL);
        CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
                            serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL);
        CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, timestamp INTEGER NOT NULL,
                              data BLOB NOT NULL);
        CREATE INDEX timestamp_idx ON messages (timestamp ASC);""")
    db.execute('INSERT INTO topics VALUES (1, ?, ?, ?, ?)', (a.topic, 'sensor_msgs/msg/PointCloud2', 'cdr', ''))
    stamps = []
    t_base = 1_700_000_000_000_000_000
    for f in range(first, last + 1):
        rng, inten = b.frame(f)
        V = rng > 0.5
        fwd = b.dirs[V] * rng[V][:, None]
        xyz = np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1)          # back to the lidar frame
        pts = np.column_stack([xyz, inten[V] if inten is not None else np.zeros(len(xyz))]).astype(np.float32)
        ts = t_base + int(round((b.t[f] - b.t[first]) * 1e9))
        db.execute('INSERT INTO messages (topic_id, timestamp, data) VALUES (1, ?, ?)', (ts, pointcloud2(ts, a.frame, pts)))
        stamps.append(ts)
    db.commit()
    db.close()
    n, dur = len(stamps), stamps[-1] - stamps[0]
    meta = f"""rosbag2_bagfile_information:
  version: 5
  storage_identifier: sqlite3
  duration:
    nanoseconds: {dur}
  starting_time:
    nanoseconds_since_epoch: {stamps[0]}
  message_count: {n}
  topics_with_message_count:
    - topic_metadata:
        name: {a.topic}
        type: sensor_msgs/msg/PointCloud2
        serialization_format: cdr
        offered_qos_profiles: ""
      message_count: {n}
  compression_format: ""
  compression_mode: ""
  relative_file_paths:
    - {db_name}
  files:
    - path: {db_name}
      starting_time:
        nanoseconds_since_epoch: {stamps[0]}
      duration:
        nanoseconds: {dur}
      message_count: {n}
"""
    open(os.path.join(a.out, 'metadata.yaml'), 'w').write(meta)
    print(f'{n} frames, {dur / 1e9:.1f} s, {os.path.getsize(db_path) / 1e6:.0f} MB -> {a.out}')


if __name__ == '__main__':
    main()
