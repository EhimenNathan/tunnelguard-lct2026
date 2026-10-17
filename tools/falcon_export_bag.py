"""Write a standard ROS 2 Humble bag (sqlite3, CDR sensor_msgs/PointCloud2 x,y,z,intensity) from the finals' Falcon
recording - either from the local frame cache or streamed straight from the organisers' Yandex Disk link (no download,
no extraction), so a test bag can be produced on any machine, including a CI runner or a cloud dev box.

  python falcon_export_bag.py <out_dir> <first> <last> [--cache falcon_cache.bin] [--topic /iv_points] [--frame seyond]
"""
import argparse
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

from export_bag import pointcloud2


def frames(cache, first, last):
    if cache:
        from falcon_cache import frames_cached
        for k, t, xyz, inten in frames_cached(cache, first, last + 1):
            yield k, t, xyz, inten
        return
    from falcon_stream import decode, frames as stream
    for k, (t, pc) in enumerate(stream()):
        if k < first:
            continue
        if k > last:
            return
        a = decode(pc)
        xyz = np.stack([a['x'], a['y'], a['z']], 1)
        ok = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)
        yield k, t, xyz[ok].astype(np.float32), a['intensity'][ok].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('first', type=int)
    ap.add_argument('last', type=int)
    ap.add_argument('--cache', default='')
    ap.add_argument('--topic', default='/iv_points')
    ap.add_argument('--frame', default='seyond')
    a = ap.parse_args()
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
    stamps, t_base = [], 1_700_000_000_000_000_000
    for k, t, xyz, inten in frames(a.cache, a.first, a.last):
        pts = np.column_stack([xyz, inten]).astype(np.float32)
        ts = t_base + int(round(t * 1e9))
        db.execute('INSERT INTO messages (topic_id, timestamp, data) VALUES (1, ?, ?)', (ts, pointcloud2(ts, a.frame, pts)))
        stamps.append(ts)
        if len(stamps) % 100 == 0:
            print(len(stamps), 'frames', flush=True)
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
    open(os.path.join(a.out, 'metadata.yaml'), 'w', newline='\n').write(meta)
    print(f'{n} frames, {dur / 1e9:.1f} s, {os.path.getsize(db_path) / 1e6:.0f} MB -> {a.out}')


if __name__ == '__main__':
    main()
