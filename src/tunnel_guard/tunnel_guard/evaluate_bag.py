"""Offline, faster-than-real-time evaluation of a rosbag2 recording (no playback needed).

  ros2 run tunnel_guard evaluate_bag --bag /data/doubleT_obstacle --out /data/results/doubleT_obstacle

Writes <out>_frames.csv (one row per lidar frame) and <out>_summary.json.  Uses exactly the same detector core
and parameters as the ROS node.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import yaml

from .core.cloud import cloud_to_arrays
from .core.detector import ObstacleDetector
from . import params as P


def load_params(cfg, path):
    if not path:
        return cfg
    with open(path) as f:
        doc = yaml.safe_load(f)
    ros = doc.get('tunnel_guard', doc).get('ros__parameters', {})

    def walk(d, prefix=''):
        for k, v in d.items():
            name = prefix + k
            if isinstance(v, dict):
                walk(v, name + '.')
            else:
                try:
                    P.apply(cfg, name, v)
                except AttributeError:
                    pass
    walk(ros)
    return cfg


def read_bag(path, topic=None):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    reader = rosbag2_py.SequentialReader()
    storage = 'mcap' if any(f.endswith('.mcap') for f in os.listdir(path)) else 'sqlite3'
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=storage),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic is None:
        topic = next(n for n, t in types.items() if t == 'sensor_msgs/msg/PointCloud2')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    while reader.has_next():
        name, data, t = reader.read_next()
        yield t, deserialize_message(data, PointCloud2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--topic', default=None)
    ap.add_argument('--params', default=None, help='YAML parameter file (same format as the node config)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-scorer', action='store_true', help='physics rules only (disable the learned scorer)')
    args = ap.parse_args()
    out = args.out or os.path.join(os.getcwd(), os.path.basename(os.path.normpath(args.bag)))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cfg = load_params(P.default_config(), args.params)
    if not cfg.scorer_model and not args.no_scorer:
        try:
            from ament_index_python.packages import get_package_share_directory
            path = os.path.join(get_package_share_directory('tunnel_guard'), 'config', 'obstacle_scorer.json')
            cfg.scorer_model = path if os.path.exists(path) else ''
        except Exception:
            pass
    from .core.geometry import warmup
    warmup()
    det = ObstacleDetector(cfg)
    rows = []
    first_detection = None
    t_wall = time.perf_counter()
    for i, (t_bag, msg) in enumerate(read_bag(args.bag, args.topic)):
        arr = cloud_to_arrays([(f.name, f.offset, f.datatype, f.count) for f in msg.fields], msg.data, msg.point_step,
                              msg.is_bigendian)
        xyz = np.stack([arr['x'], arr['y'], arr['z']], 1)
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        res = det.process(xyz, stamp, intensity=arr.get('intensity'), ring=arr.get('ring'))
        obs = [dict(id=o['id'], zone=o['zone'], distance=round(o['distance'], 2), lateral=round(o['lateral'], 2),
                    height=round(o['height'], 2), points=o['n'], ttc=None if math.isinf(o['ttc']) else round(o['ttc'], 2))
               for o in res.obstacles]
        if res.level == 2 and first_detection is None:
            first_detection = dict(frame=i, stamp=stamp, distance=res.nearest_distance)
        rows.append(dict(frame=i, stamp=f'{stamp:.3f}', level=res.level, nearest_distance=f'{res.nearest_distance:.2f}',
                         clear_distance=f'{res.clear_distance:.1f}', n_obstacles=len(obs),
                         processing_ms=f'{1e3 * res.timings["total"]:.1f}', obstacles=json.dumps(obs)))
        if i % 50 == 0:
            print(f'frame {i}: level {res.level} nearest {res.nearest_distance:.1f} clear {res.clear_distance:.0f} m '
                  f'{1e3 * res.timings["total"]:.0f} ms', flush=True)
    wall = time.perf_counter() - t_wall
    with open(out + '_frames.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ms = np.array([float(r['processing_ms']) for r in rows])
    summary = dict(bag=args.bag, frames=len(rows), stop_frames=sum(r['level'] == 2 for r in rows),
                   caution_frames=sum(r['level'] == 1 for r in rows), first_stop=first_detection,
                   processing_ms_mean=float(ms.mean()), processing_ms_p95=float(np.percentile(ms, 95)),
                   wall_seconds=wall)
    with open(out + '_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
