"""ROS 2 node: subscribes to a lidar PointCloud2 stream and publishes obstacle decisions.

Published topics (namespace ~/ = /tunnel_guard/):
  ~/status            tunnel_guard_msgs/ObstacleStatus   decision for the train control system
  ~/detections        vision_msgs/Detection3DArray       standard interface for the perception pipeline
  ~/nearest_distance  std_msgs/Float32                   along-track distance to nearest in-gauge obstacle (NaN = none)
  ~/alarm             std_msgs/UInt8                     0 CLEAR, 1 CAUTION, 2 STOP
  ~/markers           visualization_msgs/MarkerArray     envelope, track centreline, obstacle boxes and labels (RViz)
  ~/obstacle_points   sensor_msgs/PointCloud2            points of confirmed obstacles
  ~/envelope_points   sensor_msgs/PointCloud2            all points inside the envelope (debug)
"""
import array
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32, UInt8, Header
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray

from .core.cloud import cloud_to_arrays
from .core.detector import ObstacleDetector, LEVEL_STOP, LEVEL_CAUTION
from .core.gauge import IN_GAUGE
from . import params as P

try:
    from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
    HAVE_VISION = True
except ImportError:  # pragma: no cover
    HAVE_VISION = False
try:
    from tunnel_guard_msgs.msg import ObstacleStatus, Obstacle
    HAVE_MSGS = True
except ImportError:  # pragma: no cover
    HAVE_MSGS = False

LEVEL_NAMES = {0: 'CLEAR', 1: 'CAUTION', 2: 'STOP'}


def make_cloud(header, xyz):
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(len(xyz))
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * msg.width
    msg.is_dense = True
    buf = array.array('B')
    buf.frombytes(np.ascontiguousarray(xyz, dtype=np.float32).tobytes())
    msg.data = buf
    return msg


class TunnelGuardNode(Node):
    def __init__(self):
        super().__init__('tunnel_guard')
        self.declare_parameter('input_topic', 'auto')
        # live lidar drivers publish best effort (sensor data QoS); bag playback is forced to reliable by the launch file
        self.declare_parameter('input_reliability', 'best_effort')
        self.declare_parameter('publish_envelope_points', True)
        self.declare_parameter('marker_step', 2.0)
        self.cfg = P.default_config()
        for name, value in P.flatten(self.cfg).items():
            self.declare_parameter(name, value)
            P.apply(self.cfg, name, self.get_parameter(name).value)
        if not self.cfg.scorer_model:
            try:
                from ament_index_python.packages import get_package_share_directory
                import os
                default_model = os.path.join(get_package_share_directory('tunnel_guard'), 'config', 'obstacle_scorer.json')
                if os.path.exists(default_model):
                    self.cfg.scorer_model = default_model
            except Exception:
                pass
        from .core.geometry import warmup
        jit = warmup()
        self.detector = ObstacleDetector(self.cfg)
        self.get_logger().info(f'scorer: {self.cfg.scorer_model or "physics rules only"}; numba kernels: {jit}')
        reliable = str(self.get_parameter('input_reliability').value).lower() == 'reliable'
        self.qos_in = QoSProfile(reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2,
                                 durability=DurabilityPolicy.VOLATILE)
        qos_out = QoSProfile(depth=10)
        if HAVE_MSGS:
            self.pub_status = self.create_publisher(ObstacleStatus, '~/status', qos_out)
        if HAVE_VISION:
            self.pub_det = self.create_publisher(Detection3DArray, '~/detections', qos_out)
        self.pub_dist = self.create_publisher(Float32, '~/nearest_distance', qos_out)
        self.pub_alarm = self.create_publisher(UInt8, '~/alarm', qos_out)
        self.pub_markers = self.create_publisher(MarkerArray, '~/markers', qos_out)
        self.pub_obst_pts = self.create_publisher(PointCloud2, '~/obstacle_points', qos_out)
        self.pub_env_pts = self.create_publisher(PointCloud2, '~/envelope_points', qos_out)
        self.sub = None
        self.frames = 0
        self.proc_sum = 0.0
        self.last_stamp = None
        topic = self.get_parameter('input_topic').value
        if topic != 'auto':
            self._subscribe(topic)
        else:
            self.discovery = self.create_timer(0.5, self._discover)
        self.create_timer(5.0, self._report)
        self.get_logger().info(f'TunnelGuard started (input_topic={topic}, reliability={self.qos_in.reliability.name.lower()}, custom msgs={HAVE_MSGS}, vision_msgs={HAVE_VISION})')

    # ------------------------------------------------------------------ plumbing
    def _discover(self):
        if self.sub is not None:
            return
        for name, types in self.get_topic_names_and_types():
            if 'sensor_msgs/msg/PointCloud2' in types and not name.startswith(self.get_fully_qualified_name()):
                self._subscribe(name)
                self.discovery.cancel()
                return

    def _subscribe(self, topic):
        self.sub = self.create_subscription(PointCloud2, topic, self._on_cloud, self.qos_in)
        self.get_logger().info(f'subscribed to {topic}')

    def _report(self):
        if self.frames:
            self.get_logger().info(f'frames={self.frames} mean processing {1e3 * self.proc_sum / self.frames:.1f} ms')

    # ------------------------------------------------------------------ processing
    def _on_cloud(self, msg: PointCloud2):
        t_in = time.perf_counter()
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if self.last_stamp is not None and stamp < self.last_stamp - 1.0:
            self.get_logger().warn('time jumped backwards (bag restarted?) - resetting state')
            self.detector.reset()
        self.last_stamp = stamp
        fields = [(f.name, f.offset, f.datatype, f.count) for f in msg.fields]
        arr = cloud_to_arrays(fields, msg.data, msg.point_step, msg.is_bigendian)
        if not all(k in arr for k in 'xyz'):
            self.get_logger().error('PointCloud2 without x/y/z fields', throttle_duration_sec=5.0)
            return
        xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1)
        inten = arr.get('intensity')
        ring = arr.get('ring', arr.get('channel'))
        res = self.detector.process(xyz, stamp, intensity=inten, ring=ring)
        proc_ms = 1e3 * (time.perf_counter() - t_in)
        self.frames += 1
        self.proc_sum += proc_ms / 1e3
        self._publish(msg.header, res, proc_ms)

    def _publish(self, header, res, proc_ms):
        Rm = res.rotation
        a = UInt8(); a.data = int(res.level); self.pub_alarm.publish(a)
        d = Float32(); d.data = float(res.nearest_distance); self.pub_dist.publish(d)
        if HAVE_MSGS:
            st = ObstacleStatus()
            st.header = header
            st.level = int(res.level)
            st.obstacle_detected = res.level == LEVEL_STOP
            st.nearest_distance = float(res.nearest_distance)
            st.nearest_time_to_collision = float(res.nearest_ttc)
            st.clear_distance = float(res.clear_distance)
            st.track_curvature = float(res.curvature)
            st.processing_time_ms = float(proc_ms)
            for o in res.obstacles:
                m = Obstacle()
                m.id = int(o['id']); m.zone = int(o['zone'])
                m.distance = float(o['distance']); m.range = float(o['range'])
                m.position = Point(x=float(o['centroid'][0]), y=float(o['centroid'][1]), z=float(o['centroid'][2]))
                m.size = Vector3(x=float(o['size'][0]), y=float(o['size'][1]), z=float(o['size'][2]))
                m.lateral_offset = float(o['lateral']); m.height_above_rail = float(o['height'])
                m.num_points = int(o['n']); m.confidence = float(o['confidence'])
                m.closing_speed = float(o['closing_speed']); m.time_to_collision = float(o['ttc'])
                m.age_frames = int(o['age'])
                st.obstacles.append(m)
            self.pub_status.publish(st)
        if HAVE_VISION:
            da = Detection3DArray(); da.header = header
            for o in res.obstacles:
                det = Detection3D(); det.header = header
                det.id = str(o['id'])
                det.bbox.center.position.x = float(o['centroid'][0])
                det.bbox.center.position.y = float(o['centroid'][1])
                det.bbox.center.position.z = float(o['centroid'][2])
                det.bbox.center.orientation.w = 1.0
                det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = (float(max(v, 0.1)) for v in o['size'])
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = 'obstacle_in_gauge' if o['zone'] == IN_GAUGE else 'obstacle_near_gauge'
                hyp.hypothesis.score = float(o['confidence'])
                det.results.append(hyp)
                da.detections.append(det)
            self.pub_det.publish(da)
        # points
        if res.obstacles:
            idx = np.concatenate([o['idx'] for o in res.obstacles])
            self.pub_obst_pts.publish(make_cloud(header, res.forward_points[idx] @ Rm))
        else:
            self.pub_obst_pts.publish(make_cloud(header, np.zeros((0, 3), np.float32)))
        if self.get_parameter('publish_envelope_points').value:
            self.pub_env_pts.publish(make_cloud(header, res.forward_points[res.zone > 0] @ Rm))
        self.pub_markers.publish(self._markers(header, res))

    def _markers(self, header, res):
        ma = MarkerArray()
        Rm = res.rotation
        g = res.geometry
        clear = Marker(); clear.header = header; clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        if g is not None and g.ok:
            step = float(self.get_parameter('marker_step').value)
            xs = np.arange(1.0, max(res.clear_distance, 2.0), step)
            hw = max(v[1] for v in self.cfg.gauge.profile)
            htop = max(v[0] for v in self.cfg.gauge.profile)
            yc = g.centre(xs); zr = g.rail_z(xs)
            col = {0: (0.1, 0.9, 0.2), 1: (1.0, 0.7, 0.0), 2: (1.0, 0.1, 0.1)}[res.level]
            for k, (off, hgt) in enumerate([(hw, 0.0), (-hw, 0.0), (hw, htop), (-hw, htop), (0.0, 0.0)]):
                m = Marker(); m.header = header; m.ns = 'envelope'; m.id = k; m.type = Marker.LINE_STRIP
                m.scale.x = 0.06 if off else 0.04
                m.color.r, m.color.g, m.color.b = col; m.color.a = 0.9 if off else 0.5
                m.pose.orientation.w = 1.0
                pts = np.stack([xs, yc + off, zr + hgt + g.roll * off], 1) @ Rm
                m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
                ma.markers.append(m)
        for o in res.obstacles:
            c = o['centroid']
            m = Marker(); m.header = header; m.ns = 'obstacles'; m.id = int(o['id']); m.type = Marker.CUBE
            m.pose.position = Point(x=float(c[0]), y=float(c[1]), z=float(c[2])); m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = (float(max(v, 0.3)) for v in o['size'])
            m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.1, 0.1, 0.6) if o['zone'] == IN_GAUGE else (1.0, 0.7, 0.0, 0.6)
            ma.markers.append(m)
            t = Marker(); t.header = header; t.ns = 'labels'; t.id = int(o['id']); t.type = Marker.TEXT_VIEW_FACING
            t.pose.position = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]) + 1.5); t.pose.orientation.w = 1.0
            t.scale.z = 0.8 + 0.01 * o['distance']
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            ttc = '' if not math.isfinite(o['ttc']) else f"  TTC {o['ttc']:.1f}s"
            t.text = f"{o['distance']:.1f} m{ttc}"
            ma.markers.append(t)
        s = Marker(); s.header = header; s.ns = 'status'; s.id = 0; s.type = Marker.TEXT_VIEW_FACING
        s.pose.position = Point(x=0.0, y=0.0, z=4.0) if Rm is None else Point(**dict(zip('xyz', (float(v) for v in np.array([8.0, 0.0, 3.5]) @ Rm))))
        s.pose.orientation.w = 1.0
        s.scale.z = 0.7
        s.color.r, s.color.g, s.color.b = {0: (0.1, 0.9, 0.2), 1: (1.0, 0.7, 0.0), 2: (1.0, 0.1, 0.1)}[res.level]
        s.color.a = 1.0
        nd = '' if math.isnan(res.nearest_distance) else f' | obstacle {res.nearest_distance:.1f} m'
        s.text = f'{LEVEL_NAMES[res.level]}{nd} | clear {res.clear_distance:.0f} m | {1e3 * res.timings["total"]:.0f} ms'
        ma.markers.append(s)
        return ma


def main(args=None):
    rclpy.init(args=args)
    node = TunnelGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
