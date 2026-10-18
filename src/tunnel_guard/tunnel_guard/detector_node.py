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
from std_srvs.srv import Trigger
from geometry_msgs.msg import TransformStamped
try:
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    HAVE_TF = True
except ImportError:  # pragma: no cover
    HAVE_TF = False

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
LEVEL_RGB = {0: (0.1, 0.9, 0.2), 1: (1.0, 0.7, 0.0), 2: (1.0, 0.1, 0.1)}
FRAME = 'tunnel_guard'      # display frame: x forward along the track, y left, z up - the same for every lidar mounting


def rot_to_quat(R):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        return ((R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def make_rgb_cloud(header, xyz, rgb):
    """PointCloud2 with x, y, z and packed rgb (RViz colour transformer 'RGB8')."""
    n = len(xyz)
    rec = np.zeros(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
    rec['x'], rec['y'], rec['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec['rgb'] = rgb
    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, n
    msg.fields = [PointField(name=k, offset=4 * i, datatype=PointField.FLOAT32 if k != 'rgb' else PointField.UINT32,
                             count=1) for i, k in enumerate(('x', 'y', 'z', 'rgb'))]
    msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 16, 16 * n, True
    buf = array.array('B')
    buf.frombytes(rec.tobytes())
    msg.data = buf
    return msg


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
        self.declare_parameter('marker_step', 8.0)
        self.declare_parameter('display_hz', 5.0)
        self._display_last = -float('inf')
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
        self.qos_in = QoSProfile(reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10 if reliable else 2,
                                 durability=DurabilityPolicy.VOLATILE)
        qos_out = QoSProfile(depth=10)
        if HAVE_MSGS:
            self.pub_status = self.create_publisher(ObstacleStatus, '~/status', qos_out)
        if HAVE_VISION:
            self.pub_det = self.create_publisher(Detection3DArray, '~/detections', qos_out)
        self.pub_dist = self.create_publisher(Float32, '~/nearest_distance', qos_out)
        self.pub_alarm = self.create_publisher(UInt8, '~/alarm', qos_out)
        self.pub_callback_ms = self.create_publisher(Float32, '~/callback_ms', qos_out)
        self.pub_markers = self.create_publisher(MarkerArray, '~/markers', qos_out)
        self.pub_obst_pts = self.create_publisher(PointCloud2, '~/obstacle_points', qos_out)
        self.pub_env_pts = self.create_publisher(PointCloud2, '~/envelope_points', qos_out)
        self.pub_cloud = self.create_publisher(PointCloud2, '~/cloud', QoSProfile(depth=2))
        self.declare_parameter('display_max_points', 60000)
        self.tf_static = StaticTransformBroadcaster(self) if HAVE_TF else None
        self.declare_parameter('display_smoothing', 0.6)
        self._disp = None
        self._t_frames = []
        self.tf_key = None
        self.sub = None
        self.sub_topic = None
        self.silent_since = None
        self.frames = 0
        self.proc_sum = 0.0
        self.last_stamp = None
        self.fixed_topic = self.get_parameter('input_topic').value
        if self.fixed_topic != 'auto':
            self._subscribe(self.fixed_topic)
        # 'auto': follow whichever PointCloud2 topic is being published, so bags with different lidar topics can be played
        # one after another without restarting (the detector resets when the source changes)
        self.discovery = self.create_timer(0.5, self._discover)
        self.create_service(Trigger, '~/reset', self._on_reset)
        self.create_timer(5.0, self._report)
        self.get_logger().info(f'TunnelGuard started (input_topic={self.fixed_topic}, reliability={self.qos_in.reliability.name.lower()}, custom msgs={HAVE_MSGS}, vision_msgs={HAVE_VISION})')

    # ------------------------------------------------------------------ plumbing
    def _discover(self):
        if self.fixed_topic != 'auto':
            return
        if self.sub is not None:
            if self.count_publishers(self.sub_topic) > 0:
                self.silent_since = None
                return
            now = time.monotonic()
            self.silent_since = self.silent_since or now
            if now - self.silent_since < 2.0:
                return
        own = self.get_fully_qualified_name()
        for name, types in self.get_topic_names_and_types():
            if ('sensor_msgs/msg/PointCloud2' in types and not name.startswith(own) and name != self.sub_topic
                    and self.count_publishers(name) > 0):
                self._subscribe(name)
                return

    def _subscribe(self, topic):
        if self.sub is not None:
            self.destroy_subscription(self.sub)
            self.detector.reset()
            self.last_stamp = None
            self._disp = None
        self.sub = self.create_subscription(PointCloud2, topic, self._on_cloud, self.qos_in)
        self.sub_topic = topic
        self.silent_since = None
        self.get_logger().info(f'subscribed to {topic}')

    def _on_reset(self, request, response):
        reliable = str(self.get_parameter('input_reliability').value).lower() == 'reliable'
        self.qos_in = QoSProfile(reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST, depth=10 if reliable else 2,
                                 durability=DurabilityPolicy.VOLATILE)
        self.fixed_topic = self.get_parameter('input_topic').value
        requested = self.fixed_topic if self.fixed_topic != 'auto' else self.sub_topic
        if requested is not None:
            self._subscribe(requested)
        self._disp = None
        self.detector.reset()
        self.last_stamp = None
        response.success = True
        response.message = 'detector state reset'
        self.get_logger().info('reset requested')
        return response

    def _report(self):
        if self.frames:
            self.get_logger().info(f'frames={self.frames} mean processing {1e3 * self.proc_sum / self.frames:.1f} ms')

    # ------------------------------------------------------------------ processing
    def _on_cloud(self, msg: PointCloud2):
        t_in = time.perf_counter()
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if self.last_stamp is not None and (stamp < self.last_stamp - 1.0 or stamp > self.last_stamp + 30.0):
            self.get_logger().warn('time jumped (bag restarted or a different bag) - resetting state')
            self.detector.reset()
            self._disp = None
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
        duration = Float32()
        duration.data = float(1000 * (time.perf_counter() - t_in))
        self.pub_callback_ms.publish(duration)

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
        if self.pub_obst_pts.get_subscription_count() > 0 and res.obstacles:
            idx = np.concatenate([o['idx'] for o in res.obstacles])
            self.pub_obst_pts.publish(make_cloud(header, res.forward_points[idx] @ Rm))
        elif self.pub_obst_pts.get_subscription_count() > 0:
            self.pub_obst_pts.publish(make_cloud(header, np.zeros((0, 3), np.float32)))
        if self.get_parameter('publish_envelope_points').value and self.pub_env_pts.get_subscription_count() > 0:
            self.pub_env_pts.publish(make_cloud(header, res.forward_points[res.zone > 0] @ Rm))
        now = time.monotonic()
        if now - self._display_last >= 1.0 / max(float(self.get_parameter('display_hz').value), 0.1):
            self._display_last = now
            self._publish_display(header, res)
            if self.pub_markers.get_subscription_count() > 0:
                self.pub_markers.publish(self._markers(header, res))

    def _publish_display(self, header, res):
        """Display frame TF (lidar frame -> tunnel_guard) and a colour-coded cloud in the display frame."""
        if res.rotation is None or res.forward_points is None:
            return
        key = (header.frame_id, res.rotation.tobytes())
        if self.tf_static is not None and key != self.tf_key and header.frame_id:
            t = TransformStamped()
            t.header.stamp = header.stamp
            t.header.frame_id = FRAME
            t.child_frame_id = header.frame_id
            q = rot_to_quat(np.asarray(res.rotation, float))
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
            self.tf_static.sendTransform(t)
            self.tf_key = key
        if self.pub_cloud.get_subscription_count() == 0:
            return
        p = res.forward_points
        zone = res.zone if res.zone is not None else np.zeros(len(p), np.uint8)
        nmax = int(self.get_parameter('display_max_points').value)
        sel = np.arange(len(p))
        if len(p) > nmax:
            sel = np.unique(np.concatenate([np.nonzero(zone > 0)[0], np.linspace(0, len(p) - 1, nmax).astype(int)]))
        q, z = p[sel], zone[sel]
        r = np.clip(np.log(np.maximum(q[:, 0], 3.0)) / np.log(400.0), 0, 1)
        g = (200 - 140 * r).astype(np.uint32)
        rgb = (g << 16) | (g << 8) | np.minimum(g + 30, 255)
        rgb = np.where(z == 1, np.uint32((255 << 16) | (190 << 8)), rgb)
        rgb = np.where(z == 2, np.uint32((255 << 16) | (40 << 8) | 40), rgb)
        if res.obstacles:
            mark = np.zeros(len(p), bool)
            mark[np.concatenate([o['idx'] for o in res.obstacles])] = True
            rgb = np.where(mark[sel], np.uint32((255 << 16) | 255), rgb)
        self.pub_cloud.publish(make_rgb_cloud(Header(stamp=header.stamp, frame_id=FRAME), q, rgb.astype(np.uint32)))

    def _markers(self, header, res):
        """Envelope tube (translucent walls and roof along the reconstructed track), edges, 50 m distance marks,
        obstacle boxes with distance / TTC labels and a status line - all in the display frame."""
        ma = MarkerArray()
        h = Header(stamp=header.stamp, frame_id=FRAME)
        # Stable marker IDs update existing RViz objects. DELETEALL every scan
        # needlessly destroys geometry/text resources and can visibly flicker.
        if not hasattr(self, '_marker_keys'):
            clear = Marker(); clear.header = h; clear.action = Marker.DELETEALL
            ma.markers.append(clear)
        g = res.geometry
        col = LEVEL_RGB[res.level]

        def mk(ns, mid, typ):
            m = Marker(); m.header = h; m.ns = ns; m.id = mid; m.type = typ; m.pose.orientation.w = 1.0
            return m

        def pt(q):
            return Point(x=float(q[0]), y=float(q[1]), z=float(q[2]))

        if g is not None and g.ok and max(res.clear_distance, res.envelope_range) > 2.0:
            step = float(self.get_parameter('marker_step').value)
            far_end = max(res.envelope_range, res.clear_distance)
            grid = np.arange(0.0, far_end + step, step)
            yc_now, zr_now = g.centre(grid), g.rail_z(grid)
            # display smoothing: the drawn envelope follows the estimate with a short time constant, so it neither
            # flickers nor lags (decisions always use the unsmoothed estimate of the current frame)
            a = float(self.get_parameter('display_smoothing').value)
            prev = self._disp
            if prev is not None and len(prev[0]) >= len(grid) and a > 0:
                yc = a * prev[0][:len(grid)] + (1 - a) * yc_now
                zr = a * prev[1][:len(grid)] + (1 - a) * zr_now
            else:
                yc, zr = yc_now, zr_now
            self._disp = (np.r_[yc, np.full(4000, yc[-1])], np.r_[zr, np.full(4000, zr[-1])])
            prof = np.array(self.cfg.gauge.profile, float)       # (h, half-width) bottom -> top
            ring = np.concatenate([np.stack([prof[:, 1], prof[:, 0]], 1),
                                   np.stack([-prof[::-1, 1], prof[::-1, 0]], 1)])

            def section(x0, x1):
                xs = grid[(grid >= x0) & (grid <= x1)]
                if len(xs) < 2:
                    return None
                i = np.searchsorted(grid, xs)
                return np.stack([np.repeat(xs[:, None], len(ring), 1),
                                 yc[i][:, None] + ring[None, :, 0],
                                 zr[i][:, None] + ring[None, :, 1] + g.roll * ring[None, :, 0]], 2)

            def tube(P, mid, alpha):
                m = mk('envelope', mid, Marker.TRIANGLE_LIST)
                m.scale.x = m.scale.y = m.scale.z = 1.0
                m.color.r, m.color.g, m.color.b = col
                m.color.a = alpha
                a_, b_ = P[:-1, :-1].reshape(-1, 3), P[:-1, 1:].reshape(-1, 3)
                c_, d_ = P[1:, :-1].reshape(-1, 3), P[1:, 1:].reshape(-1, 3)
                m.points = [pt(q) for q in np.stack([a_, b_, c_, b_, d_, c_], 1).reshape(-1, 3)]
                return m

            near_P = section(1.0, res.clear_distance)            # verified clear: solid envelope
            far_P = section(res.clear_distance, far_end)         # far field (temporal evidence): early-warning zone
            for mid, P, alpha in ((0, near_P, 0.18), (1, far_P, 0.06)):
                if P is not None:
                    ma.markers.append(tube(P, mid, alpha))
            P = near_P if far_P is None else (far_P if near_P is None else np.concatenate([near_P, far_P[1:]]))
            if P is not None:
                for k, j in enumerate((0, len(prof) - 1, len(prof), len(ring) - 1)):
                    e = mk('envelope_edges', k, Marker.LINE_STRIP)
                    e.scale.x = 0.06
                    e.color.r, e.color.g, e.color.b = col
                    e.color.a = 0.95
                    e.points = [pt(q) for q in P[:, j]]
                    ma.markers.append(e)
            for k, dist in enumerate(range(50, int(far_end) + 1, 50)):
                i = int(np.searchsorted(grid, dist))
                t = mk('distance_marks', k, Marker.TEXT_VIEW_FACING)
                t.pose.position = Point(x=float(dist), y=float(yc[min(i, len(yc) - 1)] - 2.2), z=float(zr[min(i, len(zr) - 1)]))
                t.scale.z = 0.6 + 0.012 * dist
                t.color.r = t.color.g = t.color.b = 0.85
                t.color.a = 0.9
                t.text = f'{dist} m'
                ma.markers.append(t)
            for k, dz in enumerate(res.dust or []):
                i = int(np.clip(np.searchsorted(grid, dz['distance']), 0, len(grid) - 1))
                m = mk('dust', k, Marker.CUBE)
                m.pose.position = Point(x=float(dz['distance'] + 1.5), y=float(yc[i]), z=float(zr[i] + 1.3))
                m.scale.x, m.scale.y, m.scale.z = 3.0, 2.4, 2.4
                m.color.r = m.color.g = m.color.b = 0.75
                m.color.a = 0.25
                ma.markers.append(m)
                t = mk('dust_labels', k, Marker.TEXT_VIEW_FACING)
                t.pose.position = Point(x=float(dz['distance'] + 1.5), y=float(yc[i]), z=float(zr[i] + 3.0))
                t.scale.z = 0.6 + 0.01 * dz['distance']
                t.color.r = t.color.g = t.color.b = 0.9
                t.color.a = 1.0
                t.text = 'ПЫЛЬ / АЭРОЗОЛЬ (фильтр)'
                ma.markers.append(t)
        p = res.forward_points
        for o in res.obstacles:
            q = p[o['idx']] if p is not None and len(o['idx']) else None
            c = q.mean(0) if q is not None else np.asarray(o['centroid_fwd'], float).copy()
            if o.get('held'):
                c[0] = float(o['distance'])
            ext = (q.max(0) - q.min(0)) if q is not None else np.asarray(o['size'], float)
            m = mk('obstacles', int(o['id']), Marker.CUBE)
            m.pose.position = pt(c)
            m.scale.x, m.scale.y, m.scale.z = (float(max(v, 0.3)) for v in ext)
            m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.1, 0.1, 0.7) if o['zone'] == IN_GAUGE else (1.0, 0.7, 0.0, 0.7)
            ma.markers.append(m)
            t = mk('labels', int(o['id']), Marker.TEXT_VIEW_FACING)
            t.pose.position = Point(x=float(c[0]), y=float(c[1]), z=float(c[2] + ext[2] / 2 + 1.0 + 0.01 * o['distance']))
            t.scale.z = 0.8 + 0.012 * o['distance']
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            ttc = '' if not math.isfinite(o['ttc']) else f"  TTC {o['ttc']:.1f}s"
            t.text = f"{o['distance']:.1f} m{ttc}" + (" | ГЕОМЕТРИЯ НЕУВЕРЕННА" if o.get("geometry_uncertain") else "")
            ma.markers.append(t)
        st = mk('status', 0, Marker.TEXT_VIEW_FACING)
        st.pose.position = Point(x=8.0, y=0.0, z=4.0)
        st.scale.z = 0.7
        st.color.r, st.color.g, st.color.b = col
        st.color.a = 1.0
        nd = '' if math.isnan(res.nearest_distance) else f' | obstacle {res.nearest_distance:.1f} m'
        now = time.monotonic()
        self._t_frames = [t for t in self._t_frames if now - t < 2.0] + [now]
        hz = (len(self._t_frames) - 1) / max(self._t_frames[-1] - self._t_frames[0], 1e-3) if len(self._t_frames) > 1 else 0.0
        st.text = (f'{LEVEL_NAMES[res.level]}{nd} | clear {res.clear_distance:.0f} m | envelope {res.envelope_range:.0f} m'
                   f' | {1e3 * res.timings["total"]:.0f} ms | {hz:.1f} Hz')
        ma.markers.append(st)
        current = {(m.ns, m.id) for m in ma.markers if m.action == Marker.ADD}
        for ns, mid in getattr(self, '_marker_keys', set()) - current:
            stale = Marker(); stale.header = h; stale.ns = ns; stale.id = mid; stale.action = Marker.DELETE
            ma.markers.append(stale)
        self._marker_keys = current
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
