"""Проверка живого ROS/RViz: падение узла, потери и частота завершают CI ошибкой.

Запускается внутри образа с DISPLAY, после source /ws/install/setup.bash.
"""
import argparse
import json
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import rclpy
from tunnel_guard.core.geometry import HAVE_NATIVE
from rclpy.node import Node
from tunnel_guard_msgs.msg import ObstacleStatus
from std_msgs.msg import Float32
from rosbag2_interfaces.srv import Resume
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--expected', type=int, default=301)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Node('cloud_output_probe')
    arrivals, stamps, processing, levels = [], [], [], []
    callback_ms = []

    def receive(msg):
        arrivals.append(time.monotonic())
        stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        processing.append(float(msg.processing_time_ms))
        levels.append(int(msg.level))

    sub = node.create_subscription(ObstacleStatus, '/tunnel_guard/status', receive, 1000)
    cb_sub = node.create_subscription(Float32, '/tunnel_guard/callback_ms', lambda m: callback_ms.append(float(m.data)), 1000)
    started = time.monotonic()
    info = yaml.safe_load((Path(args.bag)/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    topic = next(t['topic_metadata']['name'] for t in info['topics_with_message_count'] if t['topic_metadata']['type']=='sensor_msgs/msg/PointCloud2')
    resume = node.create_client(Resume, '/rosbag2_player/resume')
    ready_since = None
    resumed = False
    resume_at = None
    with (out / 'launch.log').open('w') as log:
        proc = subprocess.Popen(['ros2', 'launch', 'tunnel_guard', 'tunnel_guard.launch.py',
                                 'rviz:=true', 'bag:=' + args.bag, 'play_delay:=3.0', 'start_paused:=true'],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        screenshot = False
        capture = None
        try:
            while time.monotonic() - started < 100:
                rclpy.spin_once(node, timeout_sec=0.1)
                if not resumed:
                    ready = (resume.service_is_ready() and node.count_publishers('/tunnel_guard/status')>0
                             and node.count_subscribers(topic)>0 and node.count_publishers(topic)>0)
                    ready_since = (ready_since or time.monotonic()) if ready else None
                    # The publisher graph appears before Humble's --delay 2.0
                    # finishes. Resuming inside that delay races player startup.
                    # Wait beyond it; keep exact 301/301 acceptance unchanged.
                    if ready_since is not None and time.monotonic()-ready_since>2.5:
                        resume_at = time.monotonic()
                        resume.call_async(Resume.Request())
                        resumed = True
                if len(arrivals) >= 100 and capture is None:
                    # Capturing software-rendered RViz may take a second; never block the output subscriber.
                    capture = subprocess.Popen(['import', '-window', 'root', str(out / 'rviz.png')])
                if capture is not None and capture.poll() == 0:
                    screenshot = (out / 'rviz.png').is_file()
                if len(arrivals) >= args.expected or (arrivals and time.monotonic() - arrivals[-1] > 8):
                    break
                if proc.poll() is not None:
                    break
        finally:
            runtime_log = (out / 'launch.log').read_text()
            import os
            # launch forwards SIGINT to its children. Signalling the entire group would deliver it twice.
            proc.send_signal(signal.SIGINT) if proc.poll() is None else None
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            node.destroy_node()
            rclpy.shutdown()
    # Trim initial discovery/compilation and the shutdown tail when measuring sustained delivery.
    warm = min(30, len(arrivals) // 4)
    times = np.array(arrivals[warm:])
    hz = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
    log_text = (out / 'launch.log').read_text()
    checks = dict(no_crash='Traceback' not in runtime_log and 'process has died' not in runtime_log,
                  messages=len(arrivals) == args.expected,
                  rate=bool(hz >= 9.0), stop=2 in levels, rviz=screenshot,
                  ordered=bool(len(stamps) > 1 and np.all(np.diff(stamps) > 0)))
    report = dict(expected=args.expected, received=len(arrivals), output_hz=hz, native_geometry=HAVE_NATIVE,
                  first_output_stamp=stamps[0] if stamps else None,
                  last_output_stamp=stamps[-1] if stamps else None,
                  max_delivery_backlog_ms=float(max(0.0, np.max((np.array(arrivals)-arrivals[0])-(np.array(stamps)-stamps[0])))*1000) if arrivals else None,
                  playback_delivery_lag_ms_p95=float(max(0.0,np.percentile(np.array(arrivals)-resume_at-(np.array(stamps)-int(info['starting_time']['nanoseconds_since_epoch'])/1e9),95))*1000) if arrivals and resume_at is not None else None,
                  playback_lag_reference='Resume request and aligned header/recording stamps of exported CI bag',
                  processing_ms_p50=float(np.median(processing)) if processing else None,
                  processing_ms_p95=float(np.percentile(processing, 95)) if processing else None,
                  processing_ms_p99=float(np.percentile(processing, 99)) if processing else None,
                  arrival_gap_ms_p99=float(np.percentile(np.diff(times), 99) * 1000) if len(times) > 1 else None,
                  processing_over_100ms=sum(v > 100 for v in processing),
                  callback_ms_p95=float(np.percentile(callback_ms, 95)) if callback_ms else None,
                  shutdown_clean='Traceback' not in log_text and 'process has died' not in log_text,
                  stop_messages=levels.count(2), checks=checks, passed=all(checks.values()))
    (out / 'live_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit('Живая проверка ROS/RViz не пройдена')


if __name__ == '__main__':
    main()
