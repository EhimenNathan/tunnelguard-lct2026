"""Launch the TunnelGuard detector (optionally RViz2 and bag playback).

  ros2 launch tunnel_guard tunnel_guard.launch.py                                  # detector only
  ros2 launch tunnel_guard tunnel_guard.launch.py rviz:=true                       # + RViz2
  ros2 launch tunnel_guard tunnel_guard.launch.py bag:=/data/doubleT_obstacle rviz:=true rate:=1.0

With bag:=..., the PointCloud2 topic is read from the bag's metadata.yaml and playback starts only after the detector
reports that it is subscribed and ready (numba kernels loaded), so no frame is lost.
"""
import os

import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessIO
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def bag_cloud_topic(bag):
    """First sensor_msgs/msg/PointCloud2 topic listed in a rosbag2 metadata.yaml, or None."""
    try:
        with open(os.path.join(bag, 'metadata.yaml')) as f:
            info = yaml.safe_load(f)['rosbag2_bagfile_information']
        for t in info['topics_with_message_count']:
            meta = t['topic_metadata']
            if meta['type'] == 'sensor_msgs/msg/PointCloud2':
                return meta['name']
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        pass
    return None


def launch_setup(context):
    share = get_package_share_directory('tunnel_guard')
    bag = LaunchConfiguration('bag').perform(context)
    if bag and LaunchConfiguration('copy_bag').perform(context).lower() in ('1', 'true', 'yes'):
        # Docker Desktop on Windows/macOS: rosbag2 reads SQLite through the host-folder mount far slower than real time
        # (most frames never reach the detector); a copy inside the container plays at full rate.
        import shutil
        import tempfile
        local = os.path.join(tempfile.mkdtemp(prefix='tunnel_guard_bag_'), os.path.basename(os.path.normpath(bag)))
        shutil.copytree(bag, local)
        bag = local
    topic = LaunchConfiguration('input_topic').perform(context)
    if bag and topic == 'auto':
        topic = bag_cloud_topic(bag) or 'auto'
    overrides = {'input_topic': topic}
    play_cmd = ['ros2', 'bag', 'play', bag, '--rate', LaunchConfiguration('rate')]
    if bag and topic != 'auto':
        # large clouds over best-effort DDS lose fragments; replaying a file must be lossless -> reliable on both ends
        qos_file = os.path.join(tempfile.gettempdir(), 'tunnel_guard_bag_qos.yaml')
        with open(qos_file, 'w') as f:
            yaml.safe_dump({topic: {'reliability': 'reliable', 'history': 'keep_last', 'depth': 10}}, f)
        play_cmd += ['--qos-profile-overrides-path', qos_file]
        overrides['input_reliability'] = 'reliable'
    detector = Node(
        package='tunnel_guard', executable='detector_node', name='tunnel_guard', output='screen',
        parameters=[LaunchConfiguration('params'), overrides],
    )
    actions = [
        detector,
        Node(
            package='rviz2', executable='rviz2', name='rviz2', output='log',
            arguments=['-d', os.path.join(share, 'rviz', 'tunnel_guard.rviz')],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ]
    if bag:
        play = ExecuteProcess(cmd=play_cmd, output='screen')
        started = []

        def on_output(event):
            if not started and b'TunnelGuard started' in event.text:
                started.append(True)
                return [TimerAction(period=LaunchConfiguration('play_delay'), actions=[play])]
            return None

        actions.append(RegisterEventHandler(OnProcessIO(target_action=detector, on_stdout=on_output,
                                                        on_stderr=on_output)))
    return actions


def generate_launch_description():
    share = get_package_share_directory('tunnel_guard')
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=os.path.join(share, 'config', 'tunnel_guard.yaml')),
        DeclareLaunchArgument('input_topic', default_value='auto',
                              description='PointCloud2 topic, or "auto" (taken from the bag, else first one found)'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('bag', default_value='', description='rosbag2 directory to play (optional)'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('copy_bag', default_value='false',
                              description='copy the bag into the container first (use on Docker Desktop for Windows/macOS)'),
        DeclareLaunchArgument('play_delay', default_value='1.0',
                              description='seconds between detector ready and bag playback'),
        OpaqueFunction(function=launch_setup),
    ])
