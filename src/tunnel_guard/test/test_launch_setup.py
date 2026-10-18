"""Construct real ROS launch actions with and without optional bag copying."""
import importlib.util
from pathlib import Path
import pytest
launch_module = pytest.importorskip('launch')
if not hasattr(launch_module, 'LaunchContext'):
    pytest.skip('ROS launch runtime is not installed', allow_module_level=True)
from launch import LaunchContext

@pytest.mark.parametrize('copy_bag',['false','true'])
def test_launch_constructs_and_keeps_reference_view_unchanged(tmp_path,monkeypatch,copy_bag):
    path=Path(__file__).parents[1]/'launch/tunnel_guard.launch.py'
    spec=importlib.util.spec_from_file_location('tg_launch_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    share=tmp_path/'share';(share/'rviz').mkdir(parents=True)
    reference=share/'rviz/tunnel_guard.rviz';reference.write_text('Visualization Manager: {}\n')
    bag=tmp_path/'bag';bag.mkdir();(bag/'metadata.yaml').write_text('rosbag2_bagfile_information:\n  topics_with_message_count:\n    - topic_metadata:\n        name: /lidar_points\n        type: sensor_msgs/msg/PointCloud2\n')
    monkeypatch.setattr(module,'get_package_share_directory',lambda name:str(share))
    context=LaunchContext();context.launch_configurations.update(bag=str(bag),copy_bag=copy_bag,input_topic='auto',rviz='true',params='/tmp/test.yaml',rate='1.0',start_paused='true',play_delay='1.0')
    actions=module.launch_setup(context)
    assert len(actions)==3
    assert reference.read_text()=='Visualization Manager: {}\n'
