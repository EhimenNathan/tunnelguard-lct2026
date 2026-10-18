"""Actual ROS message deltas: stable updates, removal, invalid-geometry cleanup."""
from types import SimpleNamespace
import numpy as np
import pytest
pytest.importorskip('rclpy')
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from tunnel_guard.detector_node import TunnelGuardNode
from tunnel_guard.core.detector import DetectorConfig
from tunnel_guard.core.geometry import TrackGeometry

def test_marker_updates_reuse_ids_and_delete_only_disappearing_objects():
    node=SimpleNamespace(cfg=DetectorConfig(),_disp=None,_t_frames=[],
                         get_parameter=lambda name:SimpleNamespace(value=5. if name=='marker_step' else 0.))
    x=np.arange(501.)
    g=TrackGeometry(x,np.zeros_like(x),np.full_like(x,-1.8),ok=True)
    obj=dict(id=123,idx=np.zeros(0,int),centroid_fwd=np.array([80.,0.,-.8]),size=np.array([.5,.5,1.7]),zone=1,distance=80.,ttc=float('inf'))
    res=SimpleNamespace(geometry=g,clear_distance=80.,envelope_range=100.,level=1,dust=[],
                       obstacles=[obj],forward_points=None,nearest_distance=float('nan'),timings={'total':.01})
    h=Header()
    first=TunnelGuardNode._markers(node,h,res)
    assert sum(m.action==Marker.DELETEALL for m in first.markers)==1
    second=TunnelGuardNode._markers(node,h,res)
    assert all(m.action==Marker.ADD for m in second.markers)
    res.obstacles=[]
    third=TunnelGuardNode._markers(node,h,res)
    removed={(m.ns,m.id) for m in third.markers if m.action==Marker.DELETE}
    assert removed=={('obstacles',123),('labels',123)}
    g.ok=False
    last=TunnelGuardNode._markers(node,h,res)
    assert node._marker_keys=={('status',0)}
    assert any(m.action==Marker.DELETE and m.ns=='envelope' for m in last.markers)
