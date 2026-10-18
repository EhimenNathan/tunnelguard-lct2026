"""Fresh stationary scans must confirm a hazard; a sparse edit must not vanish."""
import numpy as np
from tunnel_guard.core.detector import ObstacleDetector,DetectorConfig,LEVEL_STOP
from tunnel_sim import make_tunnel

def test_identical_fresh_scans_confirm_static_hazard():
    det=ObstacleDetector(DetectorConfig(forward_axis='x',mount_calibration=False,scorer_model='none'))
    p=make_tunnel(seed=41,obstacles=[(90.,0.,.6,.6,1.)])
    for k in range(30):
        res=det.process(p,k*.1)
        assert res.stamp==k*.1
    assert res.level==LEVEL_STOP and abs(res.nearest_distance-90)<2

def test_repeated_stamp_sparse_and_intensity_edits_are_not_discarded():
    det=ObstacleDetector(DetectorConfig(forward_axis='x',mount_calibration=False,scorer_model='none'))
    p=make_tunnel(seed=42);intensity=np.zeros(len(p),np.float32)
    first=det.process(p,0.,intensity=intensity)
    assert det.process(p,0.,intensity=intensity) is first
    # Mutate the original caller-owned buffer, outside the sampled fingerprint.
    step=max(1,len(p)//4096);i=1 if step>1 else 0
    p[i]=[120.,0.,.5]
    changed=det.process(p,0.,intensity=intensity)
    assert changed is not first
    intensity[i]=150.
    updated=det.process(p,0.,intensity=intensity)
    assert updated is not changed
    assert det.process(p,0.,intensity=intensity) is updated


def test_uncertain_warning_limits_verified_clear_distance(monkeypatch):
    from tunnel_guard.core.geometry import TrackGeometry
    cfg=DetectorConfig(forward_axis='x',mount_calibration=False,scorer_model='none',
                       settle_frames=0,scatter_check=False,cand_min_points=1000)
    det=ObstacleDetector(cfg);x=np.arange(501.)
    g=TrackGeometry(x,np.zeros_like(x),np.full_like(x,-1.8),ok=True,
                    rail_range=45.,valid_range=400.,trusted_range=400.,
                    sigma_l=np.full_like(x,.8),sigma_v=np.full_like(x,.025))
    monkeypatch.setattr(det.geo,'estimate',lambda p,travel=0.:g)
    wall=np.column_stack([np.linspace(395,400,30),np.full(30,5.),np.full(30,3.)])
    target=np.column_stack([np.linspace(300,300.1,8),np.zeros(8),np.full(8,-.8)])
    p=np.vstack([wall,target]).astype(np.float32)
    for k in range(25):res=det.process(p,k*.1)
    assert res.level==1 and np.isnan(res.nearest_distance)
    warnings=[o for o in res.obstacles if o.get('geometry_uncertain')]
    assert warnings and 299.<res.clear_distance<=min(o['distance'] for o in warnings)
