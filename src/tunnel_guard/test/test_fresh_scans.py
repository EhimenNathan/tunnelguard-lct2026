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
