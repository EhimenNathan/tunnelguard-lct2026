"""Different objects at one range must not share STOP evidence."""
import numpy as np
from tunnel_guard.core.detector import ObstacleDetector
from tunnel_guard.core.farfield import FarField, FarFieldConfig
from tunnel_guard.core.geometry import TrackGeometry


def test_fusion_uses_lateral_and_vertical_identity_and_selects_one():
    far=dict(distance=100.,lateral=0.,h_min=.2,height=1.7)
    def obj(lateral,height=1.7,size=(.5,.4,1.5)):
        return dict(distance=100.,lateral=lateral,height=height,size=size)
    wall=obj(1.4);cable=obj(0.,3.2,(.5,.3,.2));person=obj(.05);other=obj(.25)
    assert ObstacleDetector._far_match(far,[wall,cable])==[]
    assert ObstacleDetector._far_match(far,[wall,cable,other,person])==[person]


def test_uncertain_warning_does_not_suppress_adjacent_certain_stop():
    x=np.arange(501.)
    g=TrackGeometry(x,np.zeros_like(x),np.full_like(x,-1.8),ok=True,
                    sigma_l=np.where(x<102,.8,.1),sigma_v=np.full_like(x,.025))
    p=np.array([[100.1,0.,-.8]]*5+[[102.8,0.,-.8]],np.float32)
    outputs=[]
    for preference in (False,True):
        far=FarField(FarFieldConfig(prefer_certain_stop=preference))
        for _ in range(12):out=far.update(p,g,30,300)
        outputs.append(out)
    assert outputs[0] and all(o['zone']==1 for o in outputs[0])
    assert any(o['zone']==2 for o in outputs[1])
