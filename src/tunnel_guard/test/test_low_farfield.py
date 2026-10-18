"""Low hazards survive accumulation without treating the track bed as a STOP."""
import numpy as np
from tunnel_guard.core.geometry import TrackGeometry
from tunnel_guard.core.farfield import FarField
from tunnel_guard.core.detector import DetectorConfig

def run(height,vertical_sigma,scatter=False):
    x=np.arange(501.)
    g=TrackGeometry(x,np.zeros_like(x),np.full_like(x,-1.8),ok=True,
                    sigma_l=np.full_like(x,.1),sigma_v=np.full_like(x,vertical_sigma))
    p=np.array([[90.+i*.01,(i-3)*.025,-1.8+height] for i in range(7)],np.float32)
    far=FarField(DetectorConfig().lowfield);out=[]
    for _ in range(25):out=far.update(p,g,30,300,exclude=np.full(len(p),scatter))
    return out

def test_low_box_detected_but_stop_requires_vertical_certainty():
    sure=run(.29,.025);uncertain=run(.29,.10)
    assert len(sure)==1 and sure[0]['zone']==2
    assert len(uncertain)==1 and uncertain[0]['zone']==1

def test_track_bed_and_scatter_never_accumulate_as_low_hazards():
    assert not run(.10,.025)
    assert not run(.29,.025,scatter=True)


def test_warning_memory_does_not_extend_stop_memory():
    from tunnel_guard.core.farfield import FarFieldConfig
    x=np.arange(501.)
    g=TrackGeometry(x,np.zeros_like(x),np.full_like(x,-1.8),ok=True,
                    sigma_l=np.full_like(x,.1),sigma_v=np.full_like(x,.025))
    p=np.array([[90.,0.,-.8]],np.float32)
    far=FarField(FarFieldConfig())
    for _ in range(9):out=far.update(p,g,30,300)
    assert out and out[0]['zone']==2
    for _ in range(7):out=far.update(p[:0],g,30,300)
    assert out and out[0]['zone']==1
    for _ in range(15):out=far.update(p[:0],g,30,300)
    assert not out


def test_low_channel_rejects_continuous_rail_including_its_endpoints():
    x=np.arange(501.)
    g=TrackGeometry(x,np.zeros_like(x),np.full_like(x,-1.8),ok=True,
                    sigma_l=np.full_like(x,.1),sigma_v=np.full_like(x,.025))
    s=np.linspace(90,100,101)
    p=np.column_stack([s,np.full_like(s,.7),np.full_like(s,-1.51)]).astype(np.float32)
    far=FarField(DetectorConfig().lowfield)
    for _ in range(25):out=far.update(p,g,30,300)
    assert not out


def test_stop_position_uses_recent_lateral_evidence():
    from tunnel_guard.core.farfield import FarFieldConfig
    x=np.arange(501.)
    g=TrackGeometry(x,np.zeros_like(x),np.full_like(x,-1.8),ok=True,
                    sigma_l=np.full_like(x,.1),sigma_v=np.full_like(x,.025))
    p=np.array([[90.,0.,-.8]],np.float32)
    far=FarField(FarFieldConfig())
    for _ in range(9):out=far.update(p,g,30,300)
    assert out and out[0]['zone']==2
    p[:,1]=-.98
    for _ in range(15):out=far.update(p,g,30,300)
    assert out and out[0]['zone']==1


def test_uncertain_lateral_geometry_warns_without_stop():
    from tunnel_guard.core.farfield import FarFieldConfig
    x=np.arange(501.)
    p=np.array([[300.,0.,-.8]],np.float32)
    for sigma,expected in [(.3,2),(.8,1),(1.4,None)]:
        g=TrackGeometry(x,np.zeros_like(x),np.full_like(x,-1.8),ok=True,
                        sigma_l=np.full_like(x,sigma),sigma_v=np.full_like(x,.025))
        far=FarField(FarFieldConfig())
        for _ in range(25):out=far.update(p,g,30,400)
        if expected is None:assert not out
        else:
            assert out and out[0]['zone']==expected
            assert out[0]['geometry_uncertain']==(expected==1)
