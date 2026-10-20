"""Physical invariants for the opt-in motion candidate, separate from frozen CI."""
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parents[1]/'src/tunnel_guard')]
from types import SimpleNamespace
import numpy as np
from motion_curve_candidate import transform,transport_geometry,motion_from_track,ReprojectedFarField
from tunnel_guard.core.geometry import TrackGeometry
from tunnel_guard.core.farfield import FarFieldConfig


def track(curvature=0.,sigma=.1):
 x=np.arange(501.);return TrackGeometry(x,.5*curvature*x*x,np.full_like(x,-1.8),ok=True,rail_range=40,valid_range=300,trusted_range=300,sigma_l=np.full_like(x,sigma),sigma_v=np.full_like(x,.025))


def test_rigid_transport_matches_world_coordinates():
 p=np.array([[100.,5.,.2],[250.,-2.,1.2]])
 m=(.7,.02,.01,.0014);q=transform(p,m);c,s=np.cos(m[3]),np.sin(m[3]);back=q@np.array([[c,s,0],[-s,c,0],[0,0,1]])+np.array(m[:3])
 np.testing.assert_allclose(back,p,atol=2e-5)


def test_track_transport_retains_target_path_and_sigma():
 g=track(.002);m=motion_from_track(g,.5);new=transport_geometry(g,m);q=transform(np.array([[100.,float(g.centre(100.)),-.8]]),m)
 assert abs(new.centre(q[0,0])-q[0,1])<.003
 assert new.sigma_at([100.])[0][0]>=.1-1e-7
 assert new.valid_range==299.5


def test_subcell_motion_does_not_bias_centroid_or_duplicate_evidence():
 g=track();geometry=SimpleNamespace(motion=(.37,0.,0.,0.));far=ReprojectedFarField(FarFieldConfig(),geometry=geometry)
 for k in range(25):
  p=np.array([[100.-.37*k,0.,-.8]],np.float32);out=far.update(p,g,30,300,travel=.37)
 assert len(out)==1 and abs(out[0]['distance']-p[0,0])<1e-4
 assert out[0]['n']==21 and out[0]['frames']==21


def test_past_scan_cannot_turn_one_hit_into_confirmation():
 g=track();far=ReprojectedFarField(FarFieldConfig());p=np.array([[100.,0.,-.8]],np.float32)
 far.update(p,g,30,300)
 for _ in range(25):assert not far.update(p[:0],g,30,300)


def test_scatter_and_outside_path_never_accumulate_as_stop():
 for outside in [False,True]:
  far=ReprojectedFarField(FarFieldConfig());p=np.array([[100.,3.5 if outside else 0.,-.8]],np.float32)
  for _ in range(30):assert not far.update(p,track(),30,300,exclude=np.array([not outside]))


def test_uncertain_geometry_stays_warning_after_reprojection():
 far=ReprojectedFarField(FarFieldConfig());p=np.array([[100.,0.,-.8]],np.float32)
 for _ in range(25):out=far.update(p,track(sigma=.8),30,300)
 assert out and out[0]['zone']==1


def test_stop_evidence_expires_without_fresh_returns():
 g=track();far=ReprojectedFarField(FarFieldConfig());p=np.array([[100.,0.,-.8]],np.float32)
 for _ in range(12):out=far.update(p,g,30,300)
 assert out and out[0]['zone']==2
 for _ in range(7):out=far.update(p[:0],g,30,300)
 assert out and out[0]['zone']==1
 for _ in range(15):out=far.update(p[:0],g,30,300)
 assert not out
