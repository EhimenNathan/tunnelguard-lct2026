"""Numerical parity of optimized kernels, including sensor axes and boundaries."""
import numpy as np
import pytest
from tunnel_guard.core import fast
from tunnel_guard.core.cloud import forward_rotation
from tunnel_guard.core.detector import ObstacleDetector,DetectorConfig
from tunnel_guard.core.geometry import TrackGeometry

pytestmark=pytest.mark.skipif(not fast.HAVE_FAST,reason='Numba optional')

def test_fused_preprocessing_preserves_indices_and_points():
    rng=np.random.default_rng(1845)
    xyz=rng.uniform([-100,-70,-10],[200,70,10],(10000,3)).astype(np.float32)
    xyz[:4]=[[0,0,0],[np.nan,0,0],[25,0,0],[24.999,0,0]]
    r2=np.einsum('ij,ij->i',xyz,xyz);valid=np.isfinite(r2)&(r2>2.25)&(r2<250000)
    level=np.array([[.9998,0,.019998],[0,1,0],[-.019998,0,.9998]],np.float64)
    for axis in ['x','-x','y','-y','z','-z','falcon']:
        cfg=DetectorConfig();det=ObstacleDetector(cfg);det.mount.R=level
        R=forward_rotation(axis).astype(np.float32)
        expected,idx,_=det._prepare_reference(xyz,valid,R.copy())
        columns=np.argmax(abs(R),axis=1).astype(np.int64);signs=R[np.arange(3),columns]
        actual,jdx=fast.prepare_forward(xyz,valid,columns,signs,level.astype(np.float32),25.,3,np.tan(np.radians(70.)))
        np.testing.assert_array_equal(idx,jdx)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-5)

def test_fused_track_coordinates_matches_curved_reference():
    rng=np.random.default_rng(842);x=np.arange(501.)
    geom=TrackGeometry(x,.0005*x*x,-1.8+.00001*x*x,roll=.025)
    for dtype in [np.float32,np.float64]:
        p=rng.uniform([-3,-10,-4],[499.5,10,7],(20000,3)).astype(dtype)
        for actual,expected in zip(geom.to_track(p),geom.to_track_reference(p)):
            np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-5)

def test_linear_scatter_search_matches_numpy(monkeypatch):
    rng=np.random.default_rng(301);p=rng.uniform([30,-2,-1],[300,2,2],(12000,3)).astype(np.float32)
    det=ObstacleDetector();actual=det._scatter_flags(p)
    monkeypatch.setattr(fast,'HAVE_FAST',False)
    np.testing.assert_array_equal(actual,det._scatter_flags(p))
