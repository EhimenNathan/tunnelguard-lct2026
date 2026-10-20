"""Paired startup reproduction, prints states before either estimator can fail."""
import sys,json
from pathlib import Path
import numpy as np
from ament_index_python.packages import get_package_share_directory
from tunnel_guard.evaluate_bag import read_bag
from tunnel_guard.core.cloud import cloud_to_arrays
from tunnel_guard.core.detector import ObstacleDetector,DetectorConfig
from tunnel_guard.core.geometry import warmup,HAVE_NATIVE
from motion_curve_candidate import install
warmup();model=str(Path(get_package_share_directory('tunnel_guard'))/'config/obstacle_scorer.json')
dets={'baseline':ObstacleDetector(DetectorConfig(scorer_model=model)),'candidate':install(ObstacleDetector(DetectorConfig(scorer_model=model)))};t0=None;rows=[]
for k,(_,msg) in enumerate(read_bag(sys.argv[1])):
 t=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
 if t0 is None:t0=t
 ar=cloud_to_arrays([(f.name,f.offset,f.datatype,f.count) for f in msg.fields],msg.data,msg.point_step,msg.is_bigendian);xyz=np.column_stack([ar[c] for c in 'xyz']);inten=ar.get('intensity',np.zeros(len(xyz)))
 valid=np.isfinite(xyz).all(1)&(abs(xyz).sum(1)>0)&(abs(xyz).max(1)<650);xyz=np.round(xyz[valid]/.02).astype(np.int16).astype(np.float32)*.02;inten=np.clip(inten[valid],0,255).astype(np.uint8).astype(np.float32)
 for name,d in dets.items():
  print('BEFORE',k,name,'speed',d.calib.ego_speed(),'frame_count',d.geo.frame_count,'motion',getattr(d.geo,'motion',None),flush=True)
  try:r=d.process(xyz,t-t0,intensity=inten)
  except Exception:
   g=d.geo.prev;print('FAILED_STATE',name,k,getattr(d.geo,'motion',None),'sigma',None if g is None else g.sigma_l.tolist(),flush=True);raise
  row=dict(k=k,name=name,geometry=r.geometry.ok,motion=getattr(d.geo,'motion',None),level=r.level);rows.append(row)
  print('AFTER',json.dumps(row),flush=True)
Path(sys.argv[2]).write_text(json.dumps(dict(native=HAVE_NATIVE,rows=rows),indent=2))
