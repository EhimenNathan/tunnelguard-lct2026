"""Отдельно измеряет информационную зону пыли и ложный СТОП, не смешивая их.
python tools/finals_dust_eval.py CACHE TRUTH OUTPUT.json
"""
import sys,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'src/tunnel_guard')]
from falcon_ladder import DISTANCES,OBJECTS,WARM,ON,CFG,segment_for
from falcon_cache import frames_cached
from falcon_eval import fill_gaps
from falcon_inject import inject
from tunnel_guard.core.detector import ObstacleDetector,DetectorConfig
from finals_bundle import save
def main(cache,truth_path,out):
 truth={r['k']:r for r in fill_gaps(json.loads(Path(truth_path).read_text()))};rows=[]
 for d in DISTANCES:
  det=ObstacleDetector(DetectorConfig(scorer_model=CFG));rng=np.random.default_rng(d);start=segment_for(d,truth)
  flags=stops=returns=0
  for i,(k,t,p,it) in enumerate(frames_cached(cache,start,start+WARM+ON)):
   obs=[dict(OBJECTS['dust'],x=float(d))] if i>=WARM else []
   q,inten,cnt=inject(p,it,obs,rng);res=det.process(q,t,intensity=inten)
   if i<WARM:continue
   flags+=any(abs(z['distance']-d)<5 for z in res.dust)
   stops+=any(abs(o['distance']-d)<4 and o['zone']==2 for o in res.obstacles)
   returns+=cnt[0] if cnt else 0
  row=dict(distance=d,frames=ON,dust_zone_frames=flags,dust_zone_rate=flags/ON,stop_frames=stops,returns_per_frame=returns/ON)
  rows.append(row);save(out,rows);print(row,flush=True)
if __name__=='__main__':main(*sys.argv[1:4])
