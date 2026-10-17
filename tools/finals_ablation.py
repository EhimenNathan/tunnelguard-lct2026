"""Текущая абляция Falcon: одинаковые первые 301 кадр, свежий детектор на вариант.
Эталон автоматический, это диагностическая абляция, не независимая оценка FP/FN.
python tools/finals_ablation.py CACHE TRUTH OUTPUT.json
"""
import sys,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'src/tunnel_guard')]
from falcon_cache import frames_cached
from falcon_eval import fill_gaps,TRACK_Y
from tunnel_guard.core.detector import ObstacleDetector,DetectorConfig
from finals_bundle import save
MODEL=str(HERE.parent/'src/tunnel_guard/config/obstacle_scorer.json')
def main(cache,truth_path,out):
 truth={r['k']:r for r in fill_gaps(json.loads(Path(truth_path).read_text()))};rows=[]
 for variant in ['full','no_far','no_mount','no_ml','no_scatter']:
  cfg=DetectorConfig(scorer_model=MODEL)
  if variant=='no_far':cfg.farfield.enabled=False
  if variant=='no_mount':cfg.mount_calibration=False
  if variant=='no_ml':cfg.scorer_model='none'
  if variant=='no_scatter':cfg.scatter_check=False
  det=ObstacleDetector(cfg);den=hits=stops=0;first=None
  for k,t,p,inten in frames_cached(cache,0,301):
   res=det.process(p,t,intensity=inten);tr=truth.get(k,{})
   if not tr.get('n') or abs(tr['y']-TRACK_Y)>=1.3:continue
   den+=1;lat=tr['y']-float(res.geometry.centre(tr['x']))
   found=[o for o in res.obstacles if abs(o['distance']-tr['x'])<6 and abs(o['lateral']-lat)<1]
   hits+=bool(found);stops+=any(o['zone']==2 for o in found)
   if found and first is None:first=dict(k=k,distance=tr['x'])
  row=dict(variant=variant,frames=301,on_track=den,hits=hits,stops=stops,hit_rate=hits/den if den else None,stop_rate=stops/den if den else None,first=first)
  rows.append(row);save(out,rows);print(row,flush=True)
if __name__=='__main__':main(*sys.argv[1:4])
