"""Isolate the low-obstacle channel on held-out seeds in the same real backgrounds.
python tools/finals_low_ablation.py CACHE TRUTH OUTPUT_JSON
Shared background segments are not independent trials of real-world accuracy.
"""
import sys,json,hashlib,subprocess,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'src/tunnel_guard')]
from falcon_eval import make_detector,fill_gaps
from falcon_cache import frames_cached
from falcon_inject import inject
from finals_protocol import OBJECTS,segment_for,WARM,ON
from tunnel_guard.core.geometry import warmup

def main(cache,truth_path,out):
 warmup();out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
 truth={r['k']:r for r in fill_gaps(json.loads(Path(truth_path).read_text()))}
 result=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parent,text=True).strip(),
  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE.parent/'src/tunnel_guard/tunnel_guard/core').glob('*.py')},
  method='Only lowfield.enabled differs; 30 warmup + 50 injection frames, same input; distance match <4m; fresh detector per condition; seeds not used in the original ladder; common backgrounds, not independent accuracy estimates',trials=[])
 for kind,d in [('box_0.3',90),('box_0.3',100),('box_0.3',150),('person',300),('dust',150)]:
  start=segment_for(d,truth);frames=list(frames_cached(cache,start,start+WARM+ON))
  for seed in [1001,2002,3003]:
   rng=np.random.default_rng(seed);dets={name:make_detector([f'lowfield.enabled={enabled}']) for name,enabled in [('without_low','false'),('current','true')]}
   measurements={name:dict(hit=0,stop=0,dust=0,first=None) for name in dets}
   for i,(_,t,p,it) in enumerate(frames):
    p,it,_=inject(p,it,[dict(OBJECTS[kind],x=float(d))] if i>=WARM else [],rng)
    for name,det in dets.items():
     res=det.process(p,t,intensity=it)
     if i<WARM:continue
     m=measurements[name];near=[o for o in res.obstacles if abs(o['distance']-d)<4]
     m['hit']+=bool(near);m['stop']+=any(o['zone']==2 for o in near)
     m['dust']+=any(abs(o['distance']-d)<5 for o in res.dust)
     if near and m['first'] is None:m['first']=i-WARM
   row=dict(object=kind,distance=d,seed=seed,segment=start,n=ON,variants=measurements)
   result['trials'].append(row);out.write_text(json.dumps(result,indent=2),encoding='utf-8')
   print(json.dumps(row),flush=True)
if __name__=='__main__':main(*sys.argv[1:4])
