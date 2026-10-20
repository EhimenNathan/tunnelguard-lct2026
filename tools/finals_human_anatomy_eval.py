"""51 new diagnostics, anatomical adult on-track and outside on both sides."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parent/'src/tunnel_guard')]
from falcon_cache import frames_cached
from falcon_eval import make_detector,fill_gaps
from falcon_inject import sensor_to_level,TRACK_Y
from human_anatomy_inject import inject_human
from finals_protocol import DISTANCES
from tunnel_guard.core.geometry import HAVE_NATIVE,warmup

def main(cache,truth_path,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);warmup();truth={r['k']:r for r in fill_gaps(json.loads(Path(truth_path).read_text()))};pool=[1500,2200,3000,3600,4000,600];frames={s:list(frames_cached(cache,s,s+80)) for s in pool};trials=[];samples=[];start_time=time.time()
 for d in DISTANCES:
  start=next(s for s in pool if all(not truth.get(k,{}).get('n') or abs(truth[k]['x']-d)>25 for k in range(s,s+80)))
  for side,y in [('inside',TRACK_Y),('outside_left',TRACK_Y+3.5),('outside_right',TRACK_Y-3.5)]:
   rng=np.random.default_rng(88000+d);det=make_detector([]);hit=stop=0;returns=[];first=None
   for i,(k,t,p,it) in enumerate(frames[start]):
    if i>=30:q,ii,pts=inject_human(p,it,float(d),y,rng,phase=(i-30)*.18)
    else:q,ii,pts=p,it,np.zeros((0,3))
    res=det.process(q,t,intensity=ii)
    if i<30:continue
    matched=[o for o in res.obstacles if abs(float(o['distance'])-d)<4 and abs(float(sensor_to_level(np.asarray(o['centroid'])[None])[0,1])-y)<.8];hit+=bool(matched);stop+=any(o['zone']==2 for o in matched);returns.append(len(pts))
    if matched and first is None:first=i-30
    if i==79 and d in [60,150,300]:samples.append(dict(distance=d,side=side,y=y,phase=(i-30)*.18,p=sensor_to_level(pts).tolist(),detected=bool(matched),stop=any(o['zone']==2 for o in matched)))
   row=dict(distance=d,side=side,y=y,seed=88000+d,start=start,n=50,hit=hit,stop=stop,returns_mean=float(np.mean(returns)),first_frame=first);trials.append(row);(out/'trials.json').write_text(json.dumps(trials,indent=2));print('HUMAN',json.dumps(row),'elapsed',round(time.time()-start_time),flush=True)
 result=dict(native=HAVE_NATIVE,backend='native' if HAVE_NATIVE else 'Windows NumPy/Numba',definition='Adult 1.75m, ellipsoid head/torso/hips/arms/legs; physical angular sampling; 17 ranges, inside and +/-3.5m outside; 30warm+50scored; one seed per range, known background windows; diagnostic, not independent real-world accuracy',trials=trials,source_sha256={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in (HERE.parent/'src/tunnel_guard/tunnel_guard/core').glob('*.py')})
 (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'samples.json').write_text(json.dumps(samples));print('Anatomy evaluation finished',round(time.time()-start_time),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cache');p.add_argument('truth');p.add_argument('out');a=p.parse_args();main(a.cache,a.truth,a.out)

