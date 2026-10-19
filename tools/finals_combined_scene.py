"""Rebuild the mixed person+dust diagnostic on the local Falcon cache."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'src/tunnel_guard')]
from falcon_cache import frames_cached
from falcon_eval import make_detector
from falcon_inject import inject,sensor_to_level
from finals_protocol import OBJECTS
from finals_visual_frame import visual_frame
from tunnel_guard.core.geometry import HAVE_NATIVE

def main(cache,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True);det=make_detector([]);rng=np.random.default_rng(7289);view_rng=np.random.default_rng(5123);clips=[];hits=stops=dust=dust_warning=dust_stop=0
    for i,(k,t,p,it) in enumerate(frames_cached(cache,100,180)):
        targets=[dict(OBJECTS['person'],x=200.),dict(OBJECTS['dust'],x=120.)] if i>=30 else []
        q,ii,cnt=inject(p,it,targets,rng);tic=time.perf_counter();res=det.process(q,t,intensity=ii);ms=1000*(time.perf_counter()-tic)
        if i<30:continue
        matched=[o for o in res.obstacles if abs(o['distance']-200)<4 and abs(float(sensor_to_level(np.asarray(o['centroid'])[None])[0,1])-.35)<.8]
        hits+=bool(matched);stops+=any(o['zone']==2 for o in matched);dust_warning+=any(abs(o['distance']-120)<4 for o in res.obstacles);dust_stop+=any(abs(o['distance']-120)<4 and o['zone']==2 for o in res.obstacles);dust+=any(abs(z['distance']-120)<5 for z in res.dust)
        clips.append(visual_frame(res,dict(k=k,local=i-30,t=t,truth_x=200.,synthetic=True,backend='Windows NumPy/Numba' if not HAVE_NATIVE else 'native',source_stage='spatial-20261019',ms=ms),view_rng))
    (out/'clip.json').write_text(json.dumps(clips,separators=(',',':')),encoding='utf-8')
    result=dict(protocol='person at 200m + dust at 120m, same real Falcon background, seed 7289, 30 warmup +50 scored; diagnostic only',native=HAVE_NATIVE,n=50,person_hit=hits,person_stop=stops,dust_flag=dust,dust_near_warning=dust_warning,dust_near_stop=dust_stop,source_sha256={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in (HERE.parent/'src/tunnel_guard/tunnel_guard/core').glob('*.py')})
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cache');p.add_argument('out');a=p.parse_args();main(a.cache,a.out)
