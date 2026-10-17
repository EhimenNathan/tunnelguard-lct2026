"""Реальные фоновые кадры + лучевые инъекции человека и пыли. Отдельная маркировка СИНТЕТИКА."""
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'src/tunnel_guard')]
from falcon_cache import frames_cached
from falcon_inject import inject
from falcon_eval import make_detector
from finals_bundle import save

def main(cache,out):
    det=make_detector([]);rng=np.random.default_rng(17);rows=[]
    for i,(k,t,xyz,intensity) in enumerate(frames_cached(cache,100,210)):
        objects=[dict(kind='person',x=200.,size=(.5,.3,1.75),rho=.1),
                 dict(kind='dust',x=120.,size=(2.4,3.,2.5),sigma=.6,rho=.05,z0=-1.9)] if i>=30 else []
        p,it,_=inject(xyz,intensity,objects,rng);r=det.process(p,t,intensity=it);g=r.geometry
        if i<30 or i%2:continue
        p=r.forward_points
        far=np.flatnonzero((p[:,0]>100)&(p[:,0]<320)&(abs(p[:,1])<7))
        near=np.flatnonzero((p[:,0]>2)&(p[:,0]<=100)&(abs(p[:,1])<7))
        sel=np.r_[rng.choice(far,min(2200,len(far)),False),rng.choice(near,min(1000,len(near)),False)]
        xx=np.arange(5.,min(r.envelope_range,400.)+1,5.) if g.ok else np.array([])
        rows.append(dict(k=k,t=t,level=int(r.level),ms=1000*r.timings['total'],truth_x=200.,
                         env=r.envelope_range,measured=g.valid_range,clear=r.clear_distance,synthetic=True,
                         p=np.round(p[sel].astype(np.float64),2).ravel().tolist(),
                         line=np.round(np.column_stack([xx,g.centre(xx),g.rail_z(xx)]),3).tolist() if len(xx) else [],
                         dust=[dict(d=float(d['distance']),n=int(d['n'])) for d in r.dust],
                         boxes=[dict(c=np.round(o['centroid_fwd'],3).tolist(),size=np.round(o['size'],3).tolist(),
                                     zone=int(o['zone']),d=round(float(o['distance']),1),n=int(o['n'])) for o in r.obstacles]))
    save(Path(out),rows)
    print('synthetic frames',len(rows),'person hit',sum(any(abs(o['d']-200)<5 for o in r['boxes']) for r in rows),flush=True)

if __name__=='__main__':main(*sys.argv[1:3])
