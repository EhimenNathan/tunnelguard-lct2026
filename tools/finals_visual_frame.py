"""Compact visualization of actual detector output, preserving object support."""
import numpy as np

def visual_frame(res,row,rng):
    p=res.forward_points;g=res.geometry
    eligible=np.flatnonzero((p[:,0]>2)&(p[:,0]<410)&(abs(p[:,1])<15)&(p[:,2]>-5)&(p[:,2]<10))
    special=np.unique(np.concatenate([np.asarray(o['idx'],int) for o in res.obstacles])) if res.obstacles else np.zeros(0,int)
    if len(special)>2000:special=rng.choice(special,2000,False)
    far=eligible[p[eligible,0]>80];near=eligible[p[eligible,0]<=80]
    sel=np.unique(np.r_[special,rng.choice(far,min(1400,len(far)),False),rng.choice(near,min(1800,len(near)),False)])
    xx=np.arange(5.,min(res.envelope_range,400.)+1,5.) if g.ok else np.array([])
    return dict(row,level=int(res.level),measured=float(g.valid_range),env=float(res.envelope_range),clear=float(res.clear_distance),
        p=np.round(p[sel].astype(np.float64),2).ravel().tolist(),line=np.round(np.column_stack([xx,g.centre(xx),g.rail_z(xx)]),3).tolist() if len(xx) else [],
        dust=[dict(d=float(d['distance']),n=int(d['n'])) for d in res.dust],
        boxes=[dict(c=np.round(o['centroid_fwd'],3).tolist(),size=np.round(o['size'],3).tolist(),zone=int(o['zone']),d=float(o['distance']),n=int(o['n'])) for o in res.obstacles])
