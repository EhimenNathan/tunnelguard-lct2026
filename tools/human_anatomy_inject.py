"""Anatomical ellipsoid union, evaluation only; frozen box ladder is unchanged."""
import numpy as np
from falcon_inject import FLOOR,TRACK_Y,level_to_sensor,sensor_to_level,ray_grid,_dirs,p_detect

def anatomy(x,y=TRACK_Y,phase=0.):
 # Coordinates: forward, lateral, height above bed; adult height 1.75m.
 parts=[('head',(.16,0,1.61),(.11,.11,.14)),('torso',(.16,0,1.18),(.145,.19,.30)),('hips',(.16,0,.87),(.14,.17,.16))]
 swing=.10*np.sin(phase)
 for side in [-1,1]:
  parts += [(f'upper_arm_{side}',(.16+side*swing,side*.255,1.14),(.065,.065,.22)),(f'forearm_{side}',(.16+side*swing*1.5,side*.27,.83),(.055,.055,.15)),(f'thigh_{side}',(.16-side*swing,side*.105,.59),(.08,.075,.25)),(f'shin_{side}',(.16-side*swing*1.5,side*.105,.20),(.065,.06,.20))]
 return [(name,np.array([x+v[0],y+v[1],FLOOR+v[2]]),np.array(rad)) for name,v,rad in parts]

def ray_hits(d,parts):
 result=np.full(len(d),np.inf)
 for _,centre,radius in parts:
  a=np.sum((d/radius)**2,axis=1);b=-2*np.sum(d*centre/radius**2,axis=1);c=np.sum((centre/radius)**2)-1;disc=b*b-4*a*c
  entry=(-b-np.sqrt(np.maximum(disc,0)))/(2*a);entry=np.where((disc>=0)&(entry>0),entry,np.inf);result=np.minimum(result,entry)
 return result

def inject_human(xyz,intensity,x,y,rng,phase=0.,rho=.10):
 parts=anatomy(x,y,phase);lo=np.min([c-r for _,c,r in parts],axis=0);hi=np.max([c+r for _,c,r in parts],axis=0)
 cs=level_to_sensor(np.array([[a,b,c] for a in [lo[0],hi[0]] for b in [lo[1],hi[1]] for c in [lo[2],hi[2]]]))
 az=np.degrees(np.arctan2(cs[:,1],cs[:,0]));el=np.degrees(np.arctan2(cs[:,2],np.hypot(cs[:,0],cs[:,1])))
 aa,ee=ray_grid(az.min(),az.max(),el.min(),el.max(),rng);ds=_dirs(aa,ee);dl=sensor_to_level(ds);entry=ray_hits(dl,parts);valid=np.isfinite(entry);ret=valid&(rng.random(len(entry))<p_detect(entry,rho));points=(ds[ret]*entry[ret,None]).astype(np.float32)
 rr=np.linalg.norm(xyz,axis=1);directions=sensor_to_level(xyz/np.maximum(rr[:,None],1e-6));cross=ray_hits(directions,parts);keep=~(np.isfinite(cross)&(rr>cross))
 return np.vstack([xyz[keep],points]),np.r_[intensity[keep],np.full(len(points),25,np.float32)],points
