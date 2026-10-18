"""Installed ROS detector: full raw organizer bags plus truthful sampled 3D output."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from tunnel_guard.core.detector import ObstacleDetector,DetectorConfig
from tunnel_guard.core.geometry import warmup,HAVE_NATIVE
from tunnel_guard.core.cloud import cloud_to_arrays
from tunnel_guard.evaluate_bag import read_bag
from tunnel_guard import params as P
import tunnel_guard.core.detector as detector_module

def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
def main(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);warmup();rng=np.random.default_rng(17)
 model=Path(get_package_share_directory('tunnel_guard'))/'config/obstacle_scorer.json';rows=[];clips=[];pieces=[];schemas=[]
 for name,expected in [('synthetic_platform_cloud_only',386),('synthetic_switch_cloud_only',736),('synthetic_tunnels_cloud_only',545)]:
  bag=root/name;det=ObstacleDetector(DetectorConfig(scorer_model=str(model)));t0=None;begin=len(rows)
  for k,(_,msg) in enumerate(read_bag(str(bag))):
   stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
   if t0 is None:t0=stamp;schemas.append(dict(piece=name,frame=msg.header.frame_id,fields=[dict(name=f.name,offset=f.offset,datatype=f.datatype,count=f.count) for f in msg.fields]))
   a=cloud_to_arrays([(f.name,f.offset,f.datatype,f.count) for f in msg.fields],msg.data,msg.point_step,msg.is_bigendian)
   p=np.column_stack([a[c] for c in 'xyz']);valid=np.isfinite(p).all(1)&(abs(p).sum(1)>0);p=p[valid]
   inten=a.get('intensity');inten=inten[valid] if inten is not None else None;ring=a.get('ring',a.get('channel'));ring=ring[valid] if ring is not None else None
   tic=time.perf_counter();res=det.process(p,stamp-t0,intensity=inten,ring=ring);ms=1000*(time.perf_counter()-tic);g=res.geometry
   row=dict(k=len(rows),piece=name+'/'+name+'_0.db3',local=k,t=stamp-t0,points=len(p),ms=ms,level=int(res.level),geometry=bool(g.ok),measured=float(g.valid_range),env=float(res.envelope_range),clear=float(res.clear_distance),dust=[dict(d=float(z['distance']),n=int(z['n'])) for z in res.dust],obs=[dict(d=float(o['distance']),l=float(o['lateral']),zone=int(o['zone']),n=int(o['n']),uncertain=bool(o.get('geometry_uncertain'))) for o in res.obstacles]);rows.append(row)
   if k%25==0:
    q=res.forward_points;eligible=np.flatnonzero((q[:,0]>2)&(q[:,0]<410)&(abs(q[:,1])<15)&(q[:,2]>-5)&(q[:,2]<10));special=np.unique(np.concatenate([np.asarray(o['idx'],int) for o in res.obstacles])) if res.obstacles else np.zeros(0,int)
    if len(special)>2000:special=rng.choice(special,2000,False)
    far=eligible[q[eligible,0]>80];near=eligible[q[eligible,0]<=80];sel=np.unique(np.r_[special,rng.choice(far,min(1400,len(far)),False),rng.choice(near,min(1800,len(near)),False)])
    xx=np.arange(5.,min(res.envelope_range,400.)+1,5.) if g.ok else np.array([])
    clips.append(dict(row,truth_x=None,organizer_synth=True,dataset_label='Синтетическая запись организаторов',p=np.round(q[sel].astype(np.float64),2).ravel().tolist(),line=np.round(np.column_stack([xx,g.centre(xx),g.rail_z(xx)]),3).tolist() if len(xx) else [],boxes=[dict(c=np.round(o['centroid_fwd'],3).tolist(),size=np.round(o['size'],3).tolist(),zone=int(o['zone']),d=float(o['distance']),n=int(o['n']),uncertain=bool(o.get('geometry_uncertain'))) for o in res.obstacles]))
  rr=rows[begin:];assert len(rr)==expected
  pieces.append(dict(piece=name+'/'+name+'_0.db3',frames=len(rr),stop=sum(r['level']==2 for r in rr),caution=sum(r['level']==1 for r in rr),dust_frames=sum(bool(r['dust']) for r in rr),duration_s=max(r['t'] for r in rr)));print(name,len(rr),'complete',flush=True)
 steady=[r for r in rows if r['local']>=30];core=Path(detector_module.__file__).parent
 summary=dict(archive='new_synth_data.zst',complete=True,frames=len(rows),pieces=3,commit=os.environ.get('TG_SOURCE_SHA'),native=HAVE_NATIVE,config=P.flatten(det.cfg),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in core.glob('*.py')},model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),by_piece=pieces,duration_s=sum(p['duration_s'] for p in pieces),stop_frames=sum(r['level']==2 for r in rows),caution_frames=sum(r['level']==1 for r in rows),dust_flagged_frames=sum(bool(r['dust']) for r in rows),geometry_ok=sum(r['geometry'] for r in steady),steady_frames=len(steady),obstacle_range_max=max((o['d'] for r in rows for o in r['obs']),default=None),p95_ms=float(np.percentile([r['ms'] for r in steady],95)),ground_truth='Нет независимых меток классов/пыли: выходы не равны TP/FP, FP/FN неизвестны')
 save(out/'summary.json',summary);save(out/'rows.json',rows);save(out/'clip.json',clips);save(out/'schema.json',schemas)
if __name__=='__main__':main(*sys.argv[1:3])
