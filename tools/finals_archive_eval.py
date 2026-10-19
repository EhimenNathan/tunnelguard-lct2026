"""Потоковый прогон локальных zst/tar ROS-бэгов без распаковки и без нового кеша.
Сохраняет решения и небольшие 3D-фрагменты; без независимого эталона FP/FN не вычисляет.
python tools/finals_archive_eval.py ARCHIVE OUTPUT_DIR
"""
import sys,json,time,tarfile,hashlib,subprocess,argparse
from pathlib import Path
import numpy as np
import zstandard
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'src/tunnel_guard')]
from sqlstream import iter_rows
from sqlpages import split_record,parse_pc2
from falcon_stream import decode
from falcon_eval import make_detector
from finals_bundle import save
from tunnel_guard import params as P

def main(src,output,clip_stride=25,clip_start=0,clip_stop=2500,clip_limit=100):
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 det=make_detector([]);rows=[];clips=[];meta=[];rng=np.random.default_rng(17);started=time.time()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parent,text=True).strip()
 source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE.parent/'src/tunnel_guard/tunnel_guard/core').glob('*.py')}
 config=P.flatten(det.cfg)
 error=None
 try:
  with open(src,'rb') as raw,zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(raw) as z,tarfile.open(fileobj=z,mode='r|') as tf:
   for m in tf:
    if m.name.endswith(('.yaml','.json','.txt')) and m.size<100000:
     content=tf.extractfile(m).read().decode('utf-8',errors='replace');meta.append(dict(name=m.name,content=content));continue
    if not m.name.endswith('.db3'):continue
    print('piece',m.name,m.size,flush=True);det.reset();first=None;local=0
    for _,payload in iter_rows(tf.extractfile(m)):
     vals=split_record(payload)
     if len(vals)<4 or not isinstance(vals[3],(bytes,bytearray)) or len(vals[3])<100000:continue
     pc=parse_pc2(vals[3]);a=decode(pc)
     if not all(key in a for key in 'xyz'):continue
     p=np.column_stack([a[k] for k in 'xyz']).astype(np.float32)
     keep=np.isfinite(p).all(1)&(np.abs(p).sum(1)>0)
     p=p[keep];inten=a['intensity'][keep] if 'intensity' in a else None
     if first is None:
      first=pc['stamp'];print('frame schema',pc['frame'],pc['fields'],'points',len(p),flush=True)
     ring=a.get('ring',a.get('channel'));ring=ring[keep] if ring is not None else None
     t=pc['stamp']-first;tic=time.perf_counter();res=det.process(p,t,intensity=inten,ring=ring);ms=(time.perf_counter()-tic)*1000;g=res.geometry
     row=dict(k=len(rows),piece=m.name,local=local,t=t,points=len(p),ms=ms,level=int(res.level),geometry=bool(g.ok),
       measured=float(g.valid_range),env=float(res.envelope_range),clear=float(res.clear_distance),
       dust=[dict(d=float(d['distance']),n=int(d['n'])) for d in res.dust],
       obs=[dict(d=float(o['distance']),l=float(o['lateral']),zone=int(o['zone']),n=int(o['n'])) for o in res.obstacles])
     rows.append(row)
     if clip_start<=local<clip_stop and local%clip_stride==0 and len(clips)<clip_limit:
      q=res.forward_points;sel=np.flatnonzero((q[:,0]>2)&(q[:,0]<410)&(abs(q[:,1])<15)&(q[:,2]>-5)&(q[:,2]<10))
      special=np.unique(np.concatenate([np.asarray(o['idx'],int) for o in res.obstacles])) if res.obstacles else np.zeros(0,int)
      special=special[(special>=0)&(special<len(q))]
      if len(special)>2000:special=rng.choice(special,2000,False)
      far=sel[q[sel,0]>80];near=sel[q[sel,0]<=80]
      context=np.unique(np.r_[rng.choice(far,min(1400,len(far)),False),rng.choice(near,min(1800,len(near)),False)])
      sel=np.unique(np.r_[context,special])
      xx=np.arange(5.,min(res.envelope_range,400.)+1,5.) if g.ok else np.array([])
      clips.append(dict(row,truth_x=None,p=np.round(q[sel].astype(np.float64),2).ravel().tolist(),
        line=np.round(np.column_stack([xx,g.centre(xx),g.rail_z(xx)]),3).tolist() if len(xx) else [],
        boxes=[dict(c=np.round(o['centroid_fwd'],3).tolist(),size=np.round(o['size'],3).tolist(),zone=int(o['zone']),d=float(o['distance']),n=int(o['n'])) for o in res.obstacles]))
     local+=1
     if local%100==0:print(len(rows),'frames',round(time.time()-started,1),'s','stop',sum(r['level']==2 for r in rows),'dust_frames',sum(bool(r['dust']) for r in rows),flush=True)
 except (zstandard.ZstdError,tarfile.ReadError,EOFError,OSError) as exc:
  error=f'{type(exc).__name__}: {exc}';print('ARCHIVE ERROR:',error,flush=True)
 save(out/'rows.json',rows);save(out/'clip.json',clips);save(out/'metadata.json',meta)
 steady=[r for r in rows if r['local']>=30];lat=[r['ms'] for r in steady]
 summary=dict(complete=error is None,archive_error=error,archive=Path(src).name,archive_bytes=Path(src).stat().st_size,commit=commit,source_sha256=source_sha256,config=config,frames=len(rows),
   duration_s=sum(max(r['t'] for r in rows if r['piece']==piece) for piece in {r['piece'] for r in rows}),pieces=len({r['piece'] for r in rows}),stop_frames=sum(r['level']==2 for r in rows),
   caution_frames=sum(r['level']==1 for r in rows),geometry_ok=sum(r['geometry'] for r in steady),steady_frames=len(steady),
   dust_flagged_frames=sum(bool(r['dust']) for r in rows),
   obstacle_range_max=max((o['d'] for r in rows for o in r['obs']),default=None),
   p95_ms=float(np.percentile(lat,95)) if lat else None,
   ground_truth='Нет независимых меток классов/препятствий; число dust flags не доказывает наличие настоящей пыли; FP/FN неизвестны')
 save(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
 if error:raise SystemExit(2)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('archive');ap.add_argument('output');ap.add_argument('--clip-stride',type=int,default=25);ap.add_argument('--clip-start',type=int,default=0);ap.add_argument('--clip-stop',type=int,default=2500);ap.add_argument('--clip-limit',type=int,default=100);a=ap.parse_args()
 assert a.clip_stride>0
 main(a.archive,a.output,a.clip_stride,a.clip_start,a.clip_stop,a.clip_limit)
