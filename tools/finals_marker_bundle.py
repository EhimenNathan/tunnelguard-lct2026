"""Inspect organizer markers and evaluate the frozen detector. No class guesses."""
import argparse,hashlib,json,sys,time,tarfile,shutil
from pathlib import Path,PurePosixPath
import numpy as np
import yaml,zstandard

def extract(src,out):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
 with open(src,'rb') as raw,zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(raw) as z,tarfile.open(fileobj=z,mode='r|') as tf:
  for m in tf:
   p=PurePosixPath(m.name)
   if p.is_absolute() or '..' in p.parts or not(m.isfile() or m.isdir()):raise ValueError('Unsafe archive member')
   dest=(out/str(p)).resolve();assert dest.is_relative_to(out)
   if m.isdir():dest.mkdir(parents=True,exist_ok=True);continue
   if m.size>shutil.disk_usage(out).free-256*1024**2:raise OSError('Cloud disk full')
   dest.parent.mkdir(parents=True,exist_ok=True)
   with tf.extractfile(m) as s,dest.open('wb') as d:shutil.copyfileobj(s,d,4*1024**2)
 print('Extracted marker archive',flush=True)

def evaluate(root,out):
 import rosbag2_py
 from rclpy.serialization import deserialize_message
 from rosidl_runtime_py.utilities import get_message
 from tunnel_guard.core.cloud import cloud_to_arrays
 from tunnel_guard.core.geometry import HAVE_NATIVE,warmup
 from falcon_eval import make_detector
 import tunnel_guard.core.detector as dm
 out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];markers=[];inventory=[];clips=[];rng=np.random.default_rng(7289);warmup()
 def dump(name,value):
  (out/name).write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'),default=lambda x:x.tolist() if hasattr(x,'tolist') else float(x)),encoding='utf-8')
 def vector(v):return [float(v.x),float(v.y),float(v.z)]
 for meta in sorted(Path(root).rglob('metadata.yaml')):
  piece=meta.parent.name;reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=str(meta.parent),storage_id='sqlite3'),rosbag2_py.ConverterOptions('',''))
  topics={t.name:t.type for t in reader.get_all_topics_and_types()};types={};inventory.append(dict(piece=piece,topics=topics));dump('inventory.json',inventory);print('TOPICS',piece,topics,flush=True)
  for topic,typ in topics.items():
   try:types[topic]=get_message(typ)
   except Exception as e:print('Unsupported message',topic,typ,type(e).__name__,flush=True)
  det=make_detector([]);first=None;local=0;mi=0
  while reader.has_next():
   topic,data,stamp=reader.read_next();typ=topics[topic]
   if topic not in types:continue
   if typ=='sensor_msgs/msg/PointCloud2':
    msg=deserialize_message(data,types[topic]);t=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    if first is None:first=t
    ar=cloud_to_arrays([(f.name,f.offset,f.datatype,f.count) for f in msg.fields],msg.data,msg.point_step,msg.is_bigendian);p=np.column_stack([ar[c] for c in 'xyz']);it=ar.get('intensity');ring=ar.get('ring',ar.get('channel'))
    tic=time.perf_counter();res=det.process(p,t-first,intensity=it,ring=ring);g=res.geometry
    row=dict(piece=piece,local=local,bag_stamp_ns=stamp,header_stamp=t,frame=msg.header.frame_id,points=len(p),cloud_data_sha256=hashlib.sha256(bytes(msg.data)).hexdigest(),ms=(time.perf_counter()-tic)*1000,level=int(res.level),rotation=res.rotation,geometry=bool(g.ok),measured=float(g.valid_range),env=float(res.envelope_range),obs=[dict(c=o['centroid'],c_fwd=o['centroid_fwd'],size=o['size'],d=float(o['distance']),l=float(o['lateral']),zone=int(o['zone']),n=int(o['n']),held=bool(o.get('held',False))) for o in res.obstacles],dust=[{k:v for k,v in z.items() if k!='idx'} for z in res.dust])
    rows.append(row)
    if local%10==0:
     q=res.forward_points;sel=np.flatnonzero((q[:,0]>2)&(q[:,0]<410)&(np.abs(q[:,1])<18));sel=rng.choice(sel,min(len(sel),3000),False);xx=np.arange(5.,min(res.envelope_range,400.)+1,5.) if g.ok else np.array([])
     clips.append(dict(piece=piece,local=local,t=t-first,level=int(res.level),p=np.round(q[sel],2),line=np.column_stack([xx,g.centre(xx),g.rail_z(xx)]) if len(xx) else [],rotation=res.rotation,boxes=[dict(c=o['centroid_fwd'],size=o['size'],zone=int(o['zone']),d=float(o['distance'])) for o in res.obstacles]))
    local+=1
    if local%100==0:print('EVAL',piece,local,flush=True)
   elif 'Marker' in typ:
    msg=deserialize_message(data,types[topic]);items=msg.markers if hasattr(msg,'markers') else [msg]
    for m in items:
     value=dict(piece=piece,topic=topic,bag_stamp_ns=stamp,header_stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,frame=m.header.frame_id,ns=m.ns,id=m.id,type=m.type,action=m.action,text=m.text,c=vector(m.pose.position),q=[m.pose.orientation.x,m.pose.orientation.y,m.pose.orientation.z,m.pose.orientation.w],scale=vector(m.scale),color=[m.color.r,m.color.g,m.color.b,m.color.a],points=[vector(p) for p in m.points[:1000]])
     markers.append(value);mi+=1
     if mi<=8:print('MARKER',json.dumps(value),flush=True)
  dump('rows.json',rows);dump('markers.json',markers);dump('clips.json',clips)
 summary=dict(native=HAVE_NATIVE,frames=len(rows),marker_records=len(markers),pieces=len(inventory),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(dm.__file__).parent.glob('*.py')},marker_types=sorted({(m['ns'],m['type'],m['text']) for m in markers}),status='Geometry references exported; class/event metrics require marker semantics review. No inferred class labels.')
 dump('summary.json',summary);print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['extract','evaluate']);a.add_argument('source');a.add_argument('out');v=a.parse_args();(extract if v.mode=='extract' else evaluate)(v.source,v.out)
