"""Paired frozen/candidate evaluation on the same full ROS bag and 68 injections.

Run inside either ROS image; imports the INSTALLED detector, never repository src.
python3 /tools/finals_improvement_eval.py BAG TRUTH_JSON OUTPUT_DIR [--ladder]
Quality inputs use the original Falcon-cache 2 cm quantization protocol.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from ament_index_python.packages import get_package_share_directory
from tunnel_guard.core.detector import ObstacleDetector,DetectorConfig
from tunnel_guard.core.cloud import cloud_to_arrays
from tunnel_guard.core.geometry import warmup,HAVE_NATIVE
from tunnel_guard.evaluate_bag import read_bag
from tunnel_guard import params as P
import tunnel_guard.core.detector as detector_module
from falcon_inject import inject
from finals_protocol import segment_for,DISTANCES,OBJECTS,WARM,ON,fill_gaps

def main():
    ap=argparse.ArgumentParser();ap.add_argument('bag');ap.add_argument('truth');ap.add_argument('out');ap.add_argument('--ladder',action='store_true');ap.add_argument('--source-label',default='installed image');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    warmup();model=Path(get_package_share_directory('tunnel_guard'))/'config/obstacle_scorer.json'
    def make():return ObstacleDetector(DetectorConfig(scorer_model=str(model)))
    det=make();cfg=P.flatten(det.cfg)
    truth={r['k']:r for r in fill_gaps(json.loads(Path(a.truth).read_text()))}
    starts={d:segment_for(d,truth) for d in DISTANCES}
    wanted=set(k for start in starts.values() for k in range(start,start+WARM+ON)) if a.ladder else set()
    saved={};rows=[];geometry=[];clips=[[],[],[]];t0=None
    X=np.array([25.,50.,100.,150.,200.,250.,300.]);rng_view=np.random.default_rng(20261017)
    for k,(_,msg) in enumerate(read_bag(a.bag)):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if t0 is None:t0=stamp
        t=stamp-t0
        ar=cloud_to_arrays([(f.name,f.offset,f.datatype,f.count) for f in msg.fields],msg.data,msg.point_step,msg.is_bigendian)
        xyz=np.column_stack([ar[c] for c in 'xyz']);inten=ar.get('intensity',np.zeros(len(xyz)))
        valid=np.isfinite(xyz).all(1)&(abs(xyz).sum(1)>0)&(abs(xyz).max(1)<650)
        xyz=np.round(xyz[valid]/.02).astype(np.int16).astype(np.float32)*.02
        inten=np.clip(inten[valid],0,255).astype(np.uint8).astype(np.float32)
        if k in wanted:saved[k]=(t,xyz.copy(),inten.copy())
        tic=time.perf_counter();res=det.process(xyz,t,intensity=inten);ms=1000*(time.perf_counter()-tic)
        tr=truth.get(k,{});known=bool(tr.get('n'));x=tr.get('x') if known else None;y=tr.get('y') if known else None
        lateral=y-float(res.geometry.centre(x)) if known and res.geometry.ok else None
        def match(o):return known and lateral is not None and abs(o['distance']-x)<6 and abs(o['lateral']-lateral)<1
        objects=res.obstacles
        rows.append(dict(k=k,t=t,truth_x=x,truth_y=y,hit=any(match(o) for o in objects),stop_hit=any(match(o) and o['zone']==2 for o in objects),unmatched_stop=any(not match(o) and o['zone']==2 for o in objects),level=res.level,ms=ms,geometry=bool(res.geometry.ok),env=float(res.envelope_range),measured=float(res.geometry.valid_range),truth_interp=bool(tr.get('interp',False)),clear=float(res.clear_distance),obs=[dict(d=float(o['distance']),l=float(o['lateral']),zone=int(o['zone'])) for o in objects]))
        g=res.geometry
        geometry.append(dict(k=k,ok=bool(g.ok),y=g.centre(X).tolist() if g.ok else None,z=g.rail_z(X).tolist() if g.ok else None))
        for ci,(lo,hi) in enumerate([(0,300),(2200,2400),(3800,4000)]):
            if not lo<=k<=hi or k%5:continue
            p=res.forward_points;eligible=np.flatnonzero((p[:,0]>2)&(p[:,0]<410)&(abs(p[:,1])<12)&(p[:,2]>-4)&(p[:,2]<9))
            far=eligible[p[eligible,0]>100];near=eligible[p[eligible,0]<=100]
            special=np.concatenate([np.asarray(o['idx'],int) for o in objects]) if objects else np.zeros(0,int)
            sel=np.unique(np.r_[rng_view.choice(far,min(len(far),1400),False),rng_view.choice(near,min(len(near),1000),False),special])
            if len(sel)>5000:sel=rng_view.choice(sel,5000,False)
            xx=np.arange(5.,min(res.envelope_range,400.)+1,5.) if g.ok else np.array([])
            clips[ci].append(dict(rows[-1],p=np.round(p[sel].astype(np.float64),2).ravel().tolist(),line=np.round(np.column_stack([xx,g.centre(xx),g.rail_z(xx)]),3).tolist() if len(xx) else [],boxes=[dict(c=np.round(o['centroid_fwd'],3).tolist(),size=np.round(o['size'],3).tolist(),zone=int(o['zone']),d=float(o['distance']),n=int(o['n'])) for o in objects]))
        if k%500==0:print('quality',k,flush=True)
    on=[r for r in rows if r['truth_x'] is not None and abs(r['truth_y']-.35)<1.3]
    beside=[r for r in rows if r['truth_x'] is not None and abs(r['truth_y']-.35)>=1.6]
    bands=[]
    for lo,hi in [(0,50),(50,100),(100,150),(150,200),(200,250),(250,320)]:
        rr=[r for r in on if lo<=r['truth_x']<hi]
        bands.append(dict(lo=lo,hi=hi,n=len(rr),hit=sum(r['hit'] for r in rr),stop=sum(r['stop_hit'] for r in rr)))
    first=next((r for r in on if r['hit']),None);ms=[r['ms'] for r in rows[30:]]
    core=Path(detector_module.__file__).parent
    summary=dict(reference='automatic trajectory proxy, not independent ground truth',input_protocol='original Falcon cache: 2 cm xyz, uint8 intensity',frames=len(rows),bands=bands,beside=len(beside),beside_stop=sum(r['stop_hit'] for r in beside),unmatched_stop_frames=sum(r['unmatched_stop'] for r in rows),first_detection=first,geometry_dropouts_after_30=sum(not r['geometry'] for r in rows[30:]),latency_ms=dict(p50=float(np.median(ms)),p95=float(np.percentile(ms,95)),p99=float(np.percentile(ms,99)),over100=sum(v>100 for v in ms)),native=HAVE_NATIVE,config=cfg,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in core.glob('*.py')})
    steady=rows[30:];ev=np.array([r['env'] for r in steady]);Y=np.array([gg['y'] if gg['ok'] else [np.nan]*len(X) for gg in geometry[30:]])
    known=[r for r in rows if r['truth_x'] is not None];first_stop=next((r for r in on if r['stop_hit']),None)
    summary.update(schema=1,truth_sha256=hashlib.sha256(Path(a.truth).read_bytes()).hexdigest(),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),source_label=a.source_label,on_track=len(on),known_reference_frames=len(known),unknown_reference_frames=len(rows)-len(known),interpolated_frames=sum(r['truth_interp'] for r in rows),first_stop=first_stop,envelope_median_m=float(np.median(ev)),envelope_p90_m=float(np.percentile(ev,90)),envelope_max_m=float(ev.max()),measured_median_m=float(np.median([r['measured'] for r in steady])),envelope_change_p95_m=float(np.percentile(abs(np.diff(ev)),95)),jitter={str(int(x)):dict(p50_cm=float(np.nanmedian(abs(np.diff(Y[:,i])))*100),p95_cm=float(np.nanpercentile(abs(np.diff(Y[:,i])),95)*100)) for i,x in enumerate(X)})
    (out/'geometry.json').write_text(json.dumps(geometry),encoding='utf-8')
    for i,clip in enumerate(clips):(out/f'falcon_clip_{i}.json').write_text(json.dumps(clip),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'rows.json').write_text(json.dumps(rows),encoding='utf-8')
    print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in ['config','source_sha256']}),flush=True)
    if not a.ladder:return
    ladder=[]
    for name,ob in OBJECTS.items():
        for d in DISTANCES:
            start=starts[d];rng=np.random.default_rng(d);det=make();hits=[];stops=[];dust=[];returns=[];first=None
            for i,k in enumerate(range(start,start+WARM+ON)):
                t,p,it=saved[k];p,it,cnt=inject(p,it,[dict(ob,x=float(d))] if i>=WARM else [],rng)
                res=det.process(p,t,intensity=it)
                if i<WARM:continue
                near=[o for o in res.obstacles if abs(o['distance']-d)<4]
                hits.append(bool(near));stops.append(any(o['zone']==2 for o in near));returns.append(cnt[0] if cnt else 0)
                dust.append(any(abs(z['distance']-d)<5 for z in (res.dust or [])))
                if near and first is None:first=i-WARM
            row=dict(object=name,distance=d,segment=start,detected=float(np.mean(hits)),stop=float(np.mean(stops)),dust_zone_rate=float(np.mean(dust)),first_frame=first,returns_per_frame=float(np.mean(returns)))
            ladder.append(row);print('ladder',json.dumps(row),flush=True)
            (out/'ladder.json').write_text(json.dumps(ladder,indent=2),encoding='utf-8')

if __name__=='__main__':main()
