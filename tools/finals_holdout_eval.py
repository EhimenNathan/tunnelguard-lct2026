"""Frozen multi-background holdout protocol; run in either installed ROS image.

python3 /tools/finals_holdout_eval.py BAG TRUTH OUTPUT_DIR --label LABEL
No repository detector source is imported. First 30 frames warm, next 50 scored.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from ament_index_python.packages import get_package_share_directory
from tunnel_guard.core.detector import ObstacleDetector,DetectorConfig
from tunnel_guard.core.cloud import cloud_to_arrays
from tunnel_guard.core.geometry import warmup,HAVE_NATIVE
from tunnel_guard.evaluate_bag import read_bag
from tunnel_guard import params as P
import tunnel_guard.core.detector as detector_module
from falcon_inject import inject,sensor_to_level
from finals_protocol import DISTANCES,OBJECTS,WARM,ON,fill_gaps

SEEDS=[104729,130363,155921]
BACKGROUND_POOL=[1500,2200,3000,3600,4000,600]

def protocol(truth):
    result=[]
    for kind in [*OBJECTS,'off_track_person']:
        for distance in DISTANCES:
            valid=[]
            for start in BACKGROUND_POOL:
                seg=[truth.get(k,{}) for k in range(start,start+WARM+ON)]
                if all(not r.get('n') or abs(r['x']-distance)>25 for r in seg):valid.append(start)
            if len(valid)<3:raise ValueError('Not enough uncontaminated backgrounds')
            for seed,start in zip(SEEDS,valid[:3]):result.append(dict(object=kind,distance=distance,seed=seed,start=start))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('bag');ap.add_argument('truth');ap.add_argument('out');ap.add_argument('--label',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);warmup()
    truth={r['k']:r for r in fill_gaps(json.loads(Path(a.truth).read_text()))};plan=protocol(truth)
    (out/'protocol.json').write_text(json.dumps(plan,indent=2))
    wanted={k for row in plan for k in range(row['start'],row['start']+WARM+ON)};saved={};t0=None
    for k,(_,msg) in enumerate(read_bag(a.bag)):
        if t0 is None:t0=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if k not in wanted:continue
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        ar=cloud_to_arrays([(f.name,f.offset,f.datatype,f.count) for f in msg.fields],msg.data,msg.point_step,msg.is_bigendian)
        xyz=np.column_stack([ar[c] for c in 'xyz']);inten=ar.get('intensity',np.zeros(len(xyz)))
        valid=np.isfinite(xyz).all(1)&(abs(xyz).sum(1)>0)&(abs(xyz).max(1)<650)
        xyz=np.round(xyz[valid]/.02).astype(np.int16).astype(np.float32)*.02
        inten=np.clip(inten[valid],0,255).astype(np.uint8).astype(np.float32)
        saved[k]=(stamp-t0,xyz,inten)
    assert set(saved)==wanted
    model=Path(get_package_share_directory('tunnel_guard'))/'config/obstacle_scorer.json'
    def make():return ObstacleDetector(DetectorConfig(scorer_model=str(model)))
    core=Path(detector_module.__file__).parent
    result=dict(label=a.label,native=HAVE_NATIVE,definition='Synthetic target in shared real backgrounds; strict sensor-frame association after fixed known mounting rotation; not independent real-world accuracy',
       config=P.flatten(make().cfg),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in core.glob('*.py')},
       protocol_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),trials=[])
    for case in plan:
        kind=case['object'];d=case['distance'];ob=dict(OBJECTS['person'] if kind=='off_track_person' else OBJECTS[kind],x=float(d))
        if kind=='off_track_person':ob['y']=3.5
        rng=np.random.default_rng(case['seed']);det=make();hit=stop=near_warning=dust=visible=0;first=None;returns=[];geometry_ok=0
        for i,k in enumerate(range(case['start'],case['start']+WARM+ON)):
            t,p,it=saved[k];q,ii,cnt=inject(p,it,[ob] if i>=WARM else [],rng);res=det.process(q,t,intensity=ii)
            if i<WARM:continue
            geometry_ok+=bool(res.geometry.ok);n=cnt[0];returns.append(n);visible+=n>0
            nearby=[o for o in res.obstacles if abs(o['distance']-d)<4]
            # Centroids are in original sensor axes; the injection's known
            # pitch defines truth independently of the detector's fitted path.
            matched=[o for o in nearby if abs(float(sensor_to_level(np.asarray(o['centroid'])[None])[0,1])-ob.get('y',.35))<.8]
            hit+=bool(matched);stop+=any(o['zone']==2 for o in matched);near_warning+=bool(nearby)
            dust+=any(abs(z['distance']-d)<5 for z in res.dust)
            if matched and first is None:first=i-WARM
        row=dict(case,n=ON,hit=hit,stop=stop,near_warning=near_warning,dust=dust,first_frame=first,returns_mean=float(np.mean(returns)),frames_with_returns=visible,geometry_ok=geometry_ok)
        result['trials'].append(row);(out/'holdout.json').write_text(json.dumps(result,indent=2));print('HOLDOUT',json.dumps(row),flush=True)
    summary=dict(label=a.label,trials=len(plan),frames_scored=sum(r['n'] for r in result['trials']),positive_hit=sum(r['hit'] for r in result['trials'] if r['object'] not in ['dust','off_track_person']),
        dust_stop=sum(r['stop'] for r in result['trials'] if r['object']=='dust'),off_track_stop=sum(r['stop'] for r in result['trials'] if r['object']=='off_track_person'),
        off_track_warning=sum(r['hit'] for r in result['trials'] if r['object']=='off_track_person'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('HOLDOUT_SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
