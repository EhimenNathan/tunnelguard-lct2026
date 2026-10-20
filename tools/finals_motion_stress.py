"""Paired analytic moving-train tests with known curve and target geometry.

Approximate angular nearest-return rendering, not a calibrated lidar digital twin.
python tools/finals_moving_curves.py BASELINE_PACKAGE_ROOT OUTPUT_JSON
"""
import sys,json,importlib.util,importlib,time,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;R=HERE.parent
sys.path[:0]=[str(R/'src/tunnel_guard'),str(R/'src/tunnel_guard/test')]
from tunnel_sim import make_tunnel,centreline
from tunnel_guard.core.detector import ObstacleDetector,DetectorConfig
from tunnel_guard.core.geometry import warmup,HAVE_NATIVE

def scan(world,labels,pose,radius,rng):
    theta=0. if radius is None else np.arctan(pose/radius)
    dx=world[:,0]-pose;dy=world[:,1]-float(centreline(np.array([pose]),radius)[0]);c,s=np.cos(theta),np.sin(theta)
    q=np.column_stack([c*dx+s*dy,-s*dx+c*dy,world[:,2]]).astype(np.float32)
    keep=(q[:,0]>2)&(q[:,0]<220)&(abs(q[:,1])<q[:,0]*np.tan(np.radians(70)))
    q,labels=q[keep],labels[keep];rr=np.linalg.norm(q,axis=1)
    az=np.degrees(np.arctan2(q[:,1],q[:,0]));el=np.degrees(np.arctan2(q[:,2],np.hypot(q[:,0],q[:,1])))
    key=np.floor((az+rng.uniform(0,.12))/.12).astype(np.int64)*10000+np.floor((el+rng.uniform(0,.12))/.12).astype(np.int64)
    order=np.lexsort((rr,key));selected=order[np.r_[True,np.diff(key[order])!=0]]
    return q[selected],labels[selected]

def main(baseline,out):
    from motion_curve_candidate import install
    baseline=Path(baseline);spec=importlib.util.spec_from_file_location('baseline_tunnel_guard',baseline/'tunnel_guard/__init__.py',submodule_search_locations=[str(baseline/'tunnel_guard')]);pkg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pkg;spec.loader.exec_module(pkg)
    old=importlib.import_module('baseline_tunnel_guard.core.detector');importlib.import_module('baseline_tunnel_guard.core.geometry').warmup();warmup()
    result=dict(native=HAVE_NATIVE,baseline_source_sha256={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in (baseline/'tunnel_guard/core').glob('*.py')},candidate_source_sha256={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in (R/'src/tunnel_guard/tunnel_guard/core').glob('*.py')},candidate_module='tools/motion_curve_candidate.py',candidate_sha256=hashlib.sha256((HERE/'motion_curve_candidate.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),method='Stress settings: three paired seed/speed settings, unseen radii +/-400 and +/-650m. Analytic static tunnel in world coordinates; train 2/5/10m/s, 80 scans at 10Hz, first30 warm; .12degree nearest-return bins; not an empirical sensor model or independent real train validation',cases=[])
    for seed,speed in [(811,2.),(977,5.),(1213,10.)]:
        for radius in [650.,-650.,400.,-400.]:
            for positive in [False,True]:
                world=make_tunnel(radius=radius,lidar_height=1.8,s_max=300.,seed=seed,obstacles=[(120.,0.,.6,.6,1.75)] if positive else [])
                obj=(world[:,0]>=120)&(world[:,0]<=120.6)&(abs(world[:,1]-centreline(world[:,0],radius))<.35)&(world[:,2]>-1.65) if positive else np.zeros(len(world),bool)
                dets={'baseline':old.ObstacleDetector(old.DetectorConfig(forward_axis='x',mount_calibration=False,scorer_model=str(baseline/'config/obstacle_scorer.json'))),'candidate':ObstacleDetector(DetectorConfig(forward_axis='x',mount_calibration=False,scorer_model=str(R/'src/tunnel_guard/config/obstacle_scorer.json')))}
                dets['candidate']=install(dets['candidate']);rng=np.random.default_rng(seed+100);rows=[]
                for k in range(80):
                    pose=k*.1*speed;p,flags=scan(world,obj,pose,radius,rng);theta=0 if radius is None else np.arctan(pose/radius);c,s=np.cos(theta),np.sin(theta)
                    gx=np.arange(pose,300.,.1);dy=centreline(gx,radius)-float(centreline(np.array([pose]),radius)[0]);xx=c*(gx-pose)+s*dy;yy=-s*(gx-pose)+c*dy
                    ox=120.3-pose;oy=float(centreline(np.array([120.3]),radius)[0]-centreline(np.array([pose]),radius)[0]);tx=c*ox+s*oy;ty=-s*ox+c*oy
                    for name in (['baseline','candidate'] if k%2==0 else ['candidate','baseline']):
                        res=dets[name].process(p,k*.1)
                        if k<30:continue
                        matched=[o for o in res.obstacles if abs(o['centroid_fwd'][0]-tx)<4 and abs(o['centroid_fwd'][1]-ty)<.8]
                        rows.append(dict(k=k,name=name,range=float(tx),target_returns=int(flags.sum()),geometry=bool(res.geometry.ok),hit=bool(matched),stop=any(o['zone']==2 for o in matched),any_stop=res.level==2,any_obstacle=bool(res.obstacles),error50=float(abs(res.geometry.centre(50.)-np.interp(50.,xx,yy))) if res.geometry.ok else None))
                summaries={}
                for name in dets:
                    rr=[r for r in rows if r['name']==name];visible=[r for r in rr if r['target_returns']>=3]
                    summaries[name]=dict(n=len(rr),geometry=sum(r['geometry'] for r in rr),hit=sum(r['hit'] for r in rr),stop=sum(r['stop'] for r in rr),any_stop=sum(r['any_stop'] for r in rr),any_obstacle=sum(r['any_obstacle'] for r in rr),visible_frames=len(visible),visible_hit=sum(r['hit'] for r in visible),error50_p95_cm=float(np.percentile([r['error50'] for r in rr if r['error50'] is not None],95))*100 if any(r['error50'] is not None for r in rr) else None)
                case=dict(seed=seed,speed_m_s=speed,radius=radius,positive=positive,summaries=summaries,rows=rows,alternate_tested=dets['candidate'].geo.alternate_tested,alternate_selected=dets['candidate'].geo.alternate_selected);result['cases'].append(case);Path(out).write_text(json.dumps(result,indent=2));print('MOVING_CURVE',json.dumps({k:v for k,v in case.items() if k!='rows'}),flush=True)
if __name__=='__main__':main(*sys.argv[1:3])
