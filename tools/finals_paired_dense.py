"""Same decoded dense frames, alternating evaluation order, two isolated packages.

python tools/finals_paired_dense.py ARCHIVE BASELINE_PACKAGE_ROOT OUTPUT.json
Baseline root contains tunnel_guard/ and config/, e.g. an export of a frozen commit.
"""
import importlib,importlib.util,json,sys,time,platform,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'src/tunnel_guard')]
from falcon_stream import frames,decode
from tunnel_guard.core.detector import ObstacleDetector,DetectorConfig
from tunnel_guard.core.geometry import warmup

def main(src,baseline,out):
    baseline=Path(baseline)
    spec=importlib.util.spec_from_file_location('baseline_tunnel_guard',baseline/'tunnel_guard/__init__.py',submodule_search_locations=[str(baseline/'tunnel_guard')])
    package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
    old=importlib.import_module('baseline_tunnel_guard.core.detector')
    old_geometry=importlib.import_module('baseline_tunnel_guard.core.geometry')
    old_geometry.warmup();warmup()
    dets={'baseline':old.ObstacleDetector(old.DetectorConfig(scorer_model=str(baseline/'config/obstacle_scorer.json'))),
          'candidate':ObstacleDetector(DetectorConfig(scorer_model=str(HERE.parent/'src/tunnel_guard/config/obstacle_scorer.json')))}
    rows=[]
    for k,(t,pc) in enumerate(frames(open(src,'rb'))):
        a=decode(pc);p=np.column_stack([a[key] for key in 'xyz']).astype(np.float32)
        order=['baseline','candidate'] if k%2==0 else ['candidate','baseline']
        for name in order:
            tic=time.perf_counter();r=dets[name].process(p,t,intensity=a.get('intensity'),ring=a.get('ring'));ms=1000*(time.perf_counter()-tic)
            if k>=30:rows.append(dict(k=k,name=name,ms=ms,level=r.level,obstacles=len(r.obstacles),timings={s:1000*r.timings[s] for s in ['prep','geometry','envelope','tracking','total']}))
        if k%30==0:print('paired dense',k,flush=True)
        if k>=119:break
    summaries={}
    for name in dets:
        rr=[r for r in rows if r['name']==name];ms=[r['ms'] for r in rr]
        summaries[name]=dict(n=len(rr),p50_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),p99_ms=float(np.percentile(ms,99)),over100=sum(x>100 for x in ms),stage_p50_ms={stage:float(np.median([r['timings'][stage] for r in rr])) for stage in ['prep','geometry','envelope','tracking','total']})
    roots={'baseline':baseline/'tunnel_guard/core','candidate':HERE.parent/'src/tunnel_guard/tunnel_guard/core'}
    result=dict(archive=Path(src).name,python=platform.python_version(),platform=platform.platform(),method='same first 120 raw frames; 30 warmup; alternating order per frame; no ROS/GUI; not WCET',summaries=summaries,source_sha256={name:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')} for name,root in roots.items()},rows=rows)
    Path(out).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summaries),flush=True)

if __name__=='__main__':main(*sys.argv[1:4])
