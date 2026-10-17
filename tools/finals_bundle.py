"""Воспроизводимый полный прогон Falcon + метрики устойчивости + реальные кадры для сайта/видео.

python tools/finals_bundle.py CACHE TRUTH OUT_DIR
Разметка автоматически извлечена из этой записи: это proxy-оценка, не независимый эталон.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'src/tunnel_guard')]
from falcon_cache import frames_cached
from falcon_eval import make_detector, fill_gaps, TRACK_Y


def clean(v):
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list): return [clean(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def save(path, obj):
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')


def main(cache, truth_path, output):
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parent,text=True).strip()
    source_hashes = {str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE.parent/'src/tunnel_guard/tunnel_guard/core').glob('*.py')}
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    truth = {r['k']: r for r in fill_gaps(json.loads(Path(truth_path).read_text()))}
    det = make_detector([])
    rows, geom, clips = [], [], [[], [], []]
    ranges = [(0, 300), (2200, 2400), (3800, 4000)]
    X = np.array([25., 50., 100., 150., 200., 250., 300.])
    rng = np.random.default_rng(20261017)
    for k, t, xyz, intensity in frames_cached(cache):
        started = time.perf_counter()
        r = det.process(xyz, t, intensity=intensity)
        ms = (time.perf_counter() - started) * 1000
        tr = truth.get(k, {})
        g = r.geometry
        known = tr.get('n', 0) > 0
        tx, ty = tr.get('x'), tr.get('y')
        tl = float(ty - g.centre(tx)) if known and g.ok else None
        # Match distance AND lateral location.  Do not score unknown-reference frames as confirmed negatives.
        match = lambda o: known and abs(o['distance'] - tx) < 6 and tl is not None and abs(o['lateral'] - tl) < 1.0
        person = [o for o in r.obstacles if match(o)]
        row = dict(k=k,t=t,level=int(r.level),ms=round(ms,3),geometry=bool(g.ok),
                   env=float(r.envelope_range),measured=float(g.valid_range),clear=float(r.clear_distance),
                   truth_x=tx if known else None,truth_y=ty,truth_interp=bool(tr.get('interp',False)),
                   hit=bool(person),stop_hit=any(o['zone']==2 for o in person),
                   unmatched_stop=any(o['zone']==2 and not match(o) for o in r.obstacles),
                   obs=[dict(d=float(o['distance']),l=float(o['lateral']),zone=int(o['zone'])) for o in r.obstacles])
        rows.append(row)
        geom.append(dict(k=k,ok=bool(g.ok),y=g.centre(X).tolist() if g.ok else None,
                         z=g.rail_z(X).tolist() if g.ok else None))
        for ci,(a,b) in enumerate(ranges):
            if a<=k<=b and k%5==0:
                p=r.forward_points
                eligible=np.flatnonzero((p[:,0]>2)&(p[:,0]<410)&(abs(p[:,1])<12)&(p[:,2]>-4)&(p[:,2]<9))
                far=eligible[p[eligible,0]>100]
                near=eligible[p[eligible,0]<=100]
                # Keep far returns and object returns: uniform sampling would erase the faint target.
                far=rng.choice(far,min(len(far),1400),replace=False)
                near=rng.choice(near,min(len(near),1000),replace=False)
                special=np.concatenate([np.asarray(o['idx'],int) for o in r.obstacles]) if r.obstacles else np.zeros(0,int)
                sel=np.unique(np.r_[far,near,special])
                if len(sel)>5000: sel=rng.choice(sel,5000,replace=False)
                line=[]
                if g.ok:
                    xx=np.arange(5.,min(r.envelope_range,400.)+1,5.)
                    line=np.round(np.column_stack([xx,g.centre(xx),g.rail_z(xx)]),3).tolist()
                clips[ci].append(dict(row,p=np.round(p[sel],2).ravel().tolist(),line=line,
                                       boxes=[dict(c=np.round(o['centroid_fwd'],3).tolist(),
                                                   size=np.round(o['size'],3).tolist(),zone=int(o['zone']),
                                                   d=round(float(o['distance']),1),n=int(o['n'])) for o in r.obstacles]))
        if k%500==0: print(k,round(t,1),'s',flush=True)
    save(out/'rows.json',rows)
    save(out/'geometry.json',geom)
    for i,c in enumerate(clips): save(out/f'falcon_clip_{i}.json',c)
    known=[r for r in rows if r['truth_x'] is not None]
    on=[r for r in known if abs(r['truth_y']-TRACK_Y)<1.3]
    beside=[r for r in known if abs(r['truth_y']-TRACK_Y)>=1.6]
    first=next((r for r in on if r['hit']),None)
    first_stop=next((r for r in on if r['stop_hit']),None)
    bands=[]
    for lo,hi in [(0,50),(50,100),(100,150),(150,200),(200,250),(250,320)]:
        rr=[r for r in on if lo<=r['truth_x']<hi]
        bands.append(dict(lo=lo,hi=hi,n=len(rr),hit=sum(r['hit'] for r in rr),stop=sum(r['stop_hit'] for r in rr),
                          recall=sum(r['hit'] for r in rr)/len(rr) if rr else None,
                          stop_rate=sum(r['stop_hit'] for r in rr)/len(rr) if rr else None))
    steady=rows[30:]; ms=[r['ms'] for r in steady]; ev=np.array([r['env'] for r in steady])
    Y=np.array([gg['y'] if gg['ok'] else [np.nan]*len(X) for gg in geom[30:]])
    jitter={str(int(x)):dict(p50_cm=float(np.nanmedian(abs(np.diff(Y[:,i])))*100),
                               p95_cm=float(np.nanpercentile(abs(np.diff(Y[:,i])),95)*100)) for i,x in enumerate(X)}
    summary=dict(schema=1,reference='automatic trajectory from the same recording; proxy, not independent ground truth',
                 matching='distance <6m AND lateral <1m; on-track |y-0.35|<1.3m; beside >=1.6m',
                 frames=len(rows),known_reference_frames=len(known),interpolated_frames=sum(r['truth_interp'] for r in rows),
                 unknown_reference_frames=len(rows)-len(known),on_track=len(on),beside=len(beside),
                 first_detection={k:first[k] for k in ['k','t','truth_x']} if first else None,
                 first_stop={k:first_stop[k] for k in ['k','t','truth_x']} if first_stop else None,
                 bands=bands,beside_stop=sum(r['stop_hit'] for r in beside),
                 unmatched_stop_frames=sum(r['unmatched_stop'] for r in rows),
                 envelope_median_m=float(np.median(ev)),envelope_p90_m=float(np.percentile(ev,90)),
                 envelope_max_m=float(ev.max()),measured_median_m=float(np.median([r['measured'] for r in steady])),
                 geometry_dropouts_after_30=sum(not r['geometry'] for r in steady),
                 envelope_change_p95_m=float(np.percentile(abs(np.diff(ev)),95)),jitter=jitter,
                 latency_ms=dict(p50=float(np.median(ms)),p95=float(np.percentile(ms,95)),p99=float(np.percentile(ms,99)),
                                 max=float(max(ms)),over_100=sum(v>100 for v in ms),n=len(ms)),
                 provenance=dict(commit=source_commit,source_sha256=source_hashes,
                                 python=platform.python_version(),platform=platform.platform(),
                                 cache_bytes=Path(cache).stat().st_size,
                                 index_sha256=hashlib.sha256(Path(cache+'.idx.npy').read_bytes()).hexdigest(),
                                 truth_sha256=hashlib.sha256(Path(truth_path).read_bytes()).hexdigest(),
                                 models={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                         (HERE.parent/'src/tunnel_guard/config').glob('obstacle_scorer*.json')}))
    save(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main(*sys.argv[1:4])
