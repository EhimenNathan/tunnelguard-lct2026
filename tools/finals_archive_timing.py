"""Изолированное время ядра на первых 120 кадрах архива; не ROS/GUI и не полный WCET.
python tools/finals_archive_timing.py ARCHIVE OUTPUT.json
"""
import sys,time,json,platform
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'src/tunnel_guard')]
from falcon_stream import frames,decode
from falcon_eval import make_detector
def main(src,out):
 det=make_detector([]);rows=[]
 for k,(t,pc) in enumerate(frames(open(src,'rb'))):
  a=decode(pc);p=np.column_stack([a[key] for key in 'xyz']).astype(np.float32)
  tic=time.perf_counter();res=det.process(p,t,intensity=a.get('intensity'),ring=a.get('ring'));elapsed=(time.perf_counter()-tic)*1000
  if k>=30:rows.append(dict(k=k,ms=elapsed,timings={key:float(v*1000) for key,v in res.timings.items()}))
  if k>=119:break
 ms=[x['ms'] for x in rows]
 report=dict(archive=Path(src).name,python=platform.python_version(),platform=platform.platform(),frames=120,warmup=30,scored=len(rows),
   timing='Только ядро, последовательный прогон без параллельных экспериментальных процессов; не ROS/GUI',
   p50_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),p99_ms=float(np.percentile(ms,99)),max_ms=max(ms),over100=sum(x>100 for x in ms),rows=rows)
 Path(out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in report.items() if k!='rows'},flush=True)
if __name__=='__main__':main(*sys.argv[1:3])
