"""Паритет и изолированное время C++/Numba ядра; не end-to-end benchmark."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from tunnel_guard.core.geometry import HAVE_NATIVE, _gn_numba
from tunnel_guard.core._geometry_native import gn

assert HAVE_NATIVE
rng=np.random.default_rng(2026)
n,N=500,12000
args=[rng.uniform(2,450,N),rng.uniform(-6,6,N),rng.uniform(-3,6,N),
      rng.integers(0,n-1,N,dtype=np.int64),rng.random(N),rng.random(N),rng.normal(0,.2,n),
      np.full(n,-1.8),rng.normal(0,.05,n),.02,rng.uniform(0,.6,(320,200)),.5,
      -8.,-2.5,.05,1.1,.3,3.2,.8,1.2,.3,3.7,n]
ref=_gn_numba(*args);cpp=gn(*args)
for a,b in zip(ref,cpp):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)
report=dict(points=N,nodes=n,parity=True,maximum_absolute_error=float(max(np.max(abs(a-b)) for a,b in zip(ref,cpp))))
for name,fn in [('cpp',gn),('numba',_gn_numba)]:
    times=[]
    for _ in range(101):
        t=time.perf_counter();fn(*args);times.append((time.perf_counter()-t)*1000)
    report[name]=dict(p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)))
Path(sys.argv[1]).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
