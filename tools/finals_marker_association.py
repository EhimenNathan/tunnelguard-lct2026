import json,collections,numpy as np
from pathlib import Path
base=Path(__file__).resolve().parents[1]/'docs/finals/submission_20261020/organizer_markers';rows=json.loads((base/'rows.json').read_text());markers=json.loads((base/'markers.json').read_text());groups=collections.defaultdict(list)
def rotation(q):
 x,y,z,w=np.asarray(q)/max(np.linalg.norm(q),1e-12)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def corners(m):return np.asarray(m['points'])@rotation(m['q']).T+np.asarray(m['c'])
for m in markers:
 if m['action']==0 and m['points']:groups[(m['piece'],round(m['header_stamp'],6))].append(m)
cases=[('platform_person','platform',set(range(17,23))),('switch_cable','switch',{1}),('tunnels_large_cube','tunnels',{0}),('tunnels_beam','tunnels',{1})];report=[];matches=[]
for name,piece,ids in cases:
 target_rows=[]
 for r in rows:
  if piece not in r['piece']:continue
  refs=[m for m in groups[(r['piece'],round(r['header_stamp'],6))] if m['id'] in ids]
  if not refs:continue
  pts=np.concatenate([corners(m) for m in refs]);lo,hi=pts.min(0),pts.max(0);centre=(lo+hi)/2;forward=centre@np.array(r['rotation']).T;found=[]
  for o in r['obs']:
   c=np.asarray(o['c']);gap=np.maximum(np.maximum(lo-c,c-hi),0);distance=np.linalg.norm(gap)
   if distance<=.5:found.append(o)
  target_rows.append(dict(local=r['local'],d=float(forward[0]),lat=float(forward[1]),hit=bool(found),stop=any(o['zone']==2 for o in found)))
 n=len(target_rows);hit=sum(v['hit'] for v in target_rows);stop=sum(v['stop'] for v in target_rows);first=next((v for v in target_rows if v['hit']),None)
 report.append(dict(name=name,ids=sorted(ids),marked_frames=n,matched_frames=hit,stop_frames=stop,first_match=first,range_min=min(v['d'] for v in target_rows),range_max=max(v['d'] for v in target_rows),definition='marker geometry association: detected centroid within union AABB + 0.5m Euclidean; no independent in-gauge/visibility labels; includes off-track or occluded targets; semantic mapping inferred from part dimensions and obstacles.md, not supplied id-class labels'))
 matches.append(dict(name=name,rows=target_rows))
print(json.dumps(report,indent=2));Path('marker_association.json').write_text(json.dumps(dict(cases=report,rows=matches),indent=2))
