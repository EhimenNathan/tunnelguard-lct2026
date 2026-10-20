exec((__import__('pathlib').Path(__file__).with_name('finals_marker_association.py')).read_text(encoding='utf-8').split('for name,piece,ids in cases:')[0])
result=[]
for name,piece,ids in cases:
 rr=[]
 for r in rows:
  if piece not in r['piece']:continue
  refs=[m for m in groups[(r['piece'],round(r['header_stamp'],6))] if m['id'] in ids]
  if not refs:continue
  rm=np.asarray(r['rotation']);pts=np.concatenate([corners(m) for m in refs])@rm.T;lo,hi=pts.min(0),pts.max(0);c=(lo+hi)/2;found=[]
  for o in r['obs']:
   q=np.asarray(o['c'])@rm.T;gap=np.maximum(np.maximum(lo-q,q-hi),0)
   if np.all(gap<=np.array([4.,.8,.5])):found.append(o)
  rr.append(dict(local=r['local'],d=float(c[0]),hit=bool(found),stop=any(o['zone']==2 for o in found)))
 result.append(dict(name=name,marked_frames=len(rr),matched=sum(v['hit'] for v in rr),stop=sum(v['stop'] for v in rr),first=next((v for v in rr if v['hit']),None),definition='centroid proximity to organizer box: <=4m longitudinal,0.8m lateral,0.5m vertical outside its AABB; class mapping inferred, no independent in-gauge/visibility labels',rows=rr))
print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in result],indent=2));Path('marker_tolerance_audit.json').write_text(json.dumps(result,indent=2))
