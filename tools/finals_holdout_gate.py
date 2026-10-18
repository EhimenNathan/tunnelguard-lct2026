"""Compare frozen holdout predictions without tuning thresholds on the holdout."""
import json,sys
from pathlib import Path

def main(base,candidate,out):
 a,b=[json.loads((Path(p)/'holdout.json').read_text()) for p in [base,candidate]]
 key=lambda r:(r['object'],r['distance'],r['seed'],r['start'])
 aa={key(r):r for r in a['trials']};bb={key(r):r for r in b['trials']}
 def total(rows,kind,field):return sum(r[field] for r in rows.values() if r['object']==kind)
 positive=lambda rows:sum(r['hit'] for r in rows.values() if r['object'] not in ['dust','off_track_person'])
 checks=dict(same_protocol=a['protocol_sha256']==b['protocol_sha256'],all_255_conditions=len(aa)==len(bb)==255 and set(aa)==set(bb),
     no_extra_dust_stop=total(bb,'dust','near_stop')<=total(aa,'dust','near_stop'),
     no_extra_offtrack_stop=total(bb,'off_track_person','stop')<=total(aa,'off_track_person','stop'),
     no_extra_offtrack_warning=total(bb,'off_track_person','hit')<=total(aa,'off_track_person','hit'),
     positive_matches_improve=positive(bb)>positive(aa),
     no_positive_case_loses_matches=all(bb[k]['hit']>=v['hit'] for k,v in aa.items() if v['object'] not in ['dust','off_track_person']))
 result=dict(checks=checks,passed=all(checks.values()),positive_before=positive(aa),positive_after=positive(bb),
     dust_stop_before=total(aa,'dust','near_stop'),dust_stop_after=total(bb,'dust','near_stop'),
     offtrack_stop_before=total(aa,'off_track_person','stop'),offtrack_stop_after=total(bb,'off_track_person','stop'),
     offtrack_warning_before=total(aa,'off_track_person','hit'),offtrack_warning_after=total(bb,'off_track_person','hit'))
 Path(out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main(*sys.argv[1:4])
