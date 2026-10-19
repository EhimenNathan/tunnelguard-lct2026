"""Regression acceptance against the current pinned runtime; not real FP/FN."""
import json,sys
from pathlib import Path

def main(root):
 root=Path(root)
 a,b=[json.loads((root/p/'summary.json').read_text()) for p in ['baseline','candidate']]
 ha=json.loads(Path('docs/finals/generalization/candidate/holdout.json').read_text())
 hb=json.loads((root/'holdout/holdout.json').read_text())
 key=lambda r:(r['object'],r['distance'],r['seed'],r['start'])
 aa={key(r):r for r in ha['trials']};bb={key(r):r for r in hb['trials']}
 positive=lambda rows:sum(r['hit'] for r in rows.values() if r['object'] not in ['dust','off_track_person'])
 checks=dict(full_falcon=a['frames']==b['frames']==4443,native=a['native'] and b['native'],
    same_model=a['model_sha256']==b['model_sha256'],
    detection_not_later=b['first_detection']['t']<=a['first_detection']['t'],
    no_band_loses_hits=all(y['hit']>=x['hit'] for x,y in zip(a['bands'],b['bands'])),
    no_band_loses_stop=all(y['stop']>=x['stop'] for x,y in zip(a['bands'],b['bands'])),
    no_extra_unmatched_stop=b['unmatched_stop_frames']<=a['unmatched_stop_frames'],
    no_extra_beside_stop=b['beside_stop']<=a['beside_stop'],
    same_255_protocol=ha['protocol_sha256']==hb['protocol_sha256'] and set(aa)==set(bb) and len(bb)==255,
    no_positive_trial_loses_hits=all(bb[k]['hit']>=v['hit'] for k,v in aa.items() if v['object'] not in ['dust','off_track_person']),
    no_positive_trial_loses_stop=all(bb[k]['stop']>=v['stop'] for k,v in aa.items() if v['object'] not in ['dust','off_track_person']),
    no_extra_dust_stop=sum(r['near_stop'] for r in bb.values() if r['object']=='dust')<=sum(r['near_stop'] for r in aa.values() if r['object']=='dust'),
    no_extra_offtrack_warning=sum(r['hit'] for r in bb.values() if r['object']=='off_track_person')<=sum(r['hit'] for r in aa.values() if r['object']=='off_track_person'),
    no_extra_offtrack_stop=sum(r['stop'] for r in bb.values() if r['object']=='off_track_person')<=sum(r['stop'] for r in aa.values() if r['object']=='off_track_person'))
 out=dict(passed=all(checks.values()),checks=checks,positive_before=positive(aa),positive_after=positive(bb),
    matched_stop_before=sum(r['stop'] for r in a['bands']),matched_stop_after=sum(r['stop'] for r in b['bands']),
    unmatched_stop_before=a['unmatched_stop_frames'],unmatched_stop_after=b['unmatched_stop_frames'],
    limitation='Previously examined regression set; automatic real trajectory; no independent class FP/FN')
 (root/'spatial_gates.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
 if not out['passed']:raise SystemExit(1)
if __name__=='__main__':main(sys.argv[1])
