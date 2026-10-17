"""Quality gates for the paired protocol; not a substitute for independent labels.
python tools/finals_compare_gate.py BASELINE_DIR CANDIDATE_DIR OUTPUT_JSON
"""
import json,sys
from pathlib import Path

def main(base,candidate,out):
 a,b=[json.loads((Path(p)/'summary.json').read_text()) for p in [base,candidate]]
 la,lb=[json.loads((Path(p)/'ladder.json').read_text()) for p in [base,candidate]]
 ab={(r['object'],r['distance']):r for r in la};bb={(r['object'],r['distance']):r for r in lb}
 checks=dict(full_recording=a['frames']==b['frames']==4443,full_ladder=len(la)==len(lb)==68 and set(ab)==set(bb),
  no_new_geometry_dropouts=b['geometry_dropouts_after_30']<=a['geometry_dropouts_after_30'],
  beside_stop_not_worse=b['beside_stop']<=a['beside_stop'],unmatched_stop_not_worse=b['unmatched_stop_frames']<=a['unmatched_stop_frames'],
  far_matches_not_worse=b['bands'][-1]['hit']>=a['bands'][-1]['hit'],
  near_matches_not_worse=sum(r['hit'] for r in b['bands'][:-1])>=sum(r['hit'] for r in a['bands'][:-1]),
  dust_no_obstacle=all(r['detected']==r['stop']==0 for r in lb if r['object']=='dust'),
  small_box_detection_improves=sum(r['detected'] for r in lb if r['object']=='box_0.3')>sum(r['detected'] for r in la if r['object']=='box_0.3'),
  person_300_stop_not_worse=bb['person',300]['stop']>=ab['person',300]['stop'])
 result=dict(reference='automatic trajectory / synthetic injections, not independent ground truth',checks=checks,passed=all(checks.values()))
 Path(out).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main(*sys.argv[1:4])
