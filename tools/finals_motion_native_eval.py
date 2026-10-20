"""Run the candidate through unchanged real/Falcon evaluation protocols."""
import sys,json,hashlib
from pathlib import Path
from motion_curve_candidate import install

def main():
 mode=sys.argv.pop(1)
 if mode=='quality':import finals_improvement_eval as evaluator
 elif mode=='holdout':import finals_holdout_eval as evaluator
 else:raise ValueError(mode)
 original=evaluator.ObstacleDetector
 evaluator.ObstacleDetector=lambda cfg:install(original(cfg))
 out=Path(sys.argv[3]);evaluator.main()
 p=out/('summary.json' if mode=='quality' else 'holdout.json')
 data=json.loads(p.read_text());data['experimental_candidate']='motion_curve_candidate.py';data['candidate_sha256']=hashlib.sha256(Path(__file__).with_name('motion_curve_candidate.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest();p.write_text(json.dumps(data,indent=2))
if __name__=='__main__':main()
