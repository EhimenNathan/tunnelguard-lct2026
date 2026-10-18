"""Prepare all native 10Hz visualization scans for the normal-speed renderer."""
import json
import shutil
import sys
from pathlib import Path


def main(source,target):
    source=Path(source);target=Path(target)
    root=Path(__file__).resolve().parents[1]
    clips=json.loads((source/'clip.json').read_text())
    summary=json.loads((source/'summary.json').read_text())
    assert summary['frames']==1667 and summary['native']
    web=target/'web';(web/'data').mkdir(parents=True,exist_ok=True);(web/'assets').mkdir(exist_ok=True)
    docs=target/'docs/finals/new_synth';docs.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'summary.json',docs/'summary.json')
    shutil.copy2(root/'web/data/gauge_profile.json',web/'data/gauge_profile.json')
    for name,n in [('platform',386),('switch',736),('tunnels',545)]:
        frames=[f for f in clips if 'synthetic_'+name+'_cloud_only' in f['piece']]
        assert len(frames)==n, 'Full original 10Hz scans are required'
        (web/f'data/new_synth_{name}.json').write_text(json.dumps(frames,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print('Prepared all 1667 scans, no interpolated point clouds or detections.')


if __name__=='__main__':main(*sys.argv[1:3])
