"""Проверка целостности публикации; не заменяет запуск детектора на новых данных."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    ladder=load('web/data/finals_ladder.json')
    assert len(ladder)==68
    assert len({(r['object'],r['distance']) for r in ladder})==68
    assert all(0<=r['detected']<=1 and 0<=r['stop']<=r['detected'] for r in ladder)
    for name in ['quality_gates.json']:
        assert load('docs/finals/improvement/'+name)['passed']
    assert load('docs/finals/generalization/holdout_gates.json')['passed']
    assert len(load('docs/finals/generalization/candidate/holdout.json')['trials'])==255
    live=load('web/data/dense_live.json')
    assert len(live)==3 and sum(r['received'] for r in live)==1667
    assert all(r['passed'] and r['received']==r['expected'] and r['shutdown_clean'] and r['output_hz']>=9.8 for r in live)
    video=load('web/data/cab_video.json')
    path=ROOT/'web/assets/tunnelguard_final.mp4'
    assert abs(video['duration_s']-166.7)<.11 and len(video['scenes'])==3
    assert video['playback_rate']==1 and [s['frames'] for s in video['scenes']]==[386,736,545]
    assert video['fps']==20 and video['resolution']==[1920,1080]
    # A packaging/video commit may differ from the detector evaluation commit.
    # Require identical actual detector modules and scorer, not an invented SHA match.
    summary=load('docs/finals/spatial/quality/candidate/summary.json')
    lock=load('docs/reproduction_lock.json')
    core=ROOT/'src/tunnel_guard/tunnel_guard/core'
    hashes={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in core.glob('*.py')}
    assert hashes==summary['source_sha256']==lock['core_source_sha256']
    assert hashlib.sha256((ROOT/'src/tunnel_guard/config/obstacle_scorer.json').read_bytes()).hexdigest()==summary['model_sha256']==lock['model_sha256']
    dense=load('docs/finals/spatial/dense/offline/summary.json')
    qualification=load('docs/finals/spatial/dense/qualification/summary.json')
    assert dense['source_sha256']==qualification['source_sha256']==hashes
    assert dense['model_sha256']==qualification['model_sha256']==summary['model_sha256']
    assert qualification['complete'] and qualification['native'] and qualification['frames']==1510
    assert dense['native'] and dense['frames']==1667
    gates=load('docs/finals/spatial/quality/spatial_gates.json')
    assert gates['passed'] and len(gates['checks'])==14 and all(gates['checks'].values())
    assert load('docs/finals/spatial/moving_curve_gates.json')['passed']
    for name,h in lock['runtime_files_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h, name
    for rel in ['live_check.json','replay_2/live_check.json','replay_3/live_check.json']:
        replay=load('docs/finals/spatial/runtime/'+rel)
        assert replay['passed'] and replay['received']==replay['expected']==301
        assert replay['shutdown_clean'] and replay['output_hz']>=9.8
    assert all(abs(s['video_start_s']-start)<.06 for s,start in zip(video['scenes'],[0,38.6,112.2]))
    assert 'video-render-status' not in (ROOT/'web/index.html').read_text(encoding='utf-8')
    assert all(.09<s<.11 for s in video['source_scan_step_s'])
    assert path.stat().st_size==video['output_bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==video['output_sha256']
    assert hashlib.sha256((ROOT/'tools/finals_cab_video.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()==video['generator_source_sha256']
    for i,name in enumerate(['platform','switch','tunnels']):
        p=ROOT/f'web/data/new_synth_{name}.json';frames=json.loads(p.read_text())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==video['web_preview_clip_sha256'][name]
        assert frames[-1]['t']<=video['scenes'][i]['source_end_s']
        assert all(b['t']>a['t'] for a,b in zip(frames,frames[1:]))
    for name in ['ablation_low_obstacles.png','holdout_distance.png','full_ros_performance.png','improvement_comparison.png','distance_ladder_current.png','video_poster.png']:
        assert (ROOT/'web/assets'/name).stat().st_size>10000
    full=load('web/data/comprehensive_video.json')
    master=ROOT/'web/assets/tunnelguard_all_datasets.mp4'
    assert full['duration_s']==440 and full['playback_rate']==1 and full['scored_injection_conditions']==68
    assert full['fps']==20 and full['resolution']==[1920,1080]
    assert full['detector_source_sha256']==hashes
    assert master.stat().st_size==full['output_bytes']<95*1024**2
    assert hashlib.sha256(master.read_bytes()).hexdigest()==full['output_sha256']
    for name,h in full['generator_sources'].items():
        assert hashlib.sha256((ROOT/'tools'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h
    injections=[c for c in full['chapters'] if c['name'].startswith('Инъекции ·')]
    assert len(injections)==4 and all(len(c['source']['distances'])==17 for c in injections)
    assert sum(c['duration_s'] for c in full['chapters'])==440
    assert load('docs/finals/spatial/comprehensive_video_validation.json')['passed']
    print('Публикация согласована: 68 + 255 условий, 1667/1667 ROS, видео и хеши трёх сцен.')
    print('Независимая реальная FP/FN этой проверкой не измеряется.')


if __name__=='__main__':
    main()
