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
    assert video['detector_commit']==load('docs/reproduction_lock.json')['runtime_commit']
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
    print('Публикация согласована: 68 + 255 условий, 1667/1667 ROS, видео и хеши трёх сцен.')
    print('Независимая реальная FP/FN этой проверкой не измеряется.')


if __name__=='__main__':
    main()
