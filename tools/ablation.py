"""Ablation study: switch off one component of the deployed detector at a time.

For every variant:
  * false STOP frames on every frame of the 5 obstacle-free recordings;
  * held-out real recording (doubleT_obstacle): frames with person A inside the gauge detected as STOP, spurious STOP;
  * ray-cast person / 0.5 m box approaching from 80, 120, 160 m (2 starts per recording) -> confirmed STOP recall.
The learned scorer was trained on candidates of the same obstacle-free recordings and on ray-cast positives, so for
variants that keep the scorer the false-alarm and synthetic columns are optimistic; the leave-one-recording-out
simulation (docs/EXPERIMENTS.md section 4) is the unbiased view of the learned stage.
Run from the scratchpad directory holding the range-image cache:  python ablation.py [variant ...]
"""
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))
sys.path.insert(0, os.getcwd())

import numpy as np

MODEL = os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json')
EMPTY = ['roundT_doubleT', 'squareT_platform_squareT_switch', 'doubleT_platform', 'roundT_squareT_pressureGate_squareT',
         'roundT_pressureGate_roundT']


def set_path(cfg, path, value):
    obj = cfg
    parts = path.split('.')
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


VARIANTS = {
    'full': {},
    'no_ml': {'scorer_model': 'none'},
    'no_shell_veto': {'scorer_shell_veto': False},
    'no_shell_test': {'shell_check': False},
    'no_containment': {'containment_check': False},
    'no_gravity': {'gravity_min_s': 1e9},
    'no_shape': {'extent_min_s': 1e9},
    'no_temporal_confirm': {'tracker.confirm_hits': 1, 'tracker.window': 1},
    'no_score_smoothing': {'scorer_smooth_k': 1},
    'no_sigma_zones': {'gauge.sigma_k': 0.0},
    'no_two_tier': {'two_tier': False},
    'no_temporal_geometry': {'geometry.temporal_weight': 0.0},
    'straight_corridor': {'geometry.smooth_lat': (1e9, 1e14), 'geometry.dp_smooth': 1e6},
    'no_ml_no_two_tier': {'scorer_model': 'none', 'two_tier': False},
    'v3': {'scorer_model': os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer_v3.json'),
           'ego_check': False, 'stop_debounce_frames': 1},
    'v3_adapt': {'scorer_model': os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer_v3.json'),
                 'adapt.enabled': True, 'ego_check': False, 'stop_debounce_frames': 1},
    'v3_ego': {'scorer_model': os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer_v3.json'),
               'ego_check': True, 'stop_debounce_frames': 1},
    'v3_ego_db': {'scorer_model': os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer_v3.json'),
                  'ego_check': True, 'stop_debounce_frames': 2, 'stop_debounce_window': 2},
    'det_v4': {},
    'router': {'scorer_model_hang': os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer_v4.json')},
    'det_v4_s4': {'scorer_model': os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer_v4.json')},                      # detector with frame-gap odometry, slack tracker gate, shell continuity, hanging exemption
    'v3_ego_db23': {'scorer_model': os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer_v3.json'),
                    'ego_check': True, 'stop_debounce_frames': 2, 'stop_debounce_window': 3},
}


def make_cfg(name, **extra):
    """'variant' or 'variant@sN' (far-field subsampling seed N of the geometry stage, for seed-variance runs)."""
    from tunnel_guard.core.detector import DetectorConfig
    base, _, seed = name.partition('@s')
    cfg = DetectorConfig(scorer_model=MODEL, **extra)
    for k, v in VARIANTS[base].items():
        set_path(cfg, k, v)
    if seed:
        cfg.geometry.sample_seed = int(seed)
    return cfg


def run_variant(name):
    from loader import BagCache
    from tunnel_guard.core.detector import ObstacleDetector
    from tunnel_guard.core.geometry import GeometryEstimator, warmup
    from tunnel_guard.core.synth import SHAPES, inject
    warmup()
    t0 = time.time()
    out = dict(variant=name)
    # ---------------------------------------------------------------- false alarms, every frame
    fp, frames, ms = {}, 0, []
    for nm in EMPTY:
        wait_for_disk()
        b = BagCache(nm)
        det = ObstacleDetector(make_cfg(name))
        stops = 0
        for f in range(b.n):
            rng, inten = b.frame(f)
            V = rng > 0.5
            fwd = b.dirs[V] * rng[V][:, None]
            xyz = np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1)
            res = det.process(xyz, b.t[f], intensity=inten[V])
            stops += int(res.level == 2)
            if f:
                ms.append(res.timings['total'] * 1e3)
        fp[nm] = stops
        frames += b.n
    out.update(fp=fp, fp_frames=int(sum(fp.values())), frames=frames, ms_med=float(np.median(ms)))
    # ---------------------------------------------------------------- held-out real recording
    b = BagCache('doubleT_obstacle')
    rows = np.load('obst_change_rows.npy')
    det = ObstacleDetector(make_cfg(name))
    inside = stop_inside = spurious = 0
    for f in range(b.n):
        rng, inten = b.frame(f)
        V = rng > 0.5
        fwd = b.dirs[V] * rng[V][:, None]
        xyz = np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1)
        res = det.process(xyz, b.t[f], intensity=inten[V])
        g = res.geometry
        r = rows[rows[:, 0] == f]
        m = (r[:, 1] > 48) & (r[:, 1] < 62) & (r[:, 3] > -2.9) & (r[:, 3] < -0.5)
        person = None
        if m.sum() >= 10 and g is not None and g.ok:
            c = np.median(r[m, 1:4], axis=0)
            c[1] = -c[1]
            lat = c[1] - float(g.centre(c[0]))
            person = (c, lat)
            if abs(lat) < 1.35:
                inside += 1
        stop_obs = [o for o in res.obstacles if o['zone'] == 2]
        matched = [o for o in stop_obs if person is not None and abs(o['distance'] - person[0][0]) < 3]
        if person is not None and abs(person[1]) < 1.35 and matched:
            stop_inside += 1
        spurious += len([o for o in stop_obs if o not in matched])
    out.update(hold_inside=inside, hold_stop=stop_inside, hold_spurious=spurious)
    # ---------------------------------------------------------------- ray-cast recall
    rs = np.random.default_rng(0)
    rec = {}
    for nm in EMPTY:
        b = BagCache(nm)
        for f0 in np.linspace(30, b.n - 12, 2).astype(int):
            geo_ref = GeometryEstimator()
            refs = {}
            for f in range(f0 - 12, f0 + 10):
                p, _, _ = b.points(f)
                refs[f] = geo_ref.estimate(p[p[:, 0] > 1.0])
            base = ObstacleDetector(make_cfg(name, forward_axis='x'))
            for f in range(f0 - 12, f0):
                p, _, _ = b.points(f)
                base.process(p, b.t[f])
            for shape_name in ('person', 'box_50cm'):
                shape = SHAPES[shape_name]
                for d0 in (80.0, 120.0, 160.0):
                    d = copy.deepcopy(base)
                    hit_any, n0 = False, None
                    for k in range(10):
                        f = f0 + k
                        dist = d0 - 1.0 * k
                        g = refs[f]
                        centre = np.array([dist, float(g.centre(dist)), float(g.rail_z(dist))])
                        rimg, inten = b.frame(f)
                        rimg2, nr = inject(rimg, b.dirs, centre, shape, rs)
                        n0 = nr if n0 is None else n0
                        V = rimg2 > 0.5
                        res = d.process(b.dirs[V] * rimg2[V][:, None], b.t[f], intensity=inten[V])
                        if any(o['zone'] == 2 and abs(o['distance'] - dist) < 3 + 0.03 * dist for o in res.obstacles):
                            hit_any = True
                    if n0 >= 3:
                        rec.setdefault(f'{shape_name}@{int(d0)}', []).append(int(hit_any))
    out['recall'] = {k: float(np.mean(v)) for k, v in rec.items()}
    out['recall_n'] = {k: len(v) for k, v in rec.items()}
    out['seconds'] = time.time() - t0
    json.dump(out, open(f'ablation_{name}.json', 'w'), indent=1)
    print(name, json.dumps({k: out[k] for k in ('fp_frames', 'hold_stop', 'hold_inside', 'hold_spurious', 'recall', 'ms_med', 'seconds')}),
          flush=True)
    return out


def wait_for_disk(min_free=400e6):
    """This laptop's pagefile grows onto a nearly full system drive; pause instead of starving the OS."""
    import shutil
    while shutil.disk_usage(os.path.abspath(os.sep)).free < min_free:
        print('low disk space - waiting', flush=True)
        time.sleep(30)


if __name__ == '__main__':
    names = sys.argv[1:] or list(VARIANTS)
    if os.environ.get('ABLATION_REVERSE'):
        names = names[::-1]
    for n in names:            # one variant per process at a time; several runners coordinate through lock files
        if os.path.exists(f'ablation_{n}.json'):
            continue
        try:
            os.close(os.open(f'ablation_{n}.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        wait_for_disk()
        run_variant(n)
