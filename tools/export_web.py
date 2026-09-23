"""Export real detector output for the interactive prototype page (web/): per scenario a gzip-compressed binary of point
clouds (int16 cm, forward frame X fwd / Y left / Z up) with per-point labels, and a JSON with the per-frame decisions,
obstacles, envelope rings and track centreline.  Same detector runs as tools/make_video.py (final model; the comparison
scenario also runs v1).  Run from the scratchpad directory holding the range-image caches:  python export_web.py
"""
import base64
import copy
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))
sys.path.insert(0, os.getcwd())

import numpy as np

from loader import BagCache
from run_drive import DriveCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.gauge import envelope_polygon
from tunnel_guard.core.geometry import GeometryEstimator, warmup
from tunnel_guard.core.synth import SHAPES, inject

CFG = os.path.join(SOL, 'src', 'tunnel_guard', 'config')
MODEL, MODEL_V1 = os.path.join(CFG, 'obstacle_scorer.json'), os.path.join(CFG, 'obstacle_scorer_v1.json')
OUT = os.path.join(SOL, 'web', 'data')
NPTS = 16000
FAR_MAX = 5000
os.makedirs(OUT, exist_ok=True)
warmup()


def r2(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 2)


def result_meta(res, gauge, speed=None):
    g = res.geometry
    clear = float(res.clear_distance)
    d = dict(level=int(res.level), nearest=r2(res.nearest_distance), clear=r2(clear), ttc=r2(res.nearest_ttc),
             ms=round(1e3 * res.timings['total'], 1), speed=r2(speed),
             radius=r2(abs(1 / res.curvature)) if abs(res.curvature) > 2e-4 else None,
             obs=[dict(zone=int(o['zone']), d=r2(o['distance']), l=r2(o['lateral']), h=r2(o['height']),
                       conf=r2(o['confidence']), ttc=r2(o['ttc']), n=int(o['n']), ego=bool(o.get('ego_veto', False)))
                  for o in res.obstacles], env=[], line=[])
    if g is not None and g.ok:
        for dd in (10, 20, 35, 55, 80, 110, 150, 200):
            if dd > min(clear, 200):
                break
            sl, sv = g.sigma_at(np.array([float(dd)]))
            poly = envelope_polygon(gauge, dd, float(sl[0]), float(sv[0]))
            ring = np.stack([np.full(len(poly), dd), g.centre(dd) + poly[:, 0],
                             g.rail_z(dd) + poly[:, 1] + g.roll * poly[:, 0]], 1)
            d['env'].append(np.round(ring, 2).ravel().tolist())
        xs = np.linspace(3, max(min(clear, 230), 4), 40)
        d['line'] = np.round(np.stack([xs, g.centre(xs), g.rail_z(xs)], 1), 2).ravel().tolist()
    return d


class Writer:
    def __init__(self, name, title, n_res=1):
        self.name, self.title, self.n_res = name, title, n_res
        self.blob, self.frames = bytearray(), []
        self.rng = np.random.default_rng(0)

    def add(self, results, gauge, t, speeds=None, syn=None, note=None):
        P = results[0].forward_points
        m = (P[:, 0] > 2) & (P[:, 0] < 230) & (np.abs(P[:, 1]) < 12) & (P[:, 2] > -4) & (P[:, 2] < 9)
        labels = []
        for res in results:
            lab = np.zeros(len(P), np.uint8)
            for o in res.obstacles:
                lab[o['idx']] = 3 if o.get('ego_veto') else (2 if o['zone'] == 2 else 1)
            labels.append(lab)
        special = np.zeros(len(P), bool)
        for lab in labels:
            special |= lab > 0
        far = np.flatnonzero(m & (P[:, 0] >= 60) & ~special)
        near = np.flatnonzero(m & (P[:, 0] < 60) & ~special)
        if len(far) > FAR_MAX:                  # far returns are sparse already; the near field shows the tunnel
            far = self.rng.choice(far, FAR_MAX, replace=False)
        room = max(NPTS - int(special.sum()) - len(far), 0)
        if len(near) > room:
            near = self.rng.choice(near, room, replace=False)
        keep = np.r_[np.flatnonzero(special), far, near]
        q = np.clip(np.round(P[keep] * 100), -32768, 32767).astype('<i2')
        off = len(self.blob)
        self.blob += q.tobytes()
        for lab in labels:
            self.blob += lab[keep].tobytes()
        self.blob += b'\x00' * ((-len(self.blob)) % 4)
        self.frames.append(dict(t=round(float(t), 2), off=off, n=int(len(keep)), syn=syn, note=note,
                                r=[result_meta(res, gauge, (speeds or [None] * len(results))[k])
                                   for k, res in enumerate(results)]))

    def save(self, info):
        fn = f'{self.name}.b64.txt'             # base64 of the gzip stream: the page host serves text, not raw binary
        with open(os.path.join(OUT, fn), 'wb') as f:
            f.write(base64.b64encode(gzip.compress(bytes(self.blob), compresslevel=9)))
        json.dump(dict(name=self.name, title=self.title, file=fn, n_res=self.n_res, frames=self.frames, **info),
                  open(os.path.join(OUT, f'{self.name}.json'), 'w', encoding='utf-8'), ensure_ascii=False,
                  separators=(',', ':'))
        print(self.name, len(self.frames), 'frames', round(len(self.blob) / 1e6, 1), 'MB raw ->',
              round(os.path.getsize(os.path.join(OUT, fn)) / 1e6, 1), 'MB gz', flush=True)


def sensor_xyz(b, f):
    rng, inten = b.frame(f)
    V = rng > 0.5
    fwd = b.dirs[V] * rng[V][:, None]
    return np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1), (inten[V] if inten is not None else None)


t0 = time.time()
# ---------------------------------------------------------------- A. dataset 1, held-out real people
w = Writer('ds1_real', 'Датасет 1 · реальные люди в тоннеле')
b = BagCache('doubleT_obstacle')
det = ObstacleDetector(DetectorConfig(scorer_model=MODEL))
for f in range(120):
    xyz, inten = sensor_xyz(b, f)
    res = det.process(xyz, b.t[f], intensity=inten)
    w.add([res], det.cfg.gauge, b.t[f])
w.save(dict(dataset=1, synthetic=False, models=['final'],
            about='Отложенная реальная запись: поезд стоит, человек A пересекает путь на ~56 м, человек B идёт рядом '
                  'с поездом вне габарита. Запись не использовалась ни при разработке, ни при обучении.'))

# ---------------------------------------------------------------- B. dataset 1, synthetic person from 200 m
D0, SPEED, F0, SEQ = 200.0, 12.0, 699, 130
w = Writer('ds1_range', 'Датасет 1 · дальность: человек с 200 м')
b = BagCache('squareT_platform_squareT_switch')
geo_ref = GeometryEstimator()
det = ObstacleDetector(DetectorConfig(forward_axis='x', scorer_model=MODEL))
rng = np.random.default_rng(7)
for f in range(F0 - 15, F0 + SEQ):
    p, _, _ = b.points(f)
    g = geo_ref.estimate(p[p[:, 0] > 1.0])
    if f < F0:
        det.process(p, b.t[f])
        continue
    dist = D0 - SPEED * (f - F0) / 10
    rimg, inten = b.frame(f)
    rimg2, nret = inject(rimg, b.dirs, np.array([dist, float(g.centre(dist)), float(g.rail_z(dist))]), SHAPES['person'], rng)
    V = rimg2 > 0.5
    res = det.process(b.dirs[V] * rimg2[V][:, None], b.t[f], intensity=inten[V])
    w.add([res], det.cfg.gauge, b.t[f], syn=dict(x=r2(dist), y=r2(g.centre(dist)), z=r2(g.rail_z(dist) + 0.9), hits=int(nret)))
w.save(dict(dataset=1, synthetic=True, models=['final'],
            about='Синтетический человек 1.75 м (отражение 15 %) вставлен лучевым моделированием в реальные лучи '
                  'Pandar128 и приближается с 200 м со скоростью 12 м/с. Всё остальное — реальная запись. '
                  'Зачем: в данных нет реальных препятствий дальше ~60 м — иначе дальность не измерить.'))
print('dataset 1 done', round(time.time() - t0), flush=True)

# ---------------------------------------------------------------- C/D. dataset 2, continuous replay of the new line
T_START, T_END = 869.0, 1071.0
W_SYN, W_CMP, D_SYN = (1051.5, 1061.5), (1062.0, 1070.0), 180.0
drive = DriveCache('cache/ds2')
idx = [i for i in range(len(drive)) if T_START <= drive.t[i] <= T_END]
det_new = ObstacleDetector(DetectorConfig(scorer_model=MODEL))
det_old = ObstacleDetector(DetectorConfig(scorer_model=MODEL_V1, ego_check=False))
det_syn, x_person = None, None
w_syn = Writer('ds2_person', 'Датасет 2 · человек на пути новой линии')
w_cmp = Writer('ds2_compare', 'Датасет 2 · ложные тревоги: v1 против финальной', n_res=2)
rng = np.random.default_rng(11)


def speed_kmh(d):
    s = list(d.calib.steps)
    return 36.0 * float(np.median(s)) if len(s) >= 3 else None


for i in idx:
    t = float(drive.t[i])
    pi, kf = drive.index[i]
    piece = drive.pieces[pi]
    xyz, inten = drive.cloud(i)
    if W_SYN[0] <= t <= W_SYN[1] and det_syn is None:
        det_syn = copy.deepcopy(det_new)
        x_person = det_new.calib.x + D_SYN
    res = det_new.process(xyz, t, intensity=inten)
    res_old = det_old.process(xyz, t, intensity=inten)
    if det_syn is not None and t <= W_SYN[1]:
        dist = x_person - det_new.calib.x
        g = res.geometry
        rimg, _ = piece.frame(kf)
        rimg2, nret = inject(rimg, piece.dirs, np.array([dist, -float(g.centre(dist)), float(g.rail_z(dist))]),
                             SHAPES['person'], rng)
        V = rimg2 > 0.5
        fwd = piece.dirs[V] * rimg2[V][:, None]
        rs = det_syn.process(np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1), t)
        w_syn.add([rs], det_syn.cfg.gauge, t, speeds=[speed_kmh(det_syn)],
                  syn=dict(x=r2(dist), y=r2(g.centre(dist)), z=r2(g.rail_z(dist) + 0.9), hits=int(nret)))
    if W_CMP[0] <= t <= W_CMP[1]:
        w_cmp.add([res, res_old], det_new.cfg.gauge, t, speeds=[speed_kmh(det_new)] * 2)
w_syn.save(dict(dataset=2, synthetic=True, models=['final'],
                about='В датасете 2 нет препятствий, поэтому обнаружение на ходу проверяется синтетическим человеком: '
                      'он стоит в тоннеле в 180 м впереди, поезд приближается с реальной скоростью поездки (~60 км/ч).'))
w_cmp.save(dict(dataset=2, synthetic=False, models=['final', 'v1'],
                about='Одни и те же секунды новой линии без синтетики. Поезд позже проехал все места тревог — препятствий '
                      'не было. Переключайте модель: прежняя v1 тормозит, финальная — нет.'))
print('done', round(time.time() - t0), flush=True)
