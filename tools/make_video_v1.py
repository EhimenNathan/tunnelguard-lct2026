"""Render the demo video docs/demo_tunnelguard.mp4 and key frames for the presentation.

Story (as requested by the case): tunnel -> point cloud -> algorithm -> obstacle detected -> distance to it.
  1. train moving through a curved tunnel with no obstacles (CLEAR, verified clear distance);
  2. held-out real recording with people in the tunnel (never used for development);
  3. detection range: a person ray-cast into the real beams of a moving recording, approaching from 200 m.
The detector is the deployed configuration (physics rules + packaged hybrid scorer). Run from the scratchpad
directory that holds the range-image cache (see tools/README.md).
"""
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard', 'tools'))
sys.path.insert(0, os.getcwd())

import numpy as np
import imageio.v2 as imageio

from loader import BagCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.geometry import GeometryEstimator, warmup
from tunnel_guard.core.synth import SHAPES, inject
from render_demo import render_frame, title_card, metrics_card, INK, MUTED, PURPLE

MODEL = os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json')
OUT = os.path.join(SOL, 'docs', 'demo_tunnelguard.mp4')
FIG = os.path.join(SOL, 'docs', 'figures')
FPS = 10
warmup()

writer = imageio.get_writer(OUT, fps=FPS, codec='libx264', macro_block_size=2,
                            ffmpeg_params=['-crf', '22', '-preset', 'slow', '-pix_fmt', 'yuv420p'])
keyframes = {}
log = {}


def put(img, n=1, fade_from=None):
    for k in range(n):
        if fade_from is not None and k < 6:
            a = (k + 1) / 7.0
            writer.append_data((fade_from * (1 - a) + img * a).astype(np.uint8))
        else:
            writer.append_data(img)


def sensor_xyz(b, f, rimg=None):
    rng, inten = b.frame(f)
    if rimg is not None:
        rng = rimg
    V = rng > 0.5
    fwd = b.dirs[V] * rng[V][:, None]
    return np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1), inten[V], int(V.sum())


t0 = time.time()
black = np.zeros((900, 1600, 3), np.uint8)
card = title_card([('TunnelGuard', 64, 'white', 'bold'),
                   ('Обнаружение посторонних объектов в габарите', 28, INK, 'normal'),
                   ('беспилотного поезда метро по данным 3D-лидара', 28, INK, 'normal'),
                   ('тоннель → облако точек → алгоритм → препятствие → расстояние', 20, '#b69cff', 'normal')],
                  footer='ROS 2 Humble · Docker · Hesai Pandar128 · ЛЦТ 2026 · кейс «Московский транспорт»')
put(card, 35, fade_from=black)
last = card

# ---------------------------------------------------------------- 1. moving train, curved tunnel, no obstacles
chap = title_card([('1', 80, PURPLE, 'bold'), ('Поезд едет по тоннелю', 40, 'white', 'bold'),
                   ('Кривая и переход «круглый → двухпутный» тоннель. Препятствий нет.', 22, INK, 'normal'),
                   ('Путь и габарит восстанавливаются по лидару в каждом кадре.', 22, MUTED, 'normal')])
put(chap, 22, fade_from=last)
b = BagCache('roundT_doubleT')
det = ObstacleDetector(DetectorConfig(scorer_model=MODEL))
first = True
for f in range(80, 210):
    xyz, inten, n = sensor_xyz(b, f)
    res = det.process(xyz, b.t[f], intensity=inten)
    if f < 110:
        continue
    img = render_frame(res, det.cfg.gauge, info=dict(
        chapter='1 · Поезд в тоннеле: кривая, смена типа тоннеля — препятствий нет',
        subtitle='roundT_doubleT · реальные кадры', clock=f'кадр {f} · t = {b.t[f]:.1f} с', points=n))
    put(img, 1, fade_from=chap if first else None) if first else put(img)
    if first:
        for _ in range(5):
            put(img)
    first = False
    if f == 170:
        keyframes['clear'] = img
    last = img
log['seg1_levels'] = None

# ---------------------------------------------------------------- 2. held-out real recording with people
chap = title_card([('2', 80, PURPLE, 'bold'), ('Реальные люди в тоннеле', 40, 'white', 'bold'),
                   ('Запись doubleT_obstacle не использовалась при разработке и настройке.', 22, INK, 'normal'),
                   ('Человек A пересекает путь на ~56 м; человек B идёт рядом с поездом вне габарита.', 22, MUTED, 'normal')])
put(chap, 26, fade_from=last)
b = BagCache('doubleT_obstacle')
det = ObstacleDetector(DetectorConfig(scorer_model=MODEL))
first_stop = None
levels = []
for f in range(b.n):
    xyz, inten, n = sensor_xyz(b, f)
    res = det.process(xyz, b.t[f], intensity=inten)
    levels.append(int(res.level))
    img = render_frame(res, det.cfg.gauge, info=dict(
        chapter='2 · Отложенная реальная запись: человек в габарите поезда',
        subtitle='doubleT_obstacle · поезд стоит · 921 600 лучей/кадр', clock=f'кадр {f} · t = {b.t[f]:.1f} с', points=n))
    if f == 0:
        put(img, 1, fade_from=chap)
        for _ in range(5):
            put(img)
    else:
        put(img)
    if res.level == 2 and first_stop is None:
        first_stop = (f, float(res.nearest_distance))
        keyframes['first_stop'] = img
    if f == 12:
        keyframes['approach'] = img
    if f == 45:
        keyframes['stop'] = img
    if f == 175:
        keyframes['walk_beside'] = img
    last = img
log['holdout_first_stop'] = first_stop
log['holdout_levels'] = np.bincount(levels, minlength=3).tolist()

# ---------------------------------------------------------------- 3. range: ray-cast person approaching from 200 m
D0, SPEED, F0, SEQ = 200.0, 12.0, 699, 130
chap = title_card([('3', 80, PURPLE, 'bold'), ('Дальность обнаружения', 40, 'white', 'bold'),
                   ('Человек (1.75 м, отражающая способность 15 %) вставлен лучевым моделированием', 22, INK, 'normal'),
                   (f'в реальные лучи Pandar128 и приближается с {D0:.0f} м со скоростью {SPEED:.0f} м/с.', 22, INK, 'normal'),
                   ('Метка «синтетика» — это не реальный объект.', 22, MUTED, 'normal')])
put(chap, 30, fade_from=last)
b = BagCache('squareT_platform_squareT_switch')
geo_ref = GeometryEstimator()
det = ObstacleDetector(DetectorConfig(forward_axis='x', scorer_model=MODEL))
rng = np.random.default_rng(7)
shape = SHAPES['person']
first_det = None
for f in range(F0 - 15, F0 + SEQ):
    p, I, _ = b.points(f)
    g = geo_ref.estimate(p[p[:, 0] > 1.0])
    if f < F0:
        det.process(p, b.t[f])
        continue
    k = f - F0
    dist = D0 - SPEED * k / FPS
    centre = np.array([dist, float(g.centre(dist)), float(g.rail_z(dist))])
    rimg, inten = b.frame(f)
    rimg2, nret = inject(rimg, b.dirs, centre, shape, rng)
    V = rimg2 > 0.5
    pts = b.dirs[V] * rimg2[V][:, None]
    res = det.process(pts, b.t[f], intensity=inten[V])
    hit = [o for o in res.obstacles if o['zone'] == 2 and abs(o['distance'] - dist) < 3 + 0.03 * dist]
    if hit and res.level == 2 and first_det is None:
        first_det = (k, float(dist), float(hit[0]['distance']))
    img = render_frame(res, det.cfg.gauge, info=dict(
        chapter=f'3 · Синтетика: человек на пути, истинная дальность {dist:.0f} м · отражений {nret}',
        subtitle='squareT_platform_squareT_switch · реальные кадры, движение',
        clock=f'кадр {f} · t = {b.t[f]:.1f} с', points=int(V.sum())))
    if k == 0:
        put(img, 1, fade_from=chap)
        for _ in range(5):
            put(img)
    else:
        put(img)
    if first_det is not None and 'far' not in keyframes:
        keyframes['far'] = img
    last = img
log['synthetic_first_stop'] = first_det
print('segments done', time.time() - t0, log, flush=True)

# ---------------------------------------------------------------- results card
R = json.load(open('report_numbers.json'))
far = f'{first_det[1]:.0f} м' if first_det else '—'
card = metrics_card('Результаты', [
    ('0.31 %', 'ложных остановок на тоннелях,\nне участвовавших в обучении'),
    (f'{R["hold_detected"]}/{R["hold_inside"]}', 'кадров с человеком в габарите\nобнаружено (отложенная запись)'),
    (f'{R["hold_dist"]:.1f} м', 'реальный человек обнаружен\nи подтверждён STOP'),
    (far, 'синтетический человек:\nпервый STOP в этом ролике'),
    ('100 % · 77 %', 'полнота (человек, видимые случаи)\nна 80 м · на 120 м'),
    (f'{R["ms_med"]:.0f} мс', 'медианная задержка на кадр\n(1 поток ноутбука i5-8250U)'),
], footer='Подробности и воспроизведение: README.md, docs/EXPERIMENTS.md, tools/')
put(card, 60, fade_from=last)
writer.close()

for name, img in keyframes.items():
    imageio.imwrite(os.path.join(FIG, f'demo_{name}.png'), img)
imageio.imwrite(os.path.join(FIG, 'demo_results_card.png'), card)
json.dump(log, open(os.path.join(SOL, 'docs', 'figures', 'demo_log.json'), 'w'), indent=1)
print('video written', OUT, os.path.getsize(OUT) / 1e6, 'MB', time.time() - t0, 's', flush=True)
