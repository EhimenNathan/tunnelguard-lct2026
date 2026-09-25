"""Render the demo video docs/demo_tunnelguard.mp4 and key frames for the presentation.

Story (as requested by the case): tunnel -> point cloud -> algorithm -> obstacle detected -> distance to it, on both
data sets and always with the final model (physics layer + ½·LightGBM + ½·CatBoost + ego-motion test); segment 5
compares it with the previous model v1 (⅓·LightGBM + ⅓·CatBoost + ⅓·PI-MLP):
  dataset 1 (original tunnels)
    1. held-out real recording with people in the tunnel (never used for development);
    2. detection range: a person ray-cast into the real beams of a moving recording, approaching from 200 m;
  dataset 2 (new line, 20-min drive, not seen during development of the detector)
    3. the train driving at line speed: speed from lidar odometry, verified clear distance;
    4. a person standing on the new line (ray-cast, static in the tunnel), the train approaching at its real speed;
    5. the same seconds with the previous model (false STOP) and the deployed one (no STOP).
The drive is replayed continuously from t = 870 s, exactly as in the sealed-test evaluation (tools/run_drive.py), so the
footage reproduces the reported numbers.  Run from the scratchpad directory holding the range-image caches.
"""
import copy
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard', 'tools'))
sys.path.insert(0, os.getcwd())

import numpy as np
import imageio.v2 as imageio
from PIL import Image

from loader import BagCache
from run_drive import DriveCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.geometry import GeometryEstimator, warmup
from tunnel_guard.core.synth import SHAPES, inject
from ds3_stream import frames as ds3_frames, nearest_object
from render_demo import render_frame, title_card, metrics_card, chart_card, INK, MUTED, PURPLE

CFG_DIR = os.path.join(SOL, 'src', 'tunnel_guard', 'config')
MODEL = os.path.join(CFG_DIR, 'obstacle_scorer.json')          # deployed (v3)
MODEL_V1 = os.path.join(CFG_DIR, 'obstacle_scorer_v1.json')    # previous model, for the comparison only
OUT = os.path.join(SOL, 'docs', 'demo_tunnelguard.mp4')
FIG = os.path.join(SOL, 'docs', 'figures')
FPS = 10
B1 = ('ДАТАСЕТ 1 · исходные тоннели', '#6d28d9')
B2 = ('ДАТАСЕТ 2 · новая линия, не использовалась при разработке детектора', '#0e7490')
SYN = ('СИНТЕТИКА · лучевое моделирование в реальных лучах', '#b45309')
M_FINAL = 'роутер двух ансамблей ½·LightGBM + ½·CatBoost + ЭГО-тест (финальная)'
M_V1 = 'v1: ⅓·LightGBM + ⅓·CatBoost + ⅓·PI-MLP (прежняя)'
warmup()

writer = imageio.get_writer(OUT, fps=FPS, codec='libx264', macro_block_size=2,
                            ffmpeg_params=['-crf', '20', '-preset', 'slow', '-pix_fmt', 'yuv420p'])
keyframes, log = {}, {}
last = None


def put(img, n=1, fade_from=None):
    global last
    for k in range(n):
        if fade_from is not None and k < 6:
            a = (k + 1) / 7.0
            writer.append_data((fade_from * (1 - a) + img * a).astype(np.uint8))
        else:
            writer.append_data(img)
    last = img


def put_seq(frames, hold_first=5):
    """Write a rendered segment: fade in from the previous image, hold the first frame briefly."""
    for j, img in enumerate(frames):
        put(img, 1, fade_from=last if j == 0 else None)
        if j == 0:
            put(img, hold_first)


def chapter(num, title, lines, n=26):
    card = title_card([(num, 80, PURPLE, 'bold'), (title, 40, 'white', 'bold')] +
                      [(t, 22, INK if i == 0 else MUTED, 'normal') for i, t in enumerate(lines)])
    put(card, n, fade_from=last)


def sensor_xyz(b, f, rimg=None):
    rng, inten = b.frame(f)
    if rimg is not None:
        rng = rimg
    V = rng > 0.5
    fwd = b.dirs[V] * rng[V][:, None]
    return np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1), (inten[V] if inten is not None else None), int(V.sum())


def packed(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, 'PNG', compress_level=3)
    return buf.getvalue()


def unpacked(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))


t0 = time.time()
black = np.zeros((900, 1600, 3), np.uint8)
last = black
put(title_card([('TunnelGuard', 64, 'white', 'bold'),
                ('Обнаружение посторонних объектов в габарите', 28, INK, 'normal'),
                ('беспилотного поезда метро по данным 3D-лидара', 28, INK, 'normal'),
                ('физика пути + ML · проверено на новой линии, которую детектор не видел', 20, '#b69cff', 'normal')],
               footer='ROS 2 Humble · Docker · Hesai Pandar128 · ЛЦТ 2026 · кейс «Московский транспорт»'), 38, fade_from=black)
put(title_card([('Что показано', 44, 'white', 'bold'),
                ('Все кадры — живой вывод детектора на реальных записях лидара.', 24, INK, 'normal'),
                ('Модель во всех сегментах — финальная: роутер двух ансамблей ½·LightGBM + ½·CatBoost + ЭГО-тест.', 22, INK, 'normal'),
                ('Только в сегменте 5 для сравнения — прежняя v1: ⅓·LightGBM + ⅓·CatBoost + ⅓·PI-MLP.', 22, MUTED, 'normal'),
                ('Синтетические объекты помечены оранжевым «СИНТЕТИКА»: в данных заказчика ничего не добавляется.', 22,
                 MUTED, 'normal')]), 50, fade_from=last)

# ================================================================ dataset 1 · 1. held-out real recording with people
chapter('1', 'Датасет 1 · реальные люди в тоннеле',
        ['Запись doubleT_obstacle не использовалась при разработке, обучении и настройке.',
         'Человек A пересекает путь на ~56 м; человек B идёт рядом с поездом вне габарита.'])
b = BagCache('doubleT_obstacle')
det = ObstacleDetector(DetectorConfig(scorer_model=MODEL))
first_stop, levels, frames = None, [], []
for f in range(b.n):
    xyz, inten, n = sensor_xyz(b, f)
    res = det.process(xyz, b.t[f], intensity=inten)
    levels.append(int(res.level))
    img = render_frame(res, det.cfg.gauge, info=dict(
        chapter='1 · Отложенная реальная запись: человек в габарите поезда', badge=B1, model=M_FINAL,
        subtitle='doubleT_obstacle · поезд стоит · 921 600 лучей/кадр', clock=f'кадр {f} · t = {b.t[f]:.1f} с', points=n))
    frames.append(img)
    if res.level == 2 and first_stop is None:
        first_stop = (f, float(res.nearest_distance))
        keyframes['first_stop'] = img
    for k, name in ((12, 'approach'), (45, 'stop'), (175, 'walk_beside')):
        if f == k:
            keyframes[name] = img
put_seq(frames)
log['holdout_first_stop'] = first_stop
log['holdout_levels'] = np.bincount(levels, minlength=3).tolist()
print('segment 1', first_stop, time.time() - t0, flush=True)

# ================================================================ dataset 1 · 2. range: ray-cast person from 200 m
D0, SPEED, F0, SEQ = 200.0, 12.0, 699, 130
chapter('2', 'Датасет 1 · дальность обнаружения',
        ['Зачем синтетика: в записях нет реальных препятствий дальше ~60 м — без неё дальность не измерить.',
         f'Человек 1.75 м вставлен лучевым моделированием в реальные лучи Pandar128 и приближается с {D0:.0f} м.',
         'Всё остальное — реальная запись; зелёная область вдали — дальняя часть реального тоннеля.',
         'Синтетика используется только для оценки: ROS-узел и данные заказчика её не содержат.'], n=45)
b = BagCache('squareT_platform_squareT_switch')
geo_ref = GeometryEstimator()
det = ObstacleDetector(DetectorConfig(forward_axis='x', scorer_model=MODEL))
rng = np.random.default_rng(7)
first_det, frames = None, []
for f in range(F0 - 15, F0 + SEQ):
    p, _, _ = b.points(f)
    g = geo_ref.estimate(p[p[:, 0] > 1.0])
    if f < F0:
        det.process(p, b.t[f])
        continue
    k = f - F0
    dist = D0 - SPEED * k / FPS
    centre = np.array([dist, float(g.centre(dist)), float(g.rail_z(dist))])
    rimg, inten = b.frame(f)
    rimg2, nret = inject(rimg, b.dirs, centre, SHAPES['person'], rng)
    V = rimg2 > 0.5
    res = det.process(b.dirs[V] * rimg2[V][:, None], b.t[f], intensity=inten[V])
    hit = [o for o in res.obstacles if o['zone'] == 2 and abs(o['distance'] - dist) < 3 + 0.03 * dist]
    if hit and res.level == 2 and first_det is None:
        first_det = (k, float(dist), float(hit[0]['distance']))
    img = render_frame(res, det.cfg.gauge, info=dict(
        chapter=f'2 · Синтетический человек на пути: истинная дальность {dist:.0f} м · отражений {nret}', badge=SYN,
        model=M_FINAL, syn_xyz=(dist, float(g.centre(dist)), float(g.rail_z(dist)) + 0.9),
        syn_label=f'синтетический человек · {dist:.0f} м',
        subtitle='squareT_platform_squareT_switch · реальные кадры', clock=f'кадр {f} · t = {b.t[f]:.1f} с',
        points=int(V.sum())))
    frames.append(img)
    if first_det is not None and 'far' not in keyframes:
        keyframes['far'] = img
put_seq(frames)
log['synthetic_first_stop'] = first_det
print('segment 2', first_det, time.time() - t0, flush=True)

# ================================================================ dataset 2 · continuous replay of the new line
T_START, T_END = 869.0, 1071.0
W_DRIVE = (1036.0, 1050.0)       # 3. driving at line speed
W_SYN = (1051.5, 1061.5)         # 4. person standing on the new line
W_CMP = (1062.0, 1070.0)         # 5. previous model vs deployed
D_SYN = 180.0
drive = DriveCache('cache/ds2')
idx = [i for i in range(len(drive)) if T_START <= drive.t[i] <= T_END]
det_new = ObstacleDetector(DetectorConfig(scorer_model=MODEL))
det_old = ObstacleDetector(DetectorConfig(scorer_model=MODEL_V1, ego_check=False))
det_syn, x_person, syn_first = None, None, None
seg3, seg4, cmp_old, cmp_new = [], [], [], []
n_old_stop = n_new_stop = 0
rng = np.random.default_rng(11)


def speed_kmh(d):
    steps = list(d.calib.steps)
    return 3.6 * 10.0 * float(np.median(steps)) if len(steps) >= 3 else None   # odometry re-locking


for i in idx:
    t = float(drive.t[i])
    pi, kf = drive.index[i]
    piece = drive.pieces[pi]
    xyz, inten = drive.cloud(i)
    n = len(xyz)
    if W_SYN[0] <= t <= W_SYN[1] and det_syn is None:
        det_syn = copy.deepcopy(det_new)                   # same history as the deployed detector
        x_person = det_new.calib.x + D_SYN                 # the person stands still in the tunnel
    res = det_new.process(xyz, t, intensity=inten)
    res_old = det_old.process(xyz, t, intensity=inten)
    clock = f'новая линия · t = {t:.1f} с'
    if W_DRIVE[0] <= t <= W_DRIVE[1]:
        img = render_frame(res, det_new.cfg.gauge, info=dict(
            chapter='3 · Поезд на линейной скорости: путь и габарит восстанавливаются в каждом кадре', badge=B2,
            model=M_FINAL,
            subtitle='20-минутная поездка · 13 км · разметки нет', clock=clock, speed=speed_kmh(det_new)))
        seg3.append(packed(img))
        if abs(t - 1042.0) < 0.05:
            keyframes['ds2_drive'] = img
    if det_syn is not None and t <= W_SYN[1]:
        dist = x_person - det_new.calib.x
        g = res.geometry
        rimg, _ = piece.frame(kf)
        rimg2, nret = inject(rimg, piece.dirs, np.array([dist, -float(g.centre(dist)), float(g.rail_z(dist))]),
                             SHAPES['person'], rng)
        V = rimg2 > 0.5
        fwd = piece.dirs[V] * rimg2[V][:, None]
        rs = det_syn.process(np.stack([-fwd[:, 1], -fwd[:, 0], fwd[:, 2]], 1), t)
        hit = [o for o in rs.obstacles if o['zone'] == 2 and abs(o['distance'] - dist) < 3 + 0.03 * dist]
        if hit and syn_first is None:
            syn_first = (round(t - W_SYN[0], 1), float(dist), float(hit[0]['distance']), speed_kmh(det_syn))
        img = render_frame(rs, det_syn.cfg.gauge, info=dict(
            chapter=f'4 · Синтетический человек на пути новой линии: истинная дальность {dist:.0f} м · отражений {nret}',
            badge=SYN, model=M_FINAL, syn_xyz=(dist, float(g.centre(dist)), float(g.rail_z(dist)) + 0.9),
            syn_label=f'синтетический человек · {dist:.0f} м',
            subtitle='поезд приближается с реальной скоростью поездки', clock=clock,
            speed=speed_kmh(det_syn)))
        seg4.append(packed(img))
        if syn_first is not None and 'ds2_person' not in keyframes:
            keyframes['ds2_person'] = img
    if W_CMP[0] <= t <= W_CMP[1]:
        n_old_stop += int(res_old.level == 2)
        n_new_stop += int(res.level == 2)
        img_o = render_frame(res_old, det_old.cfg.gauge, info=dict(
            chapter='5 · Те же секунды — прежняя модель v1', badge=('ПРЕЖНЯЯ МОДЕЛЬ v1', '#9f1239'), model=M_V1,
            note=f'ложных STOP-кадров: {n_old_stop}' + ('  ·  тормозить не из-за чего: поезд позже проехал это место'
                                                     if res_old.level == 2 else ''),
            note_color='#9f1239dd', subtitle='новая линия · препятствий нет', clock=clock, speed=speed_kmh(det_new)))
        img_n = render_frame(res, det_new.cfg.gauge, info=dict(
            chapter='5 · Те же секунды — финальная модель', badge=('ФИНАЛЬНАЯ МОДЕЛЬ', '#15803d'), model=M_FINAL,
            note=f'ложных STOP-кадров: {n_new_stop}', note_color='#15803ddd',
            subtitle='новая линия · препятствий нет', clock=clock, speed=speed_kmh(det_new)))
        cmp_old.append(packed(img_o))
        cmp_new.append(packed(img_n))
        if res_old.level == 2 and 'ds2_old_false' not in keyframes:
            keyframes['ds2_old_false'] = img_o
            keyframes['ds2_new_same'] = img_n
log['ds2_person_first_stop'] = syn_first
log['ds2_compare_stop_frames'] = dict(previous=n_old_stop, deployed=n_new_stop)
print('dataset 2', syn_first, n_old_stop, n_new_stop, time.time() - t0, flush=True)

chapter('3', 'Датасет 2 · новая линия',
        ['20 минут реальной поездки, 13 км, ~10 станций. Детектор этих данных при разработке не видел.',
         'Скорость — по лидарной одометрии («штрихкод» стен тоннеля). Препятствий на линии нет.'])
put_seq([unpacked(d) for d in seg3])
chapter('4', 'Датасет 2 · человек на пути новой линии',
        ['В датасете 2 нет препятствий, поэтому обнаружение на ходу проверяется синтетическим человеком.',
         f'Он стоит в тоннеле в {D_SYN:.0f} м впереди; поезд приближается с реальной скоростью поездки.',
         'Вставка лучевым моделированием в реальные лучи этой поездки — это не реальный объект.'], n=38)
put_seq([unpacked(d) for d in seg4])
chapter('5', 'Датасет 2 · ложные тревоги: было → стало',
        ['Одни и те же секунды новой линии, без синтетики. Поезд позже проехал все места тревог — препятствий не было.',
         'Прежняя v1 (⅓·LightGBM + ⅓·CatBoost + ⅓·PI-MLP) тормозит;',
         'финальная (роутер двух ансамблей + ЭГО-тест) — нет.'], n=38)
put_seq([unpacked(d) for d in cmp_old], hold_first=3)
put_seq([unpacked(d) for d in cmp_new], hold_first=3)

# ================================================================ dataset 3 · the organisers' synthetic objects
ORG = ('СИНТЕТИКА ОРГАНИЗАТОРОВ · их бэг, реальный тоннель', '#b45309')
chapter('6', 'Датасет 3 · объекты организаторов',
        ['Бэг организаторов: реальная запись тоннеля с десятью их синтетическими объектами через ~100 м.',
         'Их генератор ставит объекты на плоскость в системе лидара, а путь идёт под уклон — объекты «парят».',
         'Показаны куб 2×2 м в центре габарита и объекты 3–7 подряд. Финальная модель, живой вывод.'], n=40)
det3 = ObstacleDetector(DetectorConfig(scorer_model=MODEL))
seg6 = []
for t3, xyz3 in ds3_frames():
    res = det3.process(xyz3, t3)
    if t3 <= 23.5 or 40.0 <= t3 <= 58.5:
        o = nearest_object(det3.calib.x + 60.0)
        dist = o[0] - det3.calib.x
        label = f'{o[1]} · ≈{dist:.0f} м' if dist > 0 else o[1]
        seg6.append(packed(render_frame(res, det3.cfg.gauge, info=dict(
            chapter='6 · ' + label, badge=ORG, model=M_FINAL,
            subtitle='бэг организаторов cloud_with_fake_obj · 16 байт на точку', clock=f't = {t3:.1f} с'))))
    if t3 > 58.5:
        break
put_seq([unpacked(d) for d in seg6])
print('segment 6 (dataset 3)', len(seg6), 'frames', time.time() - t0, flush=True)

# ================================================================ results
first_stop_m = f'{first_stop[1]:.1f} м' if first_stop else '—'
far = f'{first_det[1]:.0f} м' if first_det else '—'
syn2 = f'{syn_first[1]:.0f} м' if syn_first else '—'
card1 = metrics_card('Датасет 1 · исходные тоннели', [
    ('100 %', 'точность STOP (58 из 58)\nотложенная запись с людьми'),
    ('96.7 %', 'полнота: 58 из 60 кадров\nс человеком в габарите'),
    (first_stop_m, 'реальный человек\nподтверждён STOP'),
    ('100 · 78 %', 'полнота: человек\nна 80 м · на 120 м'),
    (far, 'синтетический человек:\nпервый STOP в этом ролике'),
    ('0', 'ложных STOP-кадров на 5 пустых\nзаписях (2 287 кадров)'),
], footer='Финальная модель: роутер двух ансамблей + ЭГО-тест · accuracy 99.0 % · F1 0.98')
put(card1, 60, fade_from=last)
card2 = metrics_card('Датасет 2 · новая линия, запечатанный тест', [
    ('1.8 / км', 'ложных STOP-событий\n(было 10.3 / км)'),
    ('×6', 'меньше ложных остановок\nна последних 2.8 км'),
    ('0.32 %', 'кадров с ложным STOP\n(было 2.06 %)'),
    ('91 / 91', 'тревога старой модели: поезд\nпроехал место — доказано ложные'),
    (syn2, 'человек на новой линии:\nпервый STOP в этом ролике'),
    ('56–70 мс', 'на кадр, одно ядро CPU\n(лидар 10 Гц = 100 мс)'),
], footer='Финальная модель · запечатанный тест: последние 5 минут поездки не использовались ни для обучения, ни для настройки')
put(card2, 60, fade_from=last)
card3 = metrics_card('Датасет 3 · объекты организаторов', [
    ('98 м', 'куб 2×2 м в центре габарита:\nпервый STOP (было 48 м)'),
    ('7 из 8', 'объектов «в габарите» — STOP\n(не найден стержень 5 см)'),
    ('1', 'ложное STOP-событие на пустом\nтоннеле за 1.8 км (кривые R 243 м)'),
    ('60 м', 'брус 2×0.2 м на рельсах:\nпервый STOP'),
    ('2', 'объекта «за габаритом» с STOP:\nнаш габарит — вагон 2.7 м'),
    ('1 параметр', 'габарит организаторов (≈ ±1.15 м)\nзадаётся в конфигурации'),
], footer='Прочитано 29 % архива организаторов (438 кадров, повреждён) — в нём все 10 объектов')
put(card3, 60, fade_from=last)
chart = chart_card('Запечатанный тест: ложные остановки', os.path.join(FIG, 'deck', 'ds2_sealed.png'),
                   caption='Среднее по 3 зёрнам подвыборки геометрии; усы — минимум…максимум')
put(chart, 50, fade_from=last)
closing = title_card([('TunnelGuard', 64, 'white', 'bold'),
                      ('«Путь свободен» или «Впереди препятствие. Нужно тормозить».', 28, INK, 'normal'),
                      (f'Реальный человек — STOP на {first_stop_m}; ложных остановок на новой линии в 6 раз меньше.', 22,
                       MUTED, 'normal'),
                      ('docker build → docker run → ros2 bag play', 20, '#b69cff', 'normal')],
                     footer='Подробности и воспроизведение: README.md, docs/EXPERIMENTS.md, tools/')
put(closing, 45, fade_from=last)
put(black, 1, fade_from=closing)
writer.close()

for name, img in keyframes.items():
    imageio.imwrite(os.path.join(FIG, f'demo_{name}.png'), img)
imageio.imwrite(os.path.join(FIG, 'demo_results_card.png'), card1)
imageio.imwrite(os.path.join(FIG, 'demo_results_card_ds2.png'), card2)
json.dump(log, open(os.path.join(FIG, 'demo_log.json'), 'w'), indent=1)
print('video written', OUT, os.path.getsize(OUT) / 1e6, 'MB', time.time() - t0, 's', log, flush=True)
