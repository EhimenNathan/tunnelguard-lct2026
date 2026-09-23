"""Russian-language figures for the presentation (template palette). Run from the scratchpad directory with the
evaluation outputs (fp_*_summary.json, synth_*.json, holdout_results.json, report_numbers*.json)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'figures', 'deck')
os.makedirs(FIG, exist_ok=True)
for fn in ('segoeui.ttf', 'seguisb.ttf', 'segoeuib.ttf'):
    p = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', fn)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
PINK, PURPLE, DARK, BLUSH, LAV, INK, MUT = '#FF0053', '#520977', '#2D1451', '#FFD6E3', '#8A83D1', '#1C1D22', '#6B6F80'
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 13, 'axes.edgecolor': '#C9C3D6', 'axes.labelcolor': INK,
                     'xtick.color': MUT, 'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.titlesize': 15, 'axes.titlecolor': INK, 'legend.frameon': False})
EMPTY = ['roundT_doubleT', 'squareT_platform_squareT_switch', 'doubleT_platform', 'roundT_squareT_pressureGate_squareT',
         'roundT_pressureGate_roundT']
NAMES = ['круглый →\nдвухпутный', 'платформа\n+ стрелка', 'двухпутный\nплатформа', 'гермозатвор\nкруг./прямоуг.', 'гермозатвор\nкруглый']


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)


# ------------------------------------------------------------------ 1. detection range
def recall_curve(path, shape):
    syn = json.load(open(path))
    d0s = sorted({r['d0'] for r in syn})
    out = []
    for d in d0s:
        vis = [r for r in syn if r['shape'] == shape and r['d0'] == d and r['returns_first'] >= 3]
        out.append(100 * np.mean([r['detected'] for r in vis]) if vis else np.nan)
    return np.array(d0s), np.array(out)


fig, ax = plt.subplots(figsize=(9.2, 4.9))
ax.axvspan(100, 200, color=PURPLE, alpha=0.05, lw=0)
ax.axvspan(200, 215, color=PURPLE, alpha=0.11, lw=0)
ax.text(150, 104, 'критерий кейса: ≥100 м — «хорошо»', ha='center', color=PURPLE, fontsize=11)
ax.text(213, 104, '≥200 м', ha='right', color=PURPLE, fontsize=10)
for shape, lab, c, lw in [('person', 'человек 1.75 м, ρ 15 %', PINK, 3.2), ('box_50cm', 'коробка 0.5 м, ρ 20 %', PURPLE, 2.4),
                          ('dark_box_50cm', 'тёмная коробка 0.5 м, ρ 5 %', LAV, 2.4), ('box_30cm', 'коробка 0.3 м, ρ 20 %', '#C58BB0', 2.0)]:
    d, r = recall_curve('synth_hybrid.json', shape)
    ax.plot(d, r, 'o-', color=c, lw=lw, ms=7, label=lab, zorder=3)
d, r = recall_curve('synth_final.json', 'person')
ax.plot(d, r, 'o--', color=PINK, lw=1.6, ms=5, alpha=0.45, label='человек, только правила', zorder=2)
ax.scatter([160.4], [50], s=0)  # keep axis
ax.annotate('в демо-ролике: первый STOP\nна 160 м', xy=(160, 30), xytext=(168, 62), color=INK, fontsize=11,
            arrowprops=dict(arrowstyle='->', color=MUT))
ax.set_xlim(30, 215); ax.set_ylim(-4, 112)
ax.set_xlabel('расстояние до объекта в начале сближения, м'); ax.set_ylabel('подтверждённый STOP, %')
ax.legend(loc='lower left', fontsize=10.5)
ax.grid(alpha=0.25)
save(fig, 'range.png')

# ------------------------------------------------------------------ 2. false alarms per tunnel
rules = json.load(open('fp_v17_summary.json'))
final = json.load(open('fp_final_summary.json'))
fig, ax = plt.subplots(figsize=(9.2, 4.6))
x = np.arange(5)
rr = [100 * rules[n]['stop_frames'] / rules[n]['frames'] for n in EMPTY]
hh = [100 * final[n]['stop_frames'] / final[n]['frames'] for n in EMPTY]
ax.bar(x - 0.2, rr, 0.38, color=BLUSH, edgecolor=PINK, lw=1, label='только физические правила')
ax.bar(x + 0.2, hh, 0.38, color=PURPLE, label='гибрид (внедрён)')
for i, n in enumerate(EMPTY):
    ax.text(i - 0.2, rr[i] + 0.08, f"{rules[n]['stop_frames']}", ha='center', color=INK, fontsize=11)
    ax.text(i + 0.2, hh[i] + 0.08, f"{final[n]['stop_frames']}", ha='center', color=PURPLE, fontsize=11, weight='bold')
    ax.text(i, -0.62, f"{final[n]['frames']} кадров", ha='center', color=MUT, fontsize=9.5)
ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=11)
ax.set_ylabel('кадров с ложным STOP, %'); ax.set_ylim(-0.75, 5.0)
ax.legend(loc='upper left', fontsize=11)
ax.grid(axis='y', alpha=0.25)
save(fig, 'false_alarms.png')

# ------------------------------------------------------------------ 3. evolution of quality and latency
vers = [('v8', 'глобальная\nгеометрия'), ('v16', 'неопределённость\nσ(x)'), ('v17', 'физические\nтесты'),
        ('hybrid', 'гибрид\nML'), ('final', 'вето оболочки\n+ ускорение')]
fp, ms = [], []
for tag, _ in vers:
    d = json.load(open(f'fp_{tag}_summary.json'))
    fp.append(100 * sum(v['stop_frames'] for v in d.values()) / sum(v['frames'] for v in d.values()))
    ms.append(np.mean([v['ms_med'] for v in d.values()]))
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={'width_ratios': [1.45, 1]})
ax = axs[0]
ax.plot(range(5), fp, 'o-', color=PINK, lw=3, ms=9)
for i, v in enumerate(fp):
    ax.text(i + (0.12 if i == 4 else 0), v + 0.28, f'{v:.2f} %' + ('*' if i == 4 else ''), ha='center', color=INK,
            fontsize=11.5, weight='bold')
ax.text(-0.3, 4.6, '* на записях, использованных для обучения', color=MUT, fontsize=9.5)
ax.scatter([4], [0.31], s=110, marker='D', color=PURPLE, zorder=4)
ax.annotate('0.31 % — на невиденных\nтоннелях (честная оценка)', xy=(3.93, 0.36), xytext=(2.35, 2.2), fontsize=10.5,
            color=PURPLE, arrowprops=dict(arrowstyle='->', color=PURPLE))
ax.set_xticks(range(5)); ax.set_xticklabels(['v8', 'v16', 'v17 · физ. тесты', 'гибрид ML', 'финал · вето'], fontsize=11)
ax.set_ylim(-0.3, 5.0); ax.set_ylabel('ложные STOP, % кадров'); ax.set_xlim(-0.35, 4.45)
ax.set_title('Ложные остановки: 3.94 % → 0.13 %', loc='left')
ax.grid(alpha=0.25)
ax = axs[1]
ax.bar(range(5), ms, color=[BLUSH, BLUSH, BLUSH, LAV, PURPLE], edgecolor=[PINK, PINK, PINK, LAV, PURPLE])
for i, v in enumerate(ms):
    ax.text(i, v + 6, f'{v:.0f}', ha='center', color=INK, fontsize=11.5, weight='bold')
ax.axhline(100, color=PINK, lw=1.4, ls='--')
ax.set_xticks(range(5)); ax.set_xticklabels(['v8', 'v16', 'v17', 'гибрид', 'финал'], fontsize=11)
ax.set_ylabel('медианная задержка, мс'); ax.set_ylim(0, 400)
ax.set_title('Задержка: 320 → 70 мс  (пунктир — 100 мс)', loc='left')
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
save(fig, 'evolution.png')

# ------------------------------------------------------------------ 4. speed
before = json.load(open('report_numbers_rules.json'))
after = json.load(open('report_numbers.json'))
rows = [('геометрия пути', before['geo_ms'], after['geo_ms']),
        ('кадр целиком\n(2400×128 лучей)', before['ms_med'], after['ms_med']),
        ('кадр целиком\n(7200×128 лучей)', 552.0, after['hold_ms'])]
fig, ax = plt.subplots(figsize=(8.6, 4.3))
y = np.arange(len(rows))[::-1]
for yi, (lab, b, a) in zip(y, rows):
    ax.barh(yi + 0.19, b, 0.36, color=BLUSH, edgecolor=PINK)
    ax.barh(yi - 0.19, a, 0.36, color=PURPLE)
    ax.text(b + 6, yi + 0.19, f'{b:.0f} мс', va='center', color=INK, fontsize=11)
    ax.text(a + 6, yi - 0.19, f'{a:.0f} мс  ·  {1000 / a:.0f} кадр/с', va='center', color=PURPLE, fontsize=11, weight='bold')
ax.axvline(100, color=PINK, ls='--', lw=1.4)
ax.text(104, 2.62, '100 мс = 10 Гц лидара', color=PINK, fontsize=10)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
ax.set_xlim(0, 640); ax.set_xlabel('медиана на кадр, мс (ноутбук i5-8250U, 15 Вт, 1 поток)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=BLUSH, ec=PINK, label='до оптимизации'),
                   Patch(fc=PURPLE, label='после: numba, векторизация, OpenBLAS')], loc='lower right', fontsize=10.5)
ax.grid(axis='x', alpha=0.25)
save(fig, 'speed.png')

# ------------------------------------------------------------------ 5. held-out timeline
hold = json.load(open('holdout_results.json'))
fr = np.array([o['frame'] for o in hold])
lat = np.array([next((g['lat'] for g in o['gts'] if g['name'] == 'A'), np.nan) for o in hold])
lev = np.array([o['level'] for o in hold])
fig, ax = plt.subplots(figsize=(10.5, 4.0))
ax.axhspan(-1.35, 1.35, color=PURPLE, alpha=0.08, lw=0)
ax.text(2, -1.2, 'габарит поезда ±1.35 м', ha='left', color=PURPLE, fontsize=10.5)
ax.plot(fr, lat, color=INK, lw=2.4, label='человек A: смещение от оси пути (эталон)')
for lv, c, lab, yy in [(1, '#FFB020', 'ВНИМАНИЕ', 3.35), (2, PINK, 'STOP', 3.75)]:
    m = lev == lv
    ax.scatter(fr[m], np.full(m.sum(), yy), s=22, color=c, marker='s', label=f'решение: {lab}')
ax.set_xlabel('кадр (10 Гц)'); ax.set_ylabel('смещение, м'); ax.set_ylim(-2.3, 4.2); ax.set_xlim(-3, 203)
ax.legend(loc='upper left', bbox_to_anchor=(0.63, 0.55), fontsize=10, ncol=1)
ax.grid(alpha=0.2)
save(fig, 'holdout.png')

# ------------------------------------------------------------------ 6. model comparison (LORO)
models = ['логистическая регрессия', 'MLP', 'физически-информированная MLP*', 'XGBoost', 'LightGBM монотонный*', 'CatBoost*']
ap = [0.877, 0.864, 0.867, 0.884, 0.893, 0.894]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={'width_ratios': [1.2, 1]})
ax = axs[0]
ax.barh(range(6), ap, color=[LAV if not m.endswith('*') else PURPLE for m in models], height=0.62)
for i, v in enumerate(ap):
    ax.text(v + 0.0008, i, f'{v:.3f}', va='center', color=INK, fontsize=11)
ax.set_yticks(range(6)); ax.set_yticklabels(models, fontsize=10.5)
ax.set_xlim(0.84, 0.905); ax.set_xlabel('средняя точность (AP) на невиденном тоннеле')
ax.set_title('Модели · * вошли в ансамбль', loc='left')
ax = axs[1]
pts = [('только правила', 1.31, 62.8, MUT, 90), ('только ML', 1.01, 64.5, LAV, 90), ('правила И ML', 0.31, 61.2, LAV, 90),
       ('гибрид (внедрён)', 0.31, 64.5, PINK, 220)]
for name, f_, r_, c, s_ in pts:
    ax.scatter(f_, r_, s=s_, color=c, zorder=3)
    ax.annotate(name, (f_, r_), textcoords='offset points', xytext=(9, 7 if name != 'правила И ML' else -16), fontsize=11,
                color=INK, weight='bold' if c == PINK else 'normal')
ax.annotate('', xy=(0.37, 64.4), xytext=(1.26, 62.9), arrowprops=dict(arrowstyle='->', color=PINK, lw=1.8))
ax.set_xlim(0, 1.7); ax.set_ylim(60.5, 65.6)
ax.set_xlabel('ложные STOP, % кадров'); ax.set_ylabel('полнота, %')
ax.set_title('Решение: ×4 меньше ложных остановок', loc='left')
ax.grid(alpha=0.25)
plt.tight_layout()
save(fig, 'hybrid.png')
print('figures written to', os.path.abspath(FIG))

# ------------------------------------------------------------------ 7. architecture
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig = plt.figure(figsize=(15, 6.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 150); ax.set_ylim(0, 62); ax.axis('off')


def box(x, y, w, h, title, body, fc='white', ec=PURPLE, tc=INK, bc=MUT, lw=2.0, ts=14, bs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6', fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 3.2, title, ha='center', va='center', fontsize=ts, weight='bold', color=tc)
    ax.text(x + w / 2, y + (h - 5) / 2, body, ha='center', va='center', fontsize=bs, color=bc, linespacing=1.35)


def arrow(x1, y1, x2, y2, c=PINK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16, color=c, lw=2))


ax.add_patch(FancyBboxPatch((1, 1), 148, 60, boxstyle='round,pad=0,rounding_size=2', fc='#FBF7FD', ec=LAV, lw=1.2, ls='--'))
ax.text(4, 57.8, 'Docker · Ubuntu 22.04 · ROS 2 Humble · узел tunnel_guard (Python, numpy/numba)', fontsize=12, color=PURPLE)
box(4, 24, 17, 20, 'ROS 2 bag', 'или живой лидар\nPointCloud2\nтопик: auto\nPandar128')
box(25, 24, 17, 20, 'Подготовка', 'декодирование\nось «вперёд»\nполе зрения\nпрореживание')
ax.add_patch(FancyBboxPatch((46, 6), 30, 48, boxstyle='round,pad=0,rounding_size=1.6', fc='#F4ECF8', ec=PURPLE, lw=2))
ax.text(61, 50.8, 'Геометрия пути', ha='center', fontsize=14, weight='bold', color=PURPLE)
for i, (t, b) in enumerate([('Рельсы', 'гребни BEV + DP Витерби'), ('Шаблон сечения', '«нормальный тоннель»'),
                            ('Глобальный DP', 'профиль до 300 м'), ('Гаусс–Ньютон', 'клотоида, веса Тьюки'),
                            ('Неопределённость σ(x)', 'слияние во времени')]):
    yy = 40.5 - i * 8.2
    ax.add_patch(FancyBboxPatch((49, yy), 24, 6.6, boxstyle='round,pad=0,rounding_size=1', fc='white', ec=LAV, lw=1.2))
    ax.text(61, yy + 4.4, t, ha='center', fontsize=11.5, weight='bold', color=INK)
    ax.text(61, yy + 1.8, b, ha='center', fontsize=9.5, color=MUT)
box(80, 36, 20, 18, 'Габарит', 'зоны ВНУТРИ / РЯДОМ\nс учётом 2σ, в пределах\nизмеренной дальности', bs=10.5)
box(80, 10, 20, 20, '3D-кластеры', 'физические тесты:\nудержание стенкой,\nкрепление к оболочке,\nгравитация, форма')
box(104, 10, 20, 20, 'Гибридный скорер', 'ансамбль LightGBM +\nCatBoost + PI-MLP\n(> 30 м), вето\nоболочки', fc='#FFF0F5', ec=PINK)
box(104, 36, 20, 18, 'Трекер', 'подтверждение M из N\nскорость сближения\nвремя до столкновения')
box(128, 24, 18, 30, 'Решение', 'CLEAR / CAUTION / STOP\n\n/tunnel_guard/status\n/detections\n/alarm\n/nearest_distance\n/markers → RViz2',
    fc=PURPLE, ec=PURPLE, tc='white', bc='#EADCF3', bs=10.5)
arrow(21, 34, 25, 34); arrow(42, 34, 46, 34); arrow(76, 45, 80, 45); arrow(90, 36, 90, 30); arrow(100, 20, 104, 20)
arrow(114, 30, 114, 36); arrow(124, 45, 128, 45)
save(fig, 'architecture.png')
print('architecture written')
