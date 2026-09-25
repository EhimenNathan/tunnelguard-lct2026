"""Explanatory deck figures (Russian, template palette): development stages, decision logic, model stack.
Run from the scratchpad directory (reads fp_<stage>_summary.json and sealed_summary.json)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'figures', 'deck')
for fn in ('segoeui.ttf', 'seguisb.ttf', 'segoeuib.ttf'):
    p = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', fn)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
PINK, PURPLE, INK, MUT, LAV, BLUSH = '#FF0053', '#520977', '#1C1D22', '#6B6F80', '#8A83D1', '#FFD6E3'
GREEN, AMBER, RED = '#16a34a', '#f59e0b', '#e11d48'
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#C9C3D6', 'xtick.color': MUT, 'ytick.color': MUT})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)


def box(ax, x, y, w, h, title, body='', fc='white', ec=PURPLE, tc=INK, bc=MUT, ts=12.5, bs=10.5, lw=1.8, left=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.18', fc=fc, ec=ec, lw=lw))
    if body:
        ax.text(x + w / 2, y + h - 0.28, title, ha='center', va='top', fontsize=ts, weight='bold', color=tc)
        ax.text(x + 0.45 if left else x + w / 2, y + h - 0.45 - ts * 0.052, body, ha='left' if left else 'center',
                va='top', fontsize=bs, color=bc, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=ts, weight='bold', color=tc)


def arrow(ax, x1, y1, x2, y2, c=MUT, text=None, tx=0, ty=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16, color=c, lw=1.6))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, fontsize=10, color=c, ha='center', weight='bold')


def fp_of(tag):
    d = json.load(open(f'fp_{tag}_summary.json'))
    return sum(v['stop_frames'] for v in d.values()), sum(v['frames'] for v in d.values()), \
        sum(v['ms_med'] for v in d.values()) / len(d)


# ================================================================ 1. development stages (replaces v8 / v16 / v17)
stages = [('v8', 'Этап 1', 'глобальная геометрия\nпути (DP + Гаусс–Ньютон)'),
          ('v16', 'Этап 2', 'неопределённость σ(x)\nи зоны 2σ'),
          ('v17', 'Этап 3', 'физические тесты:\nгравитация, форма,\nстенка, оболочка'),
          ('hybrid', 'Этап 4', 'ансамбль ML v1\n(⅓ LGBM + ⅓ CatBoost\n+ ⅓ PI-MLP)'),
          ('final', 'Этап 5', 'вето оболочки\n+ ускорение numba')]
S = json.load(open('sealed_summary.json', encoding='utf-8'))
ev = [r['per_km'] for r in S['rows']]            # v1, ½+½, ½+½+ego on the sealed block of dataset 2
fig = plt.figure(figsize=(14.5, 5.4))
ax = fig.add_axes([0.04, 0.30, 0.56, 0.58])
vals = [fp_of(t) for t, _, _ in stages]
pct = [100 * a / b for a, b, _ in vals]
ax.bar(range(5), pct, color=[BLUSH] * 3 + [LAV, PURPLE], edgecolor=[PINK] * 3 + [LAV, PURPLE], width=0.62)
for i, ((a, b, ms), p) in enumerate(zip(vals, pct)):
    ax.text(i, p + 0.12, f'{p:.2f} %\n{a} кадр.', ha='center', fontsize=10.5, color=INK, weight='bold')
ax.set_xticks(range(5))
ax.set_xticklabels([f'{s}\n{d}' for _, s, d in stages], fontsize=9.5, color=INK)
ax.set_ylim(0, 5.2)
ax.set_ylabel('ложные STOP, % кадров', color=INK)
ax.set_title('Датасет 1 · 5 пустых тоннелей, 2 287 кадров', loc='left', fontsize=13, weight='bold', color=INK)
ax.grid(axis='y', alpha=0.25)
ax.text(-0.35, 4.85, 'Этап 0 (рельсы + стены): 249 из 252 кадров с ложной тревогой на одной кривой', fontsize=10,
        color=MUT)
ax2 = fig.add_axes([0.67, 0.30, 0.31, 0.58])
names = ['Этап 5 · v1\n⅓+⅓+⅓', 'Этап 6\n½ + ½\n(+ датасет 2)', 'Этап 7\n+ ЭГО-тест', 'Этап 8 · финал\nроутер + ЭГО']
ax2.bar(range(len(ev)), ev, color=['#B9B4C6', LAV, LAV, PINK][:len(ev)], width=0.6)
for i, v in enumerate(ev):
    ax2.text(i, v + 0.25, f'{v:.1f} / км', ha='center', fontsize=11, color=INK, weight='bold')
ax2.set_xticks(range(len(ev)))
ax2.set_xticklabels(names, fontsize=9.5, color=INK)
ax2.set_ylim(0, max(ev) * 1.25)
ax2.set_ylabel('ложные STOP-события на км', color=INK)
ax2.set_title('Датасет 2 · запечатанный тест, 2.8 км', loc='left', fontsize=13, weight='bold', color=INK)
ax2.grid(axis='y', alpha=0.25)
save(fig, 'evolution_stages.png')

# ================================================================ 2. decision logic
fig, ax = plt.subplots(figsize=(14, 7.2))
ax.set_xlim(0, 28); ax.set_ylim(0, 14.4); ax.axis('off')
box(ax, 0.3, 11.2, 5.2, 2.6, '1 · Кадр лидара', '10 Гц · до 921 600 точек\nобрезка FOV, прорежение\nближней зоны')
box(ax, 6.1, 11.2, 5.6, 2.6, '2 · Геометрия пути', 'ось, уклон, кривизна до 200 м\n+ неопределённость σ(x)')
box(ax, 12.3, 11.2, 7.2, 2.6, '3 · Точки → зоны габарита',
    'ВНУТРИ: в габарите, сжатом на 2σ\nРЯДОМ: у границы габарита\nучитываются точки выше 0.15 м')
box(ax, 20.1, 11.2, 7.6, 2.6, '4 · Кластеры → тесты', 'гравитация (> 60 м: опора на пол),\nформа, стенка за объектом,\nкрепление к оболочке тоннеля')
for x1, x2 in ((5.5, 6.1), (11.7, 12.3), (19.5, 20.1)):
    arrow(ax, x1, 12.5, x2, 12.5)
box(ax, 4.5, 6.2, 19.0, 4.1, '5 · Решение STOP для объекта (все условия одновременно)',
    '• подтверждён трекером: ≥ 3 обнаружений из 5 кадров (0.3 с)\n'
    '• ≥ 2 точки ВНУТРИ габарита и в пределах измеренной в этом кадре геометрии\n'
    '• до 30 м: физические тесты пройдены   ·   дальше 30 м: оценка эксперта (на полу / парящие) ≥ его порога\n'
    '  (сглажена по 5 кадрам); крепление к оболочке тоннеля = всегда вето\n'
    '• ЭГО-тест: объект приближается со скоростью поезда (не «едет» вместе с ним)',
    ec=PINK, lw=2.4, ts=13, bs=11.2, left=True)
arrow(ax, 23.9, 11.2, 18.0, 10.3)
box(ax, 0.8, 0.6, 7.8, 4.4, 'СТОП', 'есть хотя бы один объект,\nпрошедший шаг 5\n→ торможение; выдаются дистанция,\nвремя до столкновения, размеры',
    fc=RED, ec=RED, tc='white', bc='white', ts=19, bs=11)
box(ax, 10.1, 0.6, 7.8, 4.4, 'ВНИМАНИЕ', 'объект РЯДОМ с габаритом, или\nдальше измеренной геометрии, или\nотклонён ML / ЭГО-тестом, или\nещё не подтверждён (2 из 5)',
    fc=AMBER, ec=AMBER, tc='white', bc='white', ts=19, bs=11)
box(ax, 19.4, 0.6, 7.8, 4.4, 'ПУТЬ СВОБОДЕН', 'объектов нет\n→ «путь проверен до X м»\n(X = горизонт видимости\nи измеренной геометрии, ≤ 200 м)',
    fc=GREEN, ec=GREEN, tc='white', bc='white', ts=19, bs=11)
arrow(ax, 9.5, 6.2, 4.7, 5.0, RED, 'да', -0.6, 0.2)
arrow(ax, 14.0, 6.2, 14.0, 5.0, AMBER, 'нет, но есть кандидат', 2.4, 0.0)
arrow(ax, 18.5, 6.2, 23.3, 5.0, GREEN, 'кандидатов нет', 1.3, 0.2)
save(fig, 'decision_logic.png')

# ================================================================ 3. model stack: v1 vs final
fig, ax = plt.subplots(figsize=(14, 6.4))
ax.set_xlim(0, 28); ax.set_ylim(0, 12.8); ax.axis('off')
for x0, title, members, extra, ec, note in (
        (0.3, 'v1 (прежняя модель)', [('LightGBM', 'монотонный', '⅓'), ('CatBoost', 'симметр. деревья', '⅓'),
                                       ('PI-MLP', 'физически-\nинформированный MLP', '⅓')],
         None, '#9CA3AF', 'обучена на датасете 1 · порог 0.37'),
        (14.3, 'Финальная модель: роутер', [('LGBM + ½ CatBoost', 'эксперт «на полу»\nлюди, коробки · порог 0.425', '½'),
                                             ('LGBM + ½ CatBoost', 'эксперт «парящие / висящие»\nнизом ≥ 0.5 м · порог 0.45', '½')],
         ('ЭГО-тест', 'лидарная одометрия: объект обязан\nприближаться со скоростью поезда'), PINK,
         'обе модели: датасеты 1 + 2 без запечатанного блока; вторая — все формы опасности')):
    ax.text(x0 + 6.7, 12.3, title, ha='center', fontsize=15, weight='bold', color=INK)
    box(ax, x0, 9.2, 13.4, 2.4, 'Физический слой (правила, без обучения)',
        'геометрия пути · габарит с 2σ · гравитация · форма\nстенка за объектом · вето оболочки · трекер 3 из 5',
        ec=PURPLE, ts=12, bs=10.5)
    n = len(members)
    w = (13.4 - 0.4 * (n - 1)) / n
    for i, (nm, sub, wt) in enumerate(members):
        x = x0 + i * (w + 0.4)
        box(ax, x, 6.1, w, 2.4, f'{wt} · {nm}', sub, ec=ec, ts=13, bs=10)
    ax.text(x0 + 6.7, 5.1, 'логит = взвешенное среднее логитов членов\n→ сглаживание по 5 кадрам → порог',
            ha='center', va='center', fontsize=10, color=MUT)
    if extra:
        box(ax, x0, 1.9, 13.4, 2.5, extra[0], extra[1], fc='#FFF1F5', ec=PINK, ts=13, bs=10.5)
    else:
        ax.text(x0 + 6.7, 3.2, 'проверки движения нет', ha='center', fontsize=11, color=MUT, style='italic')
    ax.text(x0 + 6.7, 1.1, note, ha='center', fontsize=10.5, color=INK)
ax.plot([14.0, 14.0], [0.6, 12.6], color='#E5E7EB', lw=1.5)
save(fig, 'model_stack.png')

# ================================================================ 4. dataset-3 scorecard (organisers' objects)
rows = [('1 · 2×2 м в центре', 'да', 'STOP с 98 м', 'ok', '116 м'),
        ('2 · 0.3 м в центре', 'да', 'STOP с 20 м', 'ok', '99 м'),
        ('3 · 0.3 м на рельсе', 'да', 'STOP с 23 м', 'ok', '8 м*'),
        ('4 · 0.3 м у края габарита', 'да', 'STOP с 20 м', 'ok', 'нет'),
        ('5 · 0.3 м за габаритом, рядом', 'нет', 'STOP с 25 м', 'bad', 'нет STOP'),
        ('6 · 2×2 м у края, в габарите', 'да', 'STOP с 25 м', 'ok', '29 м'),
        ('7 · 2×2 м за габаритом', 'нет', 'STOP с 29 м', 'bad', 'нет STOP'),
        ('8 · 2×2 м «сверху габарита»', '?', 'STOP с 75 м', 'amb', '—'),
        ('9 · брус 2×0.2 м на рельсах', 'да', 'STOP с 60 м', 'ok', '45 м*'),
        ('10 · стержень 5 см с потолка', 'да', 'нет (4 точки на 12 м)', 'bad', '30 м')]
fig, ax = plt.subplots(figsize=(14, 7.3))
ax.set_xlim(0, 28); ax.set_ylim(-0.6, 12.6); ax.axis('off')
cx = [0.3, 9.6, 12.6, 20.6]
for x, h in zip(cx, ['Объект организаторов', 'нужен STOP?', 'их бэг · финальная модель', 'реплика · финал']):
    ax.text(x, 12.0, h, fontsize=12, weight='bold', color=PURPLE)
for r, (name, need, res, verdict, rep) in enumerate(rows):
    y = 10.9 - r * 1.08
    ax.add_patch(FancyBboxPatch((0.1, y - 0.42), 27.7, 0.9, boxstyle='round,pad=0,rounding_size=0.12',
                                fc='#F7F5FB' if r % 2 == 0 else 'white', ec='none'))
    ax.text(cx[0], y, name, fontsize=12, color=INK, va='center')
    ax.text(cx[1], y, need, fontsize=12, color=INK, va='center', weight='bold')
    col = {'ok': GREEN, 'bad': RED, 'amb': AMBER}[verdict]
    ax.text(cx[2], y, res + {'ok': '', 'bad': '  (ошибка)', 'amb': '  (?)'}[verdict], fontsize=12, va='center', weight='bold', color=col)
    ax.text(cx[3], y, rep, fontsize=11.5, va='center', color=MUT)
ax.text(0.3, -0.35, 'Их бэг: читаются 29 % архива (438 кадров, 1.8 км) — в них все 10 объектов. Ложных STOP на пустом тоннеле: 1 событие '
        '(5 кадров, горизонтальный срез свода на 146 м). 5 и 7: наш габарит — вагон 2.7 м, их ≈ ±1.15 м от оси лидара. '
        '* реплика: трасса идёт вверх, объекты тонут в балласте.', fontsize=9.5, color=MUT, wrap=True)
save(fig, 'ds3_scorecard.png')
print('figures ->', FIG)
