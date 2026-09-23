"""Russian-language figures for the new-line (dataset 2) slides: the traversal self-labelling principle and the sealed
test.  Run from the scratchpad directory; the sealed-test chart reads sealed_summary.json (tools/sealed_summary.py)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from matplotlib import font_manager

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'figures', 'deck')
for fn in ('segoeui.ttf', 'seguisb.ttf', 'segoeuib.ttf'):
    p = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', fn)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
PINK, PURPLE, INK, MUT, LAV = '#FF0053', '#520977', '#1C1D22', '#6B6F80', '#8A83D1'   # template palette (deck figures)
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 13, 'axes.edgecolor': '#C9C3D6', 'axes.labelcolor': INK,
                     'xtick.color': MUT, 'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.titlesize': 15, 'axes.titlecolor': INK, 'legend.frameon': False})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)


def tunnel(ax, y, train_x, alarm_x, title, passed):
    ax.add_patch(Rectangle((0, y - 1.1), 100, 2.2, fc='#F4F2F8', ec='none'))
    ax.plot([0, 100], [y - 1.1, y - 1.1], color='#C9C3D6', lw=1.2)
    ax.plot([0, 100], [y + 1.1, y + 1.1], color='#C9C3D6', lw=1.2)
    ax.add_patch(FancyBboxPatch((train_x - 14, y - 0.6), 14, 1.2, boxstyle='round,pad=0.02,rounding_size=0.4',
                                fc=PURPLE, ec='none'))
    ax.text(train_x - 7, y, 'поезд', ha='center', va='center', color='white', fontsize=11, weight='bold')
    if not passed:
        ax.annotate('', xy=(alarm_x - 0.6, y), xytext=(train_x + 0.4, y),
                    arrowprops=dict(arrowstyle='->', color=MUT, lw=1.2, ls='--'))
        ax.text((train_x + alarm_x) / 2, y + 0.35, 's — расстояние до тревоги', ha='center', color=MUT, fontsize=10)
    ax.plot(alarm_x, y, marker='X', ms=16, color=PINK if not passed else '#B9B4C6', mec='white', mew=1.2)
    ax.plot([alarm_x, alarm_x], [y - 1.1, y + 1.1], color=PINK if not passed else '#B9B4C6', lw=1, ls=':')
    if not passed:
        ax.text(alarm_x, y - 1.55, 'X = x_поезда + s', ha='center', color=PINK, fontsize=10.5)
    else:
        ax.text(alarm_x, y - 1.55, 'X: метка «ложная тревога» — доказано физикой, без ручной разметки', ha='center',
                color=PURPLE, fontsize=11, weight='bold')
    ax.text(0, y + 1.5, title, ha='left', color=INK, fontsize=13, weight='bold')


fig, ax = plt.subplots(figsize=(11.5, 5.6))
tunnel(ax, 6.0, 30, 72, '1 · кадр t: тревога на координате тоннеля X (одометрия по стенам)', passed=False)
tunnel(ax, 1.0, 88, 72, '2 · позже: поезд проехал X — твёрдого объекта там быть не могло', passed=True)
ax.set_xlim(-1, 101); ax.set_ylim(-1.2, 8.2); ax.axis('off')
save(fig, 'ds2_traversal.png')

if os.path.exists('sealed_summary.json'):
    S = json.load(open('sealed_summary.json', encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    names = [r['label'] for r in S['rows']]
    mean = [r['events_mean'] for r in S['rows']]
    lo = [r['events_mean'] - r['events_min'] for r in S['rows']]
    hi = [r['events_max'] - r['events_mean'] for r in S['rows']]
    cols = ['#B9B4C6'] + [LAV] * (len(names) - 2) + [PINK]
    ax.bar(range(len(names)), mean, color=cols, width=0.62, yerr=[lo, hi], capsize=5, ecolor=INK)
    for i, r in enumerate(S['rows']):
        ax.text(i, r['events_max'] + 0.8, f"{r['events_mean']:.1f}\n{r['per_km']:.1f} / км", ha='center', color=INK,
                fontsize=11, weight='bold' if i == len(names) - 1 else 'normal')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel('ложные STOP-события за 5 мин (2.8 км)')
    ax.set_ylim(0, max(r['events_max'] for r in S['rows']) * 1.3 + 2)
    ax.set_title('Запечатанный тест: последние 5 минут новой линии', loc='left')
    ax.text(0.99, 0.97, 'среднее по 3 зёрнам подвыборки; усы — мин…макс', transform=ax.transAxes, ha='right', va='top',
            color=MUT, fontsize=10)
    save(fig, 'ds2_sealed.png')
print('figures ->', FIG)

# ---------------------------------------------------------------- leakage-free validation scheme
from matplotlib.patches import Rectangle as R_

cols = ['T1', 'T2', 'T3', 'T4', 'T5', 'люди', 'B0', 'B1', 'B2', 'B3']
TR, VA, NEVER, SEAL = '#8A83D1', '#FF0053', '#E4E1EC', '#1C1D22'
fig, ax = plt.subplots(figsize=(12.5, 6.6))
ax.axis('off')
x0, cw, rh = 2.6, 1.0, 0.62
groups = list(range(5)) + [6, 7, 8]
rows = [(f'фолд {k + 1}', g) for k, g in enumerate(groups)] + [('финал', None)]
for r, (name, val) in enumerate(rows):
    y = -r * (rh + 0.12) - (0.35 if name == 'финал' else 0)
    ax.text(x0 - 0.2, y + rh / 2, name, ha='right', va='center', fontsize=11.5,
            color=PURPLE if name == 'финал' else MUT, weight='bold' if name == 'финал' else 'normal')
    for c in range(len(cols)):
        gap = 0.35 if c >= 5 else 0
        gap += 0.35 if c >= 6 else 0
        x = x0 + c * cw + gap
        if c == 5:
            fc = NEVER if name != 'финал' else '#16c784'
        elif c == 9:
            fc = SEAL if name != 'финал' else '#16c784'
        elif val is not None and c == val:
            fc = VA
        else:
            fc = TR
        ax.add_patch(R_((x, y), cw - 0.08, rh, fc=fc, ec='white', lw=1.5))
        if c in (5, 9) and name != 'финал':
            ax.text(x + (cw - 0.08) / 2, y + rh / 2, '—' if c == 5 else '🔒' if False else '×', ha='center',
                    va='center', color=MUT if c == 5 else 'white', fontsize=11)
yt = 0.95
for c, lab in enumerate(cols):
    gap = (0.35 if c >= 5 else 0) + (0.35 if c >= 6 else 0)
    ax.text(x0 + c * cw + gap + (cw - 0.08) / 2, yt, lab, ha='center', va='bottom', fontsize=11.5, color=INK,
            weight='bold')
ax.text(x0 + 2.46, yt + 0.55, 'Датасет 1 · 5 пустых тоннелей', ha='center', fontsize=12, color=INK, weight='bold')
ax.text(x0 + 5 * cw + 0.35 + 0.46, yt + 0.55, 'отложена', ha='center', fontsize=12, color=INK, weight='bold')
ax.text(x0 + 6 * cw + 0.7 + 1.96, yt + 0.55, 'Датасет 2 · 4 блока по 5 мин', ha='center', fontsize=12, color=INK,
        weight='bold')
yb = -len(rows) * (rh + 0.12) - 0.55
items = [(TR, 'обучение (порог выбирается только здесь)'), (VA, 'проверка фолда'),
         (NEVER, 'запись с людьми: не используется'), (SEAL, 'B3 запечатан: не используется'),
         ('#16c784', 'финальный тест')]
for j, (fc, lab) in enumerate(items):
    xx, yy = x0 - 1.8 + (j % 3) * 4.6 + (0.9 if j % 3 else 0), yb - (j // 3) * 0.55
    ax.add_patch(R_((xx, yy), 0.32, 0.32, fc=fc, ec='#C9C3D6', lw=0.8))
    ax.text(xx + 0.45, yy + 0.16, lab, va='center', fontsize=10.5, color=INK)
ax.set_xlim(x0 - 2.2, x0 + 10 * cw + 1.0)
ax.set_ylim(yb - 0.85, yt + 1.1)
save(fig, 'leakage_cv.png')
print('leakage scheme ->', FIG)
