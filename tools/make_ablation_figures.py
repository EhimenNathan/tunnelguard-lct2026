"""Standard ablation plots from ablation_*.json (tools/ablation.py) and the leave-one-recording-out simulation of the
learned stage.  usage (from the scratchpad):  python make_ablation_figures.py [en|ru]"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

LANG = sys.argv[1] if len(sys.argv) > 1 else 'en'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'figures', 'ablation' if LANG == 'en' else 'deck')
os.makedirs(OUT, exist_ok=True)
for fn in ('segoeui.ttf', 'segoeuib.ttf'):
    p = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', fn)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
PINK, PURPLE, BLUSH, LAV, INK, MUT, GREEN = '#FF0053', '#520977', '#FFD6E3', '#8A83D1', '#1C1D22', '#6B6F80', '#16A36A'
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 12, 'axes.edgecolor': '#C9C3D6', 'axes.labelcolor': INK,
                     'xtick.color': MUT, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.titlesize': 14, 'legend.frameon': False})
T = {
    'en': dict(full='full system', no_ml='− learned scorer (rules only)', no_shell_veto='− shell veto in ML path',
               no_shell_test='− shell-attachment test', no_containment='− wall containment test', no_gravity='− gravity test',
               no_shape='− shape (beam extent) test', no_temporal_confirm='− M-of-N confirmation',
               no_score_smoothing='− score smoothing', no_sigma_zones='− 2σ uncertainty zones', no_two_tier='− two-tier range rule',
               no_temporal_geometry='− temporal geometry prior', straight_corridor='straight corridor (no curvature)',
               fp='false STOP frames (5 empty recordings, 2 287 frames)', dfp='false STOP frames', rec='confirmed STOP recall, %',
               hold='held-out real recording', hold_stop='person-in-gauge frames with STOP', hold_sp='spurious STOP detections',
               trade='trade-off', x_trade='false STOP frames, % of empty frames', y_trade='mean recall (person, box; 80–160 m), %',
               loro='Learned stage, leave-one-recording-out (unbiased)', loro_x='false STOP frames, %', loro_y='object recall, %',
               heat='Ablation summary (change vs. full system)'),
    'ru': dict(full='полная система', no_ml='− ML-скорер (только правила)', no_shell_veto='− вето оболочки в ML',
               no_shell_test='− тест крепления к оболочке', no_containment='− удержание стенкой', no_gravity='− тест гравитации',
               no_shape='− тест формы (шаг лучей)', no_temporal_confirm='− подтверждение M из N',
               no_score_smoothing='− сглаживание скора', no_sigma_zones='− зоны с учётом 2σ', no_two_tier='− правило измеренной дальности',
               no_temporal_geometry='− слияние геометрии во времени', straight_corridor='прямой коридор (без кривизны)',
               fp='ложные STOP-кадры (5 пустых записей, 2 287 кадров)', dfp='ложные STOP-кадры', rec='полнота STOP, %',
               hold='отложенная реальная запись', hold_stop='кадров с человеком в габарите — STOP', hold_sp='лишних STOP',
               trade='компромисс', x_trade='ложные STOP, % кадров', y_trade='средняя полнота (человек, коробка; 80–160 м), %',
               loro='ML-этап: leave-one-tunnel-out (честно)', loro_x='ложные STOP, % кадров', loro_y='полнота, %',
               heat='Абляция: изменение относительно полной системы'),
}[LANG]

order = ['full', 'no_ml', 'no_shell_veto', 'no_shell_test', 'no_containment', 'no_gravity', 'no_shape', 'no_temporal_confirm',
         'no_score_smoothing', 'no_sigma_zones', 'no_two_tier', 'no_temporal_geometry', 'straight_corridor']
A = {}
for p in glob.glob('ablation_*.json'):
    d = json.load(open(p))
    A[d['variant']] = d
order = [v for v in order if v in A]
full = A['full']
KEYS = ['person@80', 'person@120', 'person@160', 'box_50cm@80', 'box_50cm@120', 'box_50cm@160']


def mean_rec(d):
    return 100 * np.mean([d['recall'].get(k, 0.0) for k in KEYS])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)


labels = [T[v] for v in order]
y = np.arange(len(order))[::-1]

# ---------------------------------------------------------------- 1. false alarms per variant
fig, ax = plt.subplots(figsize=(9.5, 6.2))
vals = [A[v]['fp_frames'] for v in order]
cols = [PURPLE if v == 'full' else (PINK if A[v]['fp_frames'] > full['fp_frames'] else LAV) for v in order]
ax.barh(y, vals, color=cols, height=0.66)
for yi, v, val in zip(y, order, vals):
    d = val - full['fp_frames']
    ax.text(val + max(vals) * 0.01, yi, f'{val}' + ('' if v == 'full' else f'  ({d:+d})'), va='center', fontsize=11, color=INK)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel(T['fp']); ax.set_xlim(0, max(vals) * 1.22 + 1)
ax.grid(axis='x', alpha=0.25)
save(fig, 'ablation_false_alarms.png')

# ---------------------------------------------------------------- 2. recall by distance per variant (grouped bars)
fig, ax = plt.subplots(figsize=(9.5, 6.8))
dists = ['80', '120', '160']
shades = [PURPLE, PINK, LAV]
h = 0.26
for j, dd in enumerate(dists):
    vals = [100 * np.mean([A[v]['recall'].get(f'person@{dd}', 0), A[v]['recall'].get(f'box_50cm@{dd}', 0)]) for v in order]
    ax.barh(y + (1 - j) * h, vals, height=h, color=shades[j], label=f'{dd} m' if LANG == 'en' else f'{dd} м')
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel(T['rec'] + (' (person + 0.5 m box)' if LANG == 'en' else ' (человек + коробка 0.5 м)'))
ax.set_xlim(0, 105); ax.legend(loc='lower right', ncol=3)
ax.grid(axis='x', alpha=0.25)
save(fig, 'ablation_recall.png')

# ---------------------------------------------------------------- 3. trade-off scatter
fig, (ax, kx) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={'width_ratios': [1.35, 1]})
fx, fy = 100 * full['fp_frames'] / full['frames'], mean_rec(full)
key_lines = []
num = 0
for v in order:
    x_, y_ = 100 * A[v]['fp_frames'] / A[v]['frames'], mean_rec(A[v])
    same_as_full = v != 'full' and abs(x_ - fx) < 1e-9 and abs(y_ - fy) < 1e-9
    c = PURPLE if v == 'full' or same_as_full else (PINK if (x_ > fx + 1e-9 or y_ < fy - 1e-9) else GREEN)
    num += 1
    if not same_as_full:
        ax.scatter(x_, y_, s=260 if v == 'full' else 190, color=c, zorder=3, edgecolor='white', lw=1)
        ax.text(x_, y_, str(num), ha='center', va='center', fontsize=8.5, color='white', weight='bold', zorder=4)
        if v != 'full':
            ax.annotate('', xy=(x_, y_), xytext=(fx, fy), arrowprops=dict(arrowstyle='->', color=c, lw=0.9, alpha=0.45,
                                                                           shrinkA=8, shrinkB=8))
    note = (' (= full)' if LANG == 'en' else ' (= полной)') if same_as_full else ''
    key_lines.append((num, T[v] + note, c, x_, y_))
ax.set_xlabel(T['x_trade']); ax.set_ylabel(T['y_trade'])
ax.set_xscale('symlog', linthresh=0.2)
ax.grid(alpha=0.25)
kx.axis('off')
for i, (n_, lab, c, x_, y_) in enumerate(key_lines):
    yy = 1 - (i + 0.5) / len(key_lines)
    kx.scatter([0.03], [yy], s=150, color=c, transform=kx.transAxes, clip_on=False)
    kx.text(0.03, yy, str(n_), ha='center', va='center', fontsize=8, color='white', weight='bold', transform=kx.transAxes)
    kx.text(0.09, yy, f'{lab}   {x_:.2f} % · {y_:.1f} %', va='center', fontsize=10, color=INK, transform=kx.transAxes)
save(fig, 'ablation_tradeoff.png')

# ---------------------------------------------------------------- 4. held-out recording
fig, axs = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True, gridspec_kw={'width_ratios': [1.4, 1]})
vals = [A[v]['hold_stop'] for v in order]
axs[0].barh(y, vals, color=[PURPLE if v == 'full' else LAV for v in order], height=0.66)
for yi, val, v in zip(y, vals, order):
    axs[0].text(val + 0.6, yi, f'{val}/{A[v]["hold_inside"]}', va='center', fontsize=10.5)
axs[0].set_yticks(y); axs[0].set_yticklabels(labels, fontsize=11)
axs[0].set_xlabel(T['hold_stop']); axs[0].set_xlim(0, max(max(vals), 1) * 1.25)
vals = [A[v]['hold_spurious'] for v in order]
axs[1].barh(y, vals, color=[PURPLE if v == 'full' else (PINK if val > full['hold_spurious'] else LAV) for v, val in zip(order, vals)], height=0.66)
for yi, val in zip(y, vals):
    axs[1].text(val + 0.3, yi, f'{val}', va='center', fontsize=10.5)
axs[1].set_xlabel(T['hold_sp']); axs[1].set_xlim(0, max(max(vals), 1) * 1.3)
for ax in axs:
    ax.grid(axis='x', alpha=0.25)
fig.suptitle(T['hold'], x=0.02, ha='left', fontweight='bold')
save(fig, 'ablation_holdout.png')

# ---------------------------------------------------------------- 5. heatmap table of deltas
cols = ['fp', 'hold_stop', 'hold_sp'] + KEYS
head = {'en': ['false STOP\nframes', 'held-out\nSTOP frames', 'held-out\nspurious', 'person\n80 m', 'person\n120 m', 'person\n160 m',
               'box\n80 m', 'box\n120 m', 'box\n160 m'],
        'ru': ['ложные\nSTOP', 'STOP с\nчеловеком', 'лишние\nSTOP', 'человек\n80 м', 'человек\n120 м', 'человек\n160 м',
               'коробка\n80 м', 'коробка\n120 м', 'коробка\n160 м']}[LANG]
raw = np.array([[A[v]['fp_frames'], A[v]['hold_stop'], A[v]['hold_spurious']] + [100 * A[v]['recall'].get(k, 0) for k in KEYS]
                for v in order], float)
delta = raw - raw[0]
good = delta.copy()
good[:, 0] *= -1; good[:, 2] *= -1          # fewer false alarms is better
# log-compressed, per-column normalised colour so a single extreme variant does not wash out the others
comp = np.sign(good) * np.log1p(np.abs(good))
scale = np.array([max(np.log1p(5.0), np.abs(comp[:, j]).max()) for j in range(comp.shape[1])])
fig, ax = plt.subplots(figsize=(12, 6.6))
ax.imshow(np.clip(comp / scale, -1, 1) * 0.8, cmap='PiYG', norm=TwoSlopeNorm(0, -1, 1), aspect='auto')
for i in range(len(order)):
    for j in range(raw.shape[1]):
        txt = f'{raw[i, j]:.0f}' if i == 0 else (f'{delta[i, j]:+.0f}' if abs(delta[i, j]) > 1e-9 else '0')
        ax.text(j, i, txt, ha='center', va='center', fontsize=10.5, color=INK, weight='bold' if i == 0 else 'normal')
ax.set_xticks(range(raw.shape[1])); ax.set_xticklabels(head, fontsize=10)
ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=10.5)
ax.xaxis.tick_top()
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(T['heat'] + (' — green = better, pink = worse' if LANG == 'en' else ' — зелёный лучше, розовый хуже'), loc='left', pad=40)
save(fig, 'ablation_heatmap.png')

# ---------------------------------------------------------------- 6. learned stage, leave-one-recording-out
loro = [  # (label_en, label_ru, fp %, recall %) — tools/train_scorer.py and tools/tune_scorer.py outputs
    ('rules only', 'только правила', 1.31, 62.8),
    ('ML only (LightGBM)', 'только ML (LightGBM)', 1.18, 64.5),
    ('rules AND ML', 'правила И ML', 0.22, 61.2),
    ('hybrid, smoothing 3', 'гибрид, сглаживание 3', 0.44, 64.5),
    ('hybrid, smoothing 5, no veto', 'гибрид, сглаживание 5, без вето', 0.31, 64.5),
    ('hybrid, smoothing 5 + veto (deployed)', 'гибрид, сглаживание 5 + вето (внедрён)', 0.31, 64.5),
]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
yy = np.arange(len(loro))[::-1]
names = [r[0] if LANG == 'en' else r[1] for r in loro]
for ax, idx, xl, fmt in [(axs[0], 2, T['loro_x'], '{:.2f}'), (axs[1], 3, T['loro_y'], '{:.1f}')]:
    vals = [r[idx] for r in loro]
    ax.barh(yy, vals, color=[PURPLE if 'deployed' in r[0] else LAV for r in loro], height=0.62)
    for y_, v in zip(yy, vals):
        ax.text(v * 1.01 + (0.02 if idx == 2 else 0.3), y_, fmt.format(v), va='center', fontsize=10.5)
    ax.set_xlabel(xl); ax.grid(axis='x', alpha=0.25)
axs[0].set_yticks(yy); axs[0].set_yticklabels(names, fontsize=10.5)
axs[0].set_xlim(0, 1.6); axs[1].set_xlim(55, 67)
fig.suptitle(T['loro'], x=0.02, ha='left', fontweight='bold')
save(fig, 'ablation_learned_stage.png')
print('ablation figures ->', os.path.abspath(OUT), 'variants:', order)
