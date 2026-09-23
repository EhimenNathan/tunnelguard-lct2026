"""How the ensemble weights were chosen: grouped out-of-fold sweep of w in  logit = w*logit(LightGBM) + (1-w)*logit(CatBoost).

Same protocol as train_scorer_v3.py: leave-one-group-out over the 5 original recordings and the first three 5-minute
blocks of the new drive; for every held-out group the decision threshold is chosen on the *other* groups only (false-STOP
budget 0.3 % of frames), then false STOP frames and object recall are measured on the held-out group.  The sealed block
(group 13) is not used.  Output: weight_sweep.json and docs/figures/deck/weight_sweep.png.
usage (scratchpad):  python weight_sweep.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_scorer_v3 as T

BUDGET = 0.003
CV = [0, 1, 2, 3, 4, 10, 11, 12]
W = np.round(np.linspace(0, 1, 11), 2)

t0 = time.time()
p_l, p_c = np.zeros(len(T.X)), np.zeros(len(T.X))
for g in CV:
    te = T.grp == g
    tr = np.isin(T.grp, [x for x in CV if x != g])
    _, f1 = T.fit_lgb_mono(T.X[tr], T.y[tr])
    _, f2 = T.fit_cat(T.X[tr], T.y[tr])
    p_l[te], p_c[te] = f1(T.X[te]), f2(T.X[te])
    print(f'group {g} fitted [{time.time() - t0:.0f}s]', flush=True)
np.savez('oof_members_v3.npz', lgb=p_l, cat=p_c)

rows = []
for w in W:
    ps = T.smooth(1 / (1 + np.exp(-(w * T.lg(p_l) + (1 - w) * T.lg(p_c)))))
    fp = frames = det = n = 0
    for g in CV:
        tau = T.choose_tau(ps, [x for x in CV if x != g], BUDGET)
        m = T.metrics(T.stop_mask(ps, tau), [g])
        fp += m['fp']
        frames += m['frames']
        if m['n_obj']:
            det += m['recall'] * m['n_obj']
            n += m['n_obj']
    rows.append(dict(w_lgb=float(w), w_cat=float(1 - w), fp=fp, frames=frames, fp_rate=fp / frames, recall=det / n))
    print(f'w_lgb={w:.1f}  false STOP {fp}/{frames} ({100 * fp / frames:.2f} %)  recall {100 * det / n:.1f} %', flush=True)
json.dump(rows, open('weight_sweep.json', 'w'), indent=1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for fn in ('segoeui.ttf', 'segoeuib.ttf'):
    p = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', fn)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#C9C3D6', 'xtick.color': '#6B6F80', 'ytick.color': '#6B6F80'})
PUR, PINK, INK, MUT = '#520977', '#FF0053', '#1C1D22', '#6B6F80'
w = [r['w_lgb'] for r in rows]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ax, key, col, lab, lim, fmt in ((axs[0], 'recall', PUR, 'полнота обнаружения вне фолда', (0, 100), '{:.1f} %'),
                                    (axs[1], 'fp_rate', PINK, 'ложные STOP-кадры вне фолда', (0, 1.0), '{:.2f} %')):
    v = [100 * r[key] for r in rows]
    ax.fill_between(w, min(v), max(v), color=col, alpha=0.12, lw=0)
    ax.plot(w, v, '-o', color=col, lw=2.2, ms=6)
    ax.set_ylim(*lim)
    ax.set_xlabel('вес LightGBM  w₁   (CatBoost: w₂ = 1 − w₁)')
    ax.set_title(lab, loc='left', fontsize=13, weight='bold', color=INK)
    ax.axvline(0.5, color=INK, ls='--', lw=1)
    ax.text(0.52, lim[0] + 0.08 * (lim[1] - lim[0]), 'внедрено: w₁ = w₂ = ½', fontsize=10.5, color=INK)
    ax.text(0.0, lim[1] * 0.93, f'весь диапазон w: {fmt.format(min(v))} … {fmt.format(max(v))}', fontsize=11,
            color=col, weight='bold', zorder=5, bbox=dict(fc='white', ec='none', pad=2))
    ax.grid(axis='y', alpha=0.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:g} %'))
fig.suptitle('Выбор весов ансамбля: 11 значений w, 8 групп вне фолда, порог выбран на обучающих группах',
             x=0.01, ha='left', fontsize=12, color=MUT)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(HERE), 'docs', 'figures', 'deck', 'weight_sweep.png'), dpi=200,
            facecolor='white', bbox_inches='tight', pad_inches=0.12)
print('done', time.time() - t0)
