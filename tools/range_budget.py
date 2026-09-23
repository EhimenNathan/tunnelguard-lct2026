"""Where does detection range go?  Physical visibility of the track corridor in the provided recordings versus the
Pandar128 link budget.  Run from the scratchpad (range-image cache + fp_final_*.npz).  Writes docs/figures/range_budget.png
and range_budget.json."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from loader import BagCache
from tunnel_guard.core.synth import max_range

for fn in ('segoeui.ttf', 'segoeuib.ttf'):
    p = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', fn)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
PINK, PURPLE, LAV, INK, MUT = '#FF0053', '#520977', '#8A83D1', '#1C1D22', '#6B6F80'
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#C9C3D6', 'axes.titleweight': 'bold', 'legend.frameon': False})
EMPTY = ['roundT_doubleT', 'squareT_platform_squareT_switch', 'doubleT_platform', 'roundT_squareT_pressureGate_squareT',
         'roundT_pressureGate_roundT']
RU = os.environ.get('LANG_FIG') == 'ru'
SHORT = (['круглый →\nдвухпутный', 'платформа\n+ стрелка', 'двухпутный,\nплатформа', 'гермо-\nзатвор 1', 'гермо-\nзатвор 2']
         if RU else ['round →\ndouble-track', 'platform\n+ switch', 'double-track\nplatform', 'pressure\ngate 1', 'pressure\ngate 2'])
L = (dict(vis='99.9 % отражений ближе', geo='геометрия пути измерена до', clr='путь проверен свободным до',
          good='хорошо (критерий кейса)', vgood='очень хорошо', exc='отлично', y1='расстояние вдоль пути, м',
          t1='Физическая видимость коридора в записях', refl='отражение', reach='дальность', min3='≈ 3 отражения за кадр — минимум для кластера',
          x2='расстояние, м', y2='ожидаемых отражений за кадр (человек)', t2='Энергетика и угловой шаг Pandar128',
          out=os.path.join(SOL, 'docs', 'figures', 'deck', 'range_budget.png'))
     if RU else
     dict(vis='99.9 % of lidar returns', geo='track geometry measured to', clr='verified clear distance',
          good='good (case criteria)', vgood='very good', exc='excellent', y1='distance along the track, m',
          t1='Physical visibility of the corridor in the recordings', refl='reflectivity', reach='reach',
          min3='about 3 returns per frame: minimum for a cluster', x2='distance, m', y2='expected returns per frame (person-sized)',
          t2='Pandar128 link budget and angular sampling', out=os.path.join(SOL, 'docs', 'figures', 'range_budget.png')))

out = {}
vis_all, valid_all, clear_all = [], [], []
for nm in EMPTY:
    rows = np.load(f'fp_final_{nm}.npz')['rows']
    b = BagCache(nm)
    far = []
    for f in range(5, b.n, 5):
        rng, _ = b.frame(f)
        V = rng > 0.5
        x = (b.dirs[V] * rng[V][:, None])[:, 0]
        far.append(float(np.percentile(x, 99.9)))
    out[nm] = dict(visibility_p50=float(np.median(far)), visibility_max=float(np.max(far)),
                   valid_p50=float(np.median(rows[:, 7])), valid_max=float(rows[:, 7].max()),
                   clear_p50=float(np.median(rows[:, 5])), clear_max=float(rows[:, 5].max()))
    vis_all.append(far); valid_all.append(rows[:, 7]); clear_all.append(rows[:, 5])
    print(nm, out[nm], flush=True)

fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={'width_ratios': [1.3, 1]})
ax = axs[0]
pos = np.arange(len(EMPTY))
for k, (data, col, lab, off) in enumerate([(vis_all, LAV, L['vis'], -0.25),
                                            (valid_all, PURPLE, L['geo'], 0.0),
                                            (clear_all, PINK, L['clr'], 0.25)]):
    bp = ax.boxplot(data, positions=pos + off, widths=0.22, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(col); patch.set_alpha(0.85); patch.set_edgecolor(col)
    for med in bp['medians']:
        med.set_color('white')
    ax.plot([], [], 's', color=col, label=lab)
for yv, lab in [(100, L['good']), (200, L['vgood']), (300, L['exc'])]:
    ax.axhline(yv, color=MUT, ls=':', lw=1)
    ax.text(len(EMPTY) - 0.5, yv + 4, lab, color=MUT, fontsize=10, ha='right')
ax.set_xticks(pos); ax.set_xticklabels(SHORT, fontsize=10.5)
ax.set_ylabel(L['y1']); ax.set_ylim(0, 320)
ax.set_title(L['t1'], loc='left')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10)
ax = axs[1]
d = np.linspace(20, 320, 301)
for rho, col in [(0.05, LAV), (0.10, PURPLE), (0.15, PINK), (0.40, '#C58BB0')]:
    rmax = max_range(rho, cap=1e9)
    # returns of a 0.5 m x 1.75 m target sampled by 0.1 deg x 0.125 deg beams, fading over the last 30 % of reach
    n = (0.5 / (0.00175 * d)) * (1.75 / (0.00218 * d)) * np.clip((rmax - d) / (0.3 * rmax), 0, 1) * (d <= 209)
    ax.semilogy(d, np.maximum(n, 1e-2), color=col, lw=2.2, label=f"{L['refl']} {int(rho * 100)} % ({L['reach']} {min(rmax, 200):.0f} {'м' if RU else 'm'})")
ax.axvspan(205, 320, color='#E9E4F0', lw=0, zorder=0)
ax.text(212, 0.2, ('за пределом\nизмерения:\n200 м по\nпаспорту' if RU else 'beyond the\ninstrumented\nrange (200 m)'),
        color=PURPLE, fontsize=9.5)
ax.axhline(3, color=MUT, ls='--', lw=1)
ax.text(24, 3.4, L['min3'], color=MUT, fontsize=9.5)
ax.set_ylim(0.1, 2000); ax.set_xlabel(L['x2']); ax.set_ylabel(L['y2'])
ax.set_title(L['t2'], loc='left')
ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(0.66, 1.0))
ax.grid(alpha=0.25, which='both')
plt.tight_layout()
plt.savefig(L['out'], dpi=180, facecolor='white')
json.dump(out, open('range_budget.json', 'w'), indent=1)
print('written')
