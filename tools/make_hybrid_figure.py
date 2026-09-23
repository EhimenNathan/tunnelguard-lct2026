"""Figure for the learned scorer: LORO model comparison and decision-level false-alarm / recall trade-off.
Numbers are the leave-one-recording-out results of tools/train_scorer.py and tools/tune_scorer.py (docs/EXPERIMENTS.md §4)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')
INK, MUT, PUR, PINK, LAV = '#1c1d22', '#6b6f80', '#520978', '#ff0053', '#8a83d1'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': '#b0b3c0', 'axes.labelcolor': INK, 'xtick.color': MUT,
                     'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False})

models = ['логистическая\nрегрессия', 'MLP', 'PI-MLP*', 'XGBoost', 'LightGBM\nмонотонный*', 'CatBoost*']
ap = [0.877, 0.864, 0.867, 0.884, 0.893, 0.894]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [1.25, 1]})
ax = axs[0]
cols = [LAV if not m.endswith('*') else PUR for m in models]
ax.barh(range(len(models)), ap, color=cols, height=0.62)
for i, v in enumerate(ap):
    ax.text(v + 0.001, i, f'{v:.3f}', va='center', color=INK, fontsize=10)
ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=10)
ax.set_xlim(0.84, 0.905); ax.set_xlabel('средняя точность (AP), тест на невиденном тоннеле')
ax.set_title('Модели на 38 физических признаках\n* — в ансамбле (после Optuna)', color=INK, fontsize=12, loc='left')

ax = axs[1]
pts = [('только правила', 1.31, 62.8, MUT), ('только ML', 1.01, 64.5, LAV), ('правила И ML', 0.31, 61.2, LAV),
       ('гибрид (внедрён)', 0.31, 64.5, PINK)]
for name, fp, rec, c in pts:
    ax.scatter(fp, rec, s=160 if c == PINK else 90, color=c, zorder=3)
    ax.annotate(name, (fp, rec), textcoords='offset points', xytext=(8, 6 if name != 'правила И ML' else -14),
                color=INK, fontsize=10)
ax.annotate('', xy=(0.36, 64.4), xytext=(1.27, 62.9), arrowprops=dict(arrowstyle='->', color=PINK, lw=1.6))
ax.set_xlim(0, 1.6); ax.set_ylim(60.5, 65.5)
ax.set_xlabel('ложные STOP, % кадров без препятствий'); ax.set_ylabel('полнота обнаружения, %')
ax.set_title('Решение: leave-one-tunnel-out\nв 4 раза меньше ложных остановок', color=INK, fontsize=12, loc='left')
ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'hybrid.png'), dpi=170, facecolor='white'); plt.close()
