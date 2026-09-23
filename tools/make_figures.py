"""Figures for the presentation/docs: architecture diagram, track-geometry illustration, free-space envelope,
demo frame + demo video (held-out recording)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'tunnel_guard'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'tunnel_guard', 'tools'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from loader import BagCache
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.gauge import GaugeConfig, envelope_polygon

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'figures')
os.makedirs(FIG, exist_ok=True)
PUR, PINK, LAV, INK, MUT = '#520978', '#ff0053', '#8a83d1', '#1c1d22', '#6b6f80'
what = sys.argv[1:] or ['arch', 'geometry', 'envelope', 'demo']

if 'arch' in what:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis('off')

    def box(x, y, w, h, title, sub, fc='#ffffff', ec=PUR, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.15', fc=fc, ec=ec, lw=2))
        ax.text(x + w / 2, y + h - 0.32, title, ha='center', va='top', fontsize=12.5, weight='bold', color=tc)
        ax.text(x + w / 2, y + h - 0.85, sub, ha='center', va='top', fontsize=9.5, color=MUT if tc == INK else '#f3e9ff', linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=PINK, lw=2.2))

    box(0.1, 2.0, 1.9, 1.6, 'Лидар / bag', 'Pandar128\nPointCloud2\nлюбой топик')
    box(2.5, 2.0, 2.0, 1.6, 'Подготовка', 'декодирование\nось «вперёд»\nFOV, прореживание')
    box(5.0, 0.3, 3.2, 4.6, '', '', fc='#f6f1fb')
    ax.text(6.6, 4.75, 'Геометрия пути', ha='center', va='top', fontsize=12.5, weight='bold', color=PUR)
    for i, (t, s) in enumerate([('Рельсы', 'BEV-хребты + DP (Витерби)'), ('Шаблон сечения', 'модель «нормального тоннеля»'),
                                ('Глобальный DP', 'грубый профиль до 300 м'), ('Гаусс–Ньютон', 'плотное выравнивание + клотоида'),
                                ('Неопределённость', 'σ(x), слияние во времени')]):
        y = 3.85 - i * 0.82
        ax.add_patch(FancyBboxPatch((5.25, y), 2.7, 0.62, boxstyle='round,pad=0.01,rounding_size=0.1', fc='white', ec=LAV, lw=1.2))
        ax.text(5.4, y + 0.43, t, fontsize=10, weight='bold', color=INK, va='center')
        ax.text(5.4, y + 0.17, s, fontsize=8.2, color=MUT, va='center')
    box(8.7, 2.0, 2.0, 1.6, 'Габарит', 'зоны IN / NEAR\nс учётом σ\nгоризонт видимости')
    box(8.7, 0.1, 2.0, 1.5, 'Кластеры', 'удержание стенкой\nкрепление к оболочке')
    box(11.1, 2.0, 1.8, 1.6, 'Решение', 'подтверждение M/N\nрасстояние, TTC\nCLEAR/CAUTION/STOP', fc=PUR, ec=PUR, tc='white')
    arrow(2.0, 2.8, 2.5, 2.8); arrow(4.5, 2.8, 5.0, 2.8); arrow(8.2, 2.8, 8.7, 2.8); arrow(9.7, 2.0, 9.7, 1.6)
    arrow(10.7, 0.85, 11.5, 2.0); arrow(10.7, 2.8, 11.1, 2.8)
    ax.text(12.0, 1.45, '/tunnel_guard/status\n/detections (vision_msgs)\n/alarm  /markers', ha='center', va='top', fontsize=8.5, color=INK)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'architecture.png'), dpi=180, transparent=False, facecolor='white'); plt.close()

if 'geometry' in what:
    b = BagCache('roundT_doubleT')
    det = ObstacleDetector(DetectorConfig(forward_axis='x'))
    for f in range(100, 127):
        p, I, _ = b.points(f)
        res = det.process(p, b.t[f], intensity=I)
    g = res.geometry
    P = res.forward_points
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [2.3, 1]})
    ax = axs[0]
    s = (P[:, 0] < 150) & (np.abs(P[:, 1]) < 16)
    ax.scatter(P[s, 0], P[s, 1], s=0.08, c='#9aa0b5', linewidths=0)
    xs = np.linspace(0, min(150, res.clear_distance), 200)
    ax.plot(xs, g.centre(xs), color=PINK, lw=2.2, label='ось пути (оценка)')
    ax.plot(xs, g.centre(xs) + 1.35, color=PUR, lw=1.2); ax.plot(xs, g.centre(xs) - 1.35, color=PUR, lw=1.2, label='габарит ±1.35 м')
    rl = g.meas['rails']; ok = rl['ev'] > 0.05
    ax.scatter(rl['xc'][ok], rl['yc'][ok], s=14, c=LAV, zorder=3, label='рельсы (DP)')
    ax.set_xlim(0, 150); ax.set_ylim(-6, 14); ax.set_xlabel('вперёд, м'); ax.set_ylabel('влево, м')
    ax.set_title('Кривая R≈500 м: путь восстановлен по лидару', loc='left', color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper left'); ax.grid(alpha=.2)
    ax = axs[1]
    x, l, h = g.to_track(P)
    for lo, hi, c, sz in [(4, 30, INK, 0.3), (40, 80, PINK, 2)]:
        m = (x > lo) & (x < hi)
        ax.scatter(l[m], h[m], s=sz, c=c, linewidths=0, label=f'{lo}–{hi} м')
    poly = envelope_polygon(GaugeConfig())
    ax.plot(poly[:, 0], poly[:, 1], color=PUR, lw=2)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-1, 5); ax.set_aspect('equal'); ax.grid(alpha=.2)
    ax.set_title('Сечения совпадают после выравнивания', loc='left', color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='lower right', markerscale=4)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'geometry.png'), dpi=170); plt.close()

if 'envelope' in what:
    EMPTY = ['roundT_doubleT', 'squareT_platform_squareT_switch', 'doubleT_platform', 'roundT_squareT_pressureGate_squareT', 'roundT_pressureGate_roundT']
    L = np.arange(-5, 5.0001, 0.05); H = np.arange(-1.0, 5.0001, 0.05)
    tot = 0; n = 0
    for nm in EMPTY:
        z = np.load(f'clear_{nm}.npz'); tot = tot + z['occ_near']; n += int(z['n'])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.imshow(np.log10(tot.T / n + 1e-3), origin='lower', extent=[L[0], L[-1], H[0], H[-1]], aspect='auto', cmap='magma', vmin=-3, vmax=0)
    poly = envelope_polygon(GaugeConfig())
    ax.plot(poly[:, 0], poly[:, 1], color='#00e5ff', lw=2.2)
    ax.set_xlabel('поперёк пути, м'); ax.set_ylabel('над головкой рельса, м')
    ax.set_title('Свободное пространство всех записей\nи габарит обнаружения', loc='left', fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'envelope.png'), dpi=170); plt.close()

if 'demo' in what:
    raise SystemExit('the demo video is rendered by tools/make_video.py')
