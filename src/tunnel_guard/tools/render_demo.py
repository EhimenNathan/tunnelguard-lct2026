"""Render TunnelGuard demo frames: driver's-eye lidar view, metrics panel, bird's-eye strip, decision banner.

  img = render_frame(res, gauge_cfg, info=dict(chapter=..., subtitle=..., clock=..., points=...))
  img = title_card(lines, ...)            # full-frame text cards for the video

The perspective camera sits at the lidar and looks along the track; points are coloured by range.
All text is rasterised, so the video does not depend on fonts installed on the viewer's machine.
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from tunnel_guard.core.gauge import envelope_polygon

W, H, DPI = 1600, 900, 100
BG = '#0a0b12'
PANEL = '#141625'
INK = '#e9ebf5'
MUTED = '#8c91a8'
PURPLE = '#8a3ffc'
LEVEL_TXT = {0: 'ПУТЬ СВОБОДЕН', 1: 'ВНИМАНИЕ', 2: 'СТОП — ПРЕПЯТСТВИЕ'}
LEVEL_EN = {0: 'CLEAR', 1: 'CAUTION', 2: 'STOP'}
LEVEL_COL = {0: '#16c784', 1: '#ffb020', 2: '#ff2d55'}


def _setup_fonts():
    for fname in ('segoeui.ttf', 'seguisb.ttf', 'segoeuib.ttf', 'segoeuil.ttf'):
        path = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', fname)
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ('Segoe UI', 'Montserrat', 'DejaVu Sans'):
        if fam in names:
            plt.rcParams['font.family'] = fam
            return fam
    return None


FONT = _setup_fonts()


VIEW_W, VIEW_H = 1120, 600
_TURBO = (matplotlib.colormaps['turbo_r'](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def _project(p, f=1050.0, cx=VIEW_W / 2, cy=300.0):
    """Pinhole projection of forward-frame points (X fwd, Y left, Z up) into the view image."""
    x = np.maximum(p[:, 0], 0.1)
    return cx - f * p[:, 1] / x, cy - f * p[:, 2] / x


def _splat(P, f=1050.0, cx=VIEW_W / 2, cy=300.0, w=VIEW_W, h=VIEW_H, scale=1.0, hi=(), min_size=1):
    """Range-coloured point splatting (far to near, near points drawn larger); hi = [(indices, rgb)] drawn on top."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (5, 6, 11)
    m = (P[:, 0] > 2.0) & (P[:, 0] < 260)
    Q = P[m]
    u, v = _project(Q, f, cx, cy)
    r = Q[:, 0]
    order = np.argsort(-r)
    u, v, r = u[order], v[order], r[order]
    ci = np.clip((np.log(r) - np.log(3)) / (np.log(220) - np.log(3)) * 255, 0, 255).astype(np.int64)
    col = _TURBO[ci]
    size = np.maximum(np.where(r < 12 / scale, 3, np.where(r < 45 / scale, 2, 1)), min_size)
    layers = [(u[size == s_], v[size == s_], col[size == s_], s_) for s_ in sorted(set(size.tolist()))]
    for idx, rgb in hi:
        hu, hv = _project(P[idx], f, cx, cy)
        layers.append((hu, hv, np.tile(np.array(rgb, np.uint8), (len(hu), 1)), max(min_size, 2) + 1))
    for lu, lv, cc, s_ in layers:
        uu = np.round(lu).astype(np.int64); vv = np.round(lv).astype(np.int64)
        for du in range(s_):
            for dv in range(s_):
                a = uu + du - s_ // 2; b = vv + dv - s_ // 2
                ok = (a >= 0) & (a < w) & (b >= 0) & (b < h)
                img[b[ok], a[ok]] = cc[ok]
    return img


def _rgb(hexcol):
    return tuple(int(hexcol[i:i + 2], 16) for i in (1, 3, 5))


def _fig():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    return fig


def _to_img(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def _card(ax, x, y, w, h, color=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=14', fc=color, ec='none', zorder=0))


def _label(ax, x, y, text, size=11, color=MUTED, **kw):
    ax.text(x, y, text, color=color, fontsize=size, zorder=5,
            bbox=dict(boxstyle='round,pad=0.35', fc='#05060bcc', ec='none'), **kw)


def render_frame(res, cfg_gauge, info=None):
    info = info or {}
    P = res.forward_points
    g = res.geometry
    lev = int(res.level)
    col = LEVEL_COL[lev]
    fig = _fig()
    base = fig.add_axes([0, 0, 1, 1], zorder=-2)
    base.set_xlim(0, W); base.set_ylim(H, 0); base.axis('off')

    # ---------------------------------------------------------------- banner
    base.add_patch(Rectangle((0, 0), W, 78, color=col))
    base.text(28, 39, LEVEL_TXT[lev], color='white', fontsize=26, weight='bold', va='center')
    if not math.isnan(res.nearest_distance):
        base.text(560, 39, f'{res.nearest_distance:.1f} м', color='white', fontsize=30, weight='bold', va='center')
    base.text(W - 28, 28, 'TunnelGuard · ROS 2 Humble', color='white', fontsize=12, va='center', ha='right', alpha=0.9)
    base.text(W - 28, 54, info.get('clock', ''), color='white', fontsize=12, va='center', ha='right', alpha=0.9)

    # ---------------------------------------------------------------- perspective view
    ax = fig.add_axes([16 / W, 1 - (96 + VIEW_H) / H, VIEW_W / W, VIEW_H / H])
    hi = [(o['idx'], _rgb(LEVEL_COL[2] if o['zone'] == 2 else LEVEL_COL[1])) for o in res.obstacles]
    ax.imshow(_splat(P, hi=hi), extent=(0, VIEW_W, VIEW_H, 0), interpolation='nearest')
    if g is not None and g.ok:
        top = min(res.clear_distance, 200.0)
        for d in [20, 35, 55, 80, 110, 150, 200]:
            if d > top:
                break
            sl, sv = g.sigma_at(np.array([float(d)]))
            poly = envelope_polygon(cfg_gauge, d, float(sl[0]), float(sv[0]))
            pts = np.stack([np.full(len(poly), d), g.centre(d) + poly[:, 0],
                            g.rail_z(d) + poly[:, 1] + g.roll * poly[:, 0]], 1)
            pu, pv = _project(pts)
            ax.plot(np.r_[pu, pu[0]], np.r_[pv, pv[0]], '-', color=col, lw=1.8 if d < 60 else 1.1, alpha=0.9)
            if d <= 80:
                ax.text(pu.max() + 6, pv.max(), f'{d} м', color=INK, fontsize=9, alpha=0.8)
    target = None
    for o in res.obstacles:
        q = P[o['idx']]
        qu, qv = _project(q)
        oc = LEVEL_COL[2] if o['zone'] == 2 else LEVEL_COL[1]
        pad = 8
        ax.add_patch(Rectangle((qu.min() - pad, qv.min() - pad), np.ptp(qu) + 2 * pad, np.ptp(qv) + 2 * pad,
                               fill=False, ec=oc, lw=2.2, ls='--' if o.get('ego_veto') else '-'))
        if o.get('ego_veto'):
            _label(ax, qu.max() + pad + 6, qv.min(), 'держит дистанцию при движении\n→ артефакт, не препятствие',
                   size=10, color=LEVEL_COL[1], va='top')
        if o['zone'] == 2 and (target is None or o['distance'] < target['distance']):
            target = o
    ax.set_xlim(0, VIEW_W); ax.set_ylim(VIEW_H, 0); ax.axis('off')
    _label(ax, 12, 24, 'Облако точек 3D-лидара (цвет — дальность) · габарит поезда вдоль восстановленного пути', size=11)
    if info.get('badge'):
        text_, bcol = info['badge']
        ax.text(12, 60, text_, color='white', fontsize=12, weight='bold', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.45', fc=bcol, ec='none'))
    if info.get('model'):
        ax.text(12, 96, 'Модель: ' + info['model'], color=INK, fontsize=11, va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.4', fc='#05060bdd', ec='none'))
    # colour = range legend (the dense green area in the distance is the far part of the real tunnel)
    ax.add_patch(FancyBboxPatch((6, 114), 104, 228, boxstyle='round,pad=0,rounding_size=8', fc='#05060bdd', ec='none',
                                zorder=5))
    ramp = np.exp(np.linspace(np.log(3), np.log(220), 200))
    ci = np.clip((np.log(ramp) - np.log(3)) / (np.log(220) - np.log(3)) * 255, 0, 255).astype(int)
    ax.imshow(_TURBO[ci][::-1][:, None, :].repeat(10, 1), extent=(14, 24, 330, 140), zorder=6, interpolation='bilinear')
    for d in (3, 20, 60, 200):
        yy = 330 - (np.log(d) - np.log(3)) / (np.log(220) - np.log(3)) * 190
        ax.text(28, yy, f'{d} м', color=INK, fontsize=9, va='center', zorder=6)
    ax.text(12, 124, 'цвет = дальность', color=MUTED, fontsize=9, va='center', zorder=6)
    if info.get('syn_xyz') is not None:
        su, sv = _project(np.asarray(info['syn_xyz'], float)[None, :])
        su, sv = float(su[0]), float(sv[0])
        if 0 < su < VIEW_W and 0 < sv < VIEW_H:
            ax.plot(su, sv, 'o', ms=26, mfc='none', mec='#ffb347', mew=2, zorder=7)
            ax.annotate(info.get('syn_label', 'синтетический человек'), xy=(su - 10, sv + 10),
                        xytext=(su - 250, sv + 150), color='#ffb347', fontsize=11.5, weight='bold', zorder=7,
                        arrowprops=dict(arrowstyle='->', color='#ffb347', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.35', fc='#05060bdd', ec='none'))
    if info.get('note'):
        ax.text(VIEW_W / 2, VIEW_H - 22, info['note'], color='white', fontsize=13, weight='bold', va='center',
                ha='center', zorder=6, bbox=dict(boxstyle='round,pad=0.55', fc=info.get('note_color', '#05060bdd'),
                                                 ec='none'))
    if target is not None:
        # 4x zoom around the obstacle
        q = P[target['idx']]
        cxz, cyz = [float(np.mean(a)) for a in _project(q)]
        zw, zh = 300, 190
        oc = LEVEL_COL[2]
        zoom = _splat(P, f=1050.0 * 4, cx=VIEW_W / 2 + (VIEW_W / 2 - cxz) * 4 - (VIEW_W / 2 - zw / 2),
                      cy=300.0 + (300.0 - cyz) * 4 - (300.0 - zh / 2), w=zw, h=zh, scale=4.0, hi=hi, min_size=3)
        zx = fig.add_axes([(16 + VIEW_W - zw - 14) / W, 1 - (96 + 14 + zh) / H, zw / W, zh / H])
        zx.imshow(zoom, extent=(0, zw, zh, 0), interpolation='nearest')
        qu, qv = _project(q, 1050.0 * 4, VIEW_W / 2 + (VIEW_W / 2 - cxz) * 4 - (VIEW_W / 2 - zw / 2),
                          300.0 + (300.0 - cyz) * 4 - (300.0 - zh / 2))
        zx.add_patch(Rectangle((qu.min() - 6, qv.min() - 6), np.ptp(qu) + 12, np.ptp(qv) + 12, fill=False, ec=oc, lw=2))
        zx.set_xlim(0, zw); zx.set_ylim(zh, 0); zx.set_xticks([]); zx.set_yticks([])
        for sp in zx.spines.values():
            sp.set_color(oc); sp.set_linewidth(2.5)
        ttc = '' if not math.isfinite(target['ttc']) else f' · {target["ttc"]:.1f} с'
        _label(zx, 8, 20, f'×4  {target["distance"]:.1f} м{ttc}', size=12, color=oc, weight='bold')
        ax.plot([cxz, VIEW_W - zw / 2 - 14], [cyz, 14 + zh], color=oc, lw=1, alpha=0.7)

    # ---------------------------------------------------------------- metrics panel
    px, py, pw, ph = 1152, 96, 432, 600
    _card(base, px, py, pw, ph)
    base.text(px + 22, py + 34, 'Результат алгоритма', color=INK, fontsize=15, weight='bold', va='center')
    ob = min((o for o in res.obstacles if o['zone'] == 2), key=lambda o: o['distance'], default=None)
    radius = abs(1.0 / res.curvature) if abs(res.curvature) > 1e-4 else float('inf')
    rows = [
        ('Препятствие', f'{ob["distance"]:.1f} м' if ob else '—', col if ob else INK),
        ('Время до столкновения', f'{ob["ttc"]:.1f} с' if ob and math.isfinite(ob['ttc']) else '—', INK),
        ('Смещение от оси / высота', f'{ob["lateral"]:+.2f} / {ob["height"]:.2f} м' if ob else '—', INK),
        ('Уверенность модели', f'{ob["confidence"]:.2f}' if ob else '—', INK),
        ('Путь проверен до', f'{res.clear_distance:.0f} м', INK),
        ('Радиус кривой', '∞ (прямая)' if not math.isfinite(radius) or radius > 5000 else f'{radius:.0f} м', INK),
        ('Скорость поезда (лидарная одометрия)', f'{info["speed"]:.0f} км/ч' if info['speed'] is not None else '—', INK) if 'speed' in info else
        ('Точек в кадре', f'{info.get("points", len(P)):,}'.replace(',', ' '), INK),
        ('Задержка обработки', f'{1e3 * res.timings["total"]:.0f} мс', INK),
    ]
    for i, (k, val, c) in enumerate(rows):
        yy = py + 88 + i * 62
        base.text(px + 22, yy, k, color=MUTED, fontsize=11.5, va='center')
        base.text(px + pw - 22, yy + 22, val, color=c, fontsize=19, weight='bold', va='center', ha='right')
        base.plot([px + 22, px + pw - 22], [yy + 44, yy + 44], color='#23263a', lw=1)

    # ---------------------------------------------------------------- bird's-eye strip
    bx = fig.add_axes([56 / W, 1 - (712 + 110) / H, 1528 / W, 110 / H])
    bx.set_facecolor('#05060b')
    mb = (P[:, 0] < 230) & (np.abs(P[:, 1]) < 12)
    bx.scatter(P[mb, 0], P[mb, 1], s=0.15, c='#7a82a6', linewidths=0, rasterized=True)
    if g is not None and g.ok:
        xs = np.linspace(0, max(min(res.clear_distance, 230), 1.0), 200)
        c0 = g.centre(xs)
        bx.fill_between(xs, c0 - 1.35, c0 + 1.35, color=col, alpha=0.18, lw=0)
        bx.plot(xs, c0 - 1.35, color=col, lw=1); bx.plot(xs, c0 + 1.35, color=col, lw=1)
    for o in res.obstacles:
        q = P[o['idx']]
        bx.scatter(q[:, 0], q[:, 1], s=9, c=LEVEL_COL[2] if o['zone'] == 2 else LEVEL_COL[1], linewidths=0)
    bx.set_xlim(0, 230); bx.set_ylim(-12, 12)
    bx.tick_params(colors=MUTED, labelsize=9, length=3)
    bx.set_yticks([])
    bx.set_xticks(range(0, 231, 25))
    for sp in bx.spines.values():
        sp.set_color('#23263a')
    base.text(30, 712 + 55, 'вид\nсверху', color=MUTED, fontsize=10, va='center', ha='center', rotation=90)
    bx.text(228, 10.5, 'м вдоль пути', color=MUTED, fontsize=9, ha='right', va='top')

    # ---------------------------------------------------------------- caption strip
    base.add_patch(Rectangle((0, 846), W, 54, color='#0f1120'))
    base.add_patch(Rectangle((0, 846), 8, 54, color=PURPLE))
    base.text(28, 873, info.get('chapter', ''), color=INK, fontsize=15, weight='bold', va='center')
    base.text(W - 28, 873, info.get('subtitle', ''), color=MUTED, fontsize=12.5, va='center', ha='right')
    return _to_img(fig)


def title_card(lines, sub=None, accent=PURPLE, footer=None):
    """lines: list of (text, size, color, weight)."""
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    ax.add_patch(Rectangle((0, 0), W, H, color=BG))
    ax.add_patch(Polygon([[0, 0], [W * 0.55, 0], [W * 0.35, H], [0, H]], closed=True, color='#12132a'))
    ax.add_patch(Rectangle((110, 250), 10, 400, color=accent))
    y = 290
    for text, size, color, weight in lines:
        ax.text(150, y, text, color=color, fontsize=size, weight=weight, va='top')
        y += size * 2.05
    if footer:
        ax.text(150, 800, footer, color=MUTED, fontsize=13, va='center')
    return _to_img(fig)


def metrics_card(title, items, footer=None):
    """items: list of (big value, label) shown as a grid of tiles."""
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    ax.add_patch(Rectangle((0, 0), W, H, color=BG))
    ax.add_patch(Rectangle((110, 90), 10, 70, color=PURPLE))
    ax.text(150, 125, title, color=INK, fontsize=34, weight='bold', va='center')
    ncol = 3
    tw, th, gx, gy = 440, 250, 30, 30
    for i, (val, label) in enumerate(items):
        cx = 110 + (i % ncol) * (tw + gx)
        cy = 210 + (i // ncol) * (th + gy)
        ax.add_patch(FancyBboxPatch((cx, cy), tw, th, boxstyle='round,pad=0,rounding_size=18', fc=PANEL, ec='none'))
        ax.text(cx + 30, cy + 95, val, color='#ff2d80' if i == 0 else INK, fontsize=44, weight='bold', va='center')
        ax.text(cx + 30, cy + 185, label, color=MUTED, fontsize=15, va='center', wrap=True)
    if footer:
        ax.text(110, 830, footer, color=MUTED, fontsize=13, va='center')
    return _to_img(fig)


def chart_card(title, png_path, caption=None):
    """A results chart (white-background PNG) on the dark video background."""
    import matplotlib.image as mpimg
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    ax.add_patch(Rectangle((0, 0), W, H, color=BG))
    ax.add_patch(Rectangle((110, 60), 10, 70, color=PURPLE))
    ax.text(150, 95, title, color=INK, fontsize=34, weight='bold', va='center')
    img = mpimg.imread(png_path)
    ih, iw = img.shape[:2]
    bw, bh = 1180, 640
    sc = min(bw / iw, bh / ih)
    w, h = iw * sc, ih * sc
    x0, y0 = (W - w) / 2, 165 + (bh - h) / 2
    ax.add_patch(FancyBboxPatch((x0 - 24, y0 - 20), w + 48, h + 40, boxstyle='round,pad=0,rounding_size=18',
                                fc='white', ec='none'))
    ax.imshow(img, extent=(x0, x0 + w, y0 + h, y0), zorder=3)
    if caption:
        ax.text(W / 2, 845, caption, color=MUTED, fontsize=14, va='center', ha='center')
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    return _to_img(fig)
