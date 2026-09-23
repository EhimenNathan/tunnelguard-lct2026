"""Summarise stream_eval.py output: decisions, STOP events (with a contact sheet of their snapshots), clear distance,
latency.   usage (scratchpad):  python summarize_stream_eval.py ds2eval"""
import collections
import glob
import json
import os
import sys

import numpy as np

out = sys.argv[1]
L = [json.loads(l) for l in open(os.path.join(out, 'frames.jsonl'))]
lev = collections.Counter(x['level'] for x in L)
ms = np.array([x['ms'] for x in L])
clear = np.array([x['clear'] for x in L])
stop_frames = [x for x in L if x['level'] == 2]
events = {}
for x in stop_frames:
    for o in x['obs']:
        if o['zone'] == 2:
            key = (x['piece'], o['id'])
            e = events.setdefault(key, dict(piece=x['piece'], id=o['id'], frames=0, d=[], lat=[], h=[], n=[], conf=[], k0=x['k']))
            e['frames'] += 1
            for f in ('d', 'lat', 'h', 'n', 'conf'):
                e[f].append(o[f])
ev = sorted(events.values(), key=lambda e: (e['piece'], e['k0']))
summary = dict(frames=len(L), pieces=len({x['piece'] for x in L}),
               levels={'CLEAR': lev.get(0, 0), 'CAUTION': lev.get(1, 0), 'STOP': lev.get(2, 0)},
               stop_rate_pct=round(100 * lev.get(2, 0) / max(len(L), 1), 3),
               stop_events=len(ev),
               latency_ms=dict(median=float(np.median(ms)), p95=float(np.percentile(ms, 95)), max=float(ms.max())),
               clear_m=dict(median=float(np.median(clear)), p10=float(np.percentile(clear, 10)), p90=float(np.percentile(clear, 90))),
               events=[dict(piece=e['piece'], id=e['id'], first_k=e['k0'], frames=e['frames'],
                            dist=[round(min(e['d']), 1), round(max(e['d']), 1)], lateral=round(float(np.median(e['lat'])), 2),
                            height=round(float(np.median(e['h'])), 2), points=int(np.median(e['n'])),
                            confidence=round(float(np.median(e['conf'])), 2)) for e in ev])
json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in summary.items() if k != 'events'}, indent=1))
for e in summary['events']:
    print(e)
snaps = sorted(glob.glob(os.path.join(out, 'snaps', '*.png')))
if snaps:
    from PIL import Image
    ims = [Image.open(p).convert('RGB') for p in snaps[:24]]
    w, h = 800, 450
    cols = 3
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * w, rows * h), 'white')
    for i, im in enumerate(ims):
        sheet.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    sheet.save(os.path.join(out, 'stop_events_sheet.png'))
    print('contact sheet:', len(ims), 'of', len(snaps), 'snapshots')

# ---------------------------------------------------------------- evidence classes and timeline
def piece_no(p):
    return int(p.split('_')[-1])


for e in summary['events']:
    e['t_s'] = round(piece_no(e['piece']) * 5.0 + e['first_k'] * 0.1, 1)
    strong = e['points'] >= 20 and e['height'] >= 0.5 and e['frames'] >= 2
    weak = e['points'] <= 10
    e['evidence'] = 'strong' if strong else ('weak' if weak else 'medium')
cls = collections.Counter(e['evidence'] for e in summary['events'])
summary['event_classes'] = dict(cls)
json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
print('evidence classes:', dict(cls))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
t = np.array([piece_no(x['piece']) * 5.0 + x['k'] * 0.1 for x in L])
lv = np.array([x['level'] for x in L])
fig, ax = plt.subplots(figsize=(13, 3.2))
for level, col, yv, lab in [(1, '#FFB020', 1, 'CAUTION'), (2, '#FF0053', 2, 'STOP')]:
    m = lv == level
    ax.scatter(t[m] / 60, np.full(m.sum(), yv), s=6, color=col, marker='|', label=f'{lab}: {m.sum()} frames')
for e in summary['events']:
    c = {'strong': '#520977', 'medium': '#8A83D1', 'weak': '#C9C3D6'}[e['evidence']]
    ax.annotate(f"{e['dist'][1]:.0f} m", (e['t_s'] / 60, 2.25), color=c, fontsize=7, rotation=90, ha='center')
ax.set_yticks([1, 2]); ax.set_yticklabels(['CAUTION', 'STOP'])
ax.set_ylim(0.5, 3.0); ax.set_xlim(0, 20.2); ax.set_xlabel('time in the recording, min')
ax.set_title('New 20-minute recording: decisions over time (labels: first STOP distance; dark = strong evidence)', loc='left', fontsize=11)
ax.legend(loc='upper right', fontsize=9, frameon=False)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(out, 'timeline.png'), dpi=160); plt.close()
print('timeline written')
