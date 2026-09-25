"""Per-obstacle report for the organisers' synthetic-obstacle bag: for each approach window (a static object closing in),
the first distance of CAUTION / STOP on it and how stable STOP is once issued.  Windows are defined by time on the
readable part of the recording.   usage (scratchpad):  python ds3_report.py run_dir [run_dir ...]"""
import json
import sys

# (label, t_from, t_to, max |lateral| of the object track [m], should STOP?)  - times of the readable 438 frames
WINDOWS = [('1 · 2×2 м в центре габарита', 0.0, 23.5, 1.5, True),
           ('2 · 0.3 м в центре габарита', 32.0, 37.0, 1.5, True),
           ('3 · 0.3 м на рельсе', 40.5, 43.0, 1.5, True),
           ('9? · 2×0.2 м лежит на рельсах', 70.0, 74.5, 1.5, True)]


def report(run):
    L = [json.loads(l) for l in open(f'{run}/frames.jsonl')]
    t0 = L[0]['t']
    out = []
    for name, a, b, lmax, should in WINDOWS:
        fr = [f for f in L if a <= f['t'] - t0 <= b]
        first_c = first_s = None
        stop_frames = 0
        hold = []
        for f in fr:
            on = [o for o in f['obs'] if abs(o['l']) < lmax and o['s'] < 200]
            if on and first_c is None:
                first_c = min(o['s'] for o in on)
            st = [o for o in on if o['zone'] == 2]
            if st:
                stop_frames += 1
                if first_s is None:
                    first_s = min(o['s'] for o in st)
            if first_s is not None:
                hold.append(1 if st else 0)
        out.append((name, first_c, first_s, stop_frames, len(fr), sum(hold) / len(hold) if hold else 0))
    return out


for run in sys.argv[1:]:
    print('==', run)
    for name, fc, fs, sf, n, hold in report(run):
        print(f"  {name:32s} first CAUTION/STOP at {fc if fc is not None else '—':>6} m   first STOP at "
              f"{fs if fs is not None else '—':>6} m   STOP frames {sf:3d}/{n:3d}   STOP held after first: {100 * hold:5.1f} %")
