"""Domain-robust scorer: add the new line's real negatives (and ray-cast positives on it) to the training data and
validate leave-one-group-out (LOGO) at the decision level.

Groups: 0-4 original recordings (ds_dsA/ds_dsB), 10-12 new-drive blocks (ds3_drive.npz, 300 s each).
Group 13 (the last 300 s of the new drive) is the sealed final test: never used for training, tuning or thresholds here.
Compared, per held-out group (threshold chosen on the training groups only, same false-alarm budget):
  deployed  : the shipped scorer (config/obstacle_scorer.json), threshold 0.37
  original  : lgb_mono + CatBoost blend trained on the original recordings only
  robust    : the same blend trained on original recordings + the other new-drive blocks
usage (scratchpad):  python train_scorer_v3.py [--final]
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'src', 'tunnel_guard'))

N_FRAMES_ORIG = {0: 252, 1: 877, 2: 345, 3: 545, 4: 268}
SIGN = {'n_norm': 1, 'h_ext_beams': 1, 'in_frac': 1, 'depth_rel': 1, 'contained': 1, 'hits': 1,
        'shell_pts': -1, 'gravity_fail': -1, 'shape_fail': -1, 'sigma_l': -1, 'l_std': -1}
TUNED = json.load(open('tuned_params.json'))


def load():
    dsA = np.load('ds_dsA.npz', allow_pickle=True)
    dsB = np.load('ds_dsB.npz', allow_pickle=True)
    ds3 = np.load('ds3_drive.npz', allow_pickle=True)
    names = list(dsA['names'])
    assert names == list(ds3['names'])
    X = np.vstack([dsA['X'], dsB['X'], ds3['X']]).astype(np.float32)
    M1 = np.vstack([dsA['meta'], dsB['meta']])[:, :10].astype(np.int64)
    M3 = ds3['meta'][:, :10].astype(np.int64)
    M = np.vstack([M1, M3])
    objs = list(dsA['objs']) + list(dsB['objs']) + list(ds3['objs'])
    n_frames = dict(N_FRAMES_ORIG)
    for blk, n in enumerate(ds3['n_frames']):
        n_frames[10 + blk] = int(n)
    return np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3), M, objs, names, n_frames


X, M, OBJ, NAMES, N_FRAMES = load()
grp, pas, seq, frame, ftid, y, objid, rule_kept, rule_in, in_raw = [M[:, i] for i in range(10)]
hits = X[:, NAMES.index('hits')]
s_feat = X[:, NAMES.index('s')]
order = np.lexsort((frame, ftid, seq, pas, grp))
print('rows', len(X), 'positives', int(y.sum()), 'per group', {int(g): int((grp == g).sum()) for g in np.unique(grp)}, flush=True)


def fit_lgb_mono(Xtr, ytr):
    import lightgbm as lgb
    p = dict(objective='binary', verbose=-1, n_jobs=2, **TUNED['lgb_mono'])
    p['monotone_constraints'] = [SIGN.get(n, 0) for n in NAMES]
    p['monotone_constraints_method'] = 'advanced'
    m = lgb.LGBMClassifier(**p).fit(Xtr, ytr)
    return m, lambda Z: m.predict_proba(Z)[:, 1]


def fit_cat(Xtr, ytr):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(verbose=0, thread_count=2, **TUNED['cat']).fit(Xtr, ytr)
    return m, lambda Z: m.predict_proba(Z)[:, 1]


def lg(v):
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))


def blend_fit(tr):
    _, p1 = fit_lgb_mono(X[tr], y[tr])
    _, p2 = fit_cat(X[tr], y[tr])
    return lambda Z: 1 / (1 + np.exp(-(lg(p1(Z)) + lg(p2(Z))) / 2))


def smooth(p, k=5):
    o = order
    key = np.stack([grp[o], pas[o], seq[o], ftid[o]], 1)
    new = np.r_[True, np.any(key[1:] != key[:-1], axis=1)]
    pv, out, start = p[o], np.empty(len(o)), 0
    for end in np.r_[np.nonzero(new)[0][1:], len(o)]:
        c = np.cumsum(np.r_[0.0, pv[start:end]])
        idx = np.arange(1, end - start + 1)
        lo = np.maximum(idx - k, 0)
        out[start:end] = (c[idx] - c[lo]) / (idx - lo)
        start = end
    ps = np.empty(len(p))
    ps[o] = out
    return ps


def stop_mask(p, tau):
    base = (in_raw >= 2) & (hits >= 3)
    rule = base & (rule_kept == 1) & (rule_in == 1)
    return (base & (p >= tau)) | (rule & (s_feat < 30))


VIS = {}
for r in OBJ:
    VIS.setdefault((int(r[0]), int(r[4])), []).append((float(r[5]), int(r[7])))


def metrics(mask, groups):
    clean = (pas == 0) & np.isin(grp, groups)
    fp = len({(g, f) for g, f in zip(grp[clean & mask], frame[clean & mask])})
    tot = sum(N_FRAMES[int(g)] for g in groups)
    inj = (pas == 1) & np.isin(grp, groups) & mask & (y == 1)
    detected = {(int(g), int(o)) for g, o in zip(grp[inj], objid[inj])}
    vis = {k: v[0][0] for k, v in VIS.items() if k[0] in groups and max(n for _, n in v) >= 3}
    bands = {}
    for k, d0 in vis.items():
        bands.setdefault(min(int(d0 // 50), 3) * 50, []).append(k in detected)
    rec = float(np.mean([k in detected for k in vis])) if vis else float('nan')
    return dict(fp=fp, frames=tot, fp_rate=fp / tot, recall=rec, n_obj=len(vis),
                bands={b: (round(float(np.mean(v)), 3), len(v)) for b, v in sorted(bands.items())})


def choose_tau(p, groups, budget):
    best = None
    for tau in np.linspace(0.05, 0.95, 37):
        m = metrics(stop_mask(p, tau), groups)
        if m['fp_rate'] <= budget and (best is None or m['recall'] > best[1] + 1e-9):
            best = (tau, m['recall'])
    return best[0] if best else 0.95


def deployed_scores():
    from tunnel_guard.core.scorer import ObstacleScorer
    sc = ObstacleScorer(os.path.join(SOL, 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json'))
    return sc.prob(X), sc.threshold


if __name__ == '__main__':
    t0 = time.time()
    budget = 0.003
    cv_groups = [0, 1, 2, 3, 4, 10, 11, 12]
    p_dep, tau_dep = deployed_scores()
    p_dep = smooth(p_dep)
    res = {'deployed': {}, 'original': {}, 'robust': {}}
    oof_orig = np.zeros(len(X))
    oof_rob = np.zeros(len(X))
    for g in cv_groups:
        te = grp == g
        train_orig = np.isin(grp, [x for x in [0, 1, 2, 3, 4] if x != g])
        train_rob = np.isin(grp, [x for x in cv_groups if x != g])
        oof_orig[te] = blend_fit(train_orig)(X[te])
        oof_rob[te] = blend_fit(train_rob)(X[te])
        print(f'group {g} fitted [{time.time() - t0:.0f}s]', flush=True)
    ps_orig, ps_rob = smooth(oof_orig), smooth(oof_rob)
    for g in cv_groups:
        others_o = [x for x in [0, 1, 2, 3, 4] if x != g]
        others_r = [x for x in cv_groups if x != g]
        res['deployed'][g] = metrics(stop_mask(p_dep, tau_dep), [g])
        res['original'][g] = metrics(stop_mask(ps_orig, choose_tau(ps_orig, others_o, budget)), [g])
        res['robust'][g] = metrics(stop_mask(ps_rob, choose_tau(ps_rob, others_r, budget)), [g])
    for name, r in res.items():
        for part, gs in [('original tunnels', [0, 1, 2, 3, 4]), ('new line blocks 1-3', [10, 11, 12])]:
            fp = sum(r[g]['fp'] for g in gs); fr = sum(r[g]['frames'] for g in gs)
            nobj = sum(r[g]['n_obj'] for g in gs); rec = sum(r[g]['recall'] * r[g]['n_obj'] for g in gs if r[g]['n_obj']) / max(nobj, 1)
            print(f'{name:9s} {part:20s} false STOP frames {fp:4d}/{fr} ({100 * fp / fr:.2f} %)  recall {100 * rec:.1f} % of {nobj}')
    json.dump({k: {str(g): v for g, v in r.items()} for k, r in res.items()}, open('train_v3_results.json', 'w'), indent=1)
    np.savez('oof_v3.npz', orig=oof_orig, rob=oof_rob)
    print('done', time.time() - t0)
