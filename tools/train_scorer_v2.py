"""Scorer v2: long-range training data (gen_dataset_v2.py) and a range-tiered decision policy, validated
leave-one-recording-out (LORO).

Decision policies simulated offline from saved candidates (thresholds chosen on the training recordings only):
  measured   : STOP only within the range this frame measured the track (v1 behaviour)
  extended   : within the measured range as above; between the measured range and the line of sight a STOP needs a
               stricter score (tau_far) and more confirmation hits (k_far) - evidence must grow with extrapolation
Recall is reported per 50 m band of the object's first position, for objects with >= 3 lidar returns at any frame.
usage (scratchpad):  python train_scorer_v2.py ds2_v2all.npz
"""
import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '.')
DS = sys.argv[1] if len(sys.argv) > 1 else 'ds2_v2all.npz'
d = np.load(DS, allow_pickle=True)
X = np.nan_to_num(d['X'].astype(np.float32), nan=0.0, posinf=1e3, neginf=-1e3)
M = d['meta']
OBJ = d['objs']
NAMES = list(d['names'])
bag, pas, seq, frame, ftid, y, objid, rule_kept, rule_in, in_raw, beyond = [M[:, i] for i in range(11)]
print('rows', len(X), 'positives', int(y.sum()), 'beyond measured', int(beyond.sum()), 'pos beyond', int((y * beyond).sum()))
N_FRAMES = {0: 252, 1: 877, 2: 345, 3: 545, 4: 268}
BAGS = np.unique(bag)
hits = X[:, NAMES.index('hits')]
s_feat = X[:, NAMES.index('s')]

SIGN = {'n_norm': 1, 'h_ext_beams': 1, 'in_frac': 1, 'depth_rel': 1, 'contained': 1, 'hits': 1,
        'shell_pts': -1, 'gravity_fail': -1, 'shape_fail': -1, 'sigma_l': -1, 'l_std': -1}
TUNED = json.load(open('tuned_params.json'))
DIST_FEATS = ['s', 'log_s', 'valid_margin', 'rail_margin', 'trusted_margin', 'sight_margin']


def fit_lgb_mono(Xtr, ytr, names):
    import lightgbm as lgb
    p = dict(objective='binary', verbose=-1, n_jobs=2, **TUNED['lgb_mono'])
    p['monotone_constraints'] = [SIGN.get(n, 0) for n in names]
    p['monotone_constraints_method'] = 'advanced'
    m = lgb.LGBMClassifier(**p).fit(Xtr, ytr)
    return lambda Z: m.predict_proba(Z)[:, 1]


def fit_cat(Xtr, ytr, names):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(verbose=0, thread_count=2, **TUNED['cat']).fit(Xtr, ytr)
    return lambda Z: m.predict_proba(Z)[:, 1]


def loro(fit, cols):
    names = [NAMES[i] for i in cols]
    oof = np.zeros(len(X))
    for b in BAGS:
        tr, te = bag != b, bag == b
        oof[te] = fit(X[tr][:, cols], y[tr], names)(X[te][:, cols])
    return oof


def lg(v):
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))


def sig(z):
    return 1 / (1 + np.exp(-z))


order = np.lexsort((frame, ftid, seq, pas, bag))


def smooth(p, k=5):
    o = order
    key = np.stack([bag[o], pas[o], seq[o], ftid[o]], 1)
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


def stop_mask(policy, p, tau, tau_far=None, k_far=None):
    base = (in_raw >= 2) & (hits >= 3)
    rule = base & (rule_kept == 1) & (rule_in == 1)
    near = (base & (p >= tau) & (beyond == 0)) | (rule & (s_feat < 30) & (beyond == 0))
    if policy == 'measured':
        return near
    if policy == 'extended':
        return near | ((in_raw >= 2) & (hits >= k_far) & (p >= tau_far) & (beyond == 1))
    if policy == 'extended_sparse':   # track-before-detect lite: sparse far clusters accumulate evidence over frames
        far_sparse = (s_feat >= 120) & (in_raw >= 1) & (hits >= k_far + 1) & (p >= tau_far)
        return near | ((in_raw >= 2) & (hits >= k_far) & (p >= tau_far) & (beyond == 1)) | far_sparse
    raise KeyError(policy)


def visible_objects(sel):
    vis = {}
    for r in OBJ:
        if int(r[0]) in sel:
            vis.setdefault(int(r[4]), []).append((float(r[5]), int(r[7])))
    return {o: v[0][0] for o, v in vis.items() if max(n for _, n in v) >= 3}


def metrics(mask, sel):
    clean = (pas == 0) & np.isin(bag, sel)
    fp = len({(b, f) for b, f in zip(bag[clean & mask], frame[clean & mask])})
    tot = sum(N_FRAMES[int(b)] for b in sel)
    inj = (pas == 1) & np.isin(bag, sel) & mask & (y == 1)
    detected = {}
    for o, s_ in zip(objid[inj], s_feat[inj]):
        detected[int(o)] = max(detected.get(int(o), 0.0), float(s_))
    vis = visible_objects(sel)
    bands = {}
    for o, d0 in vis.items():
        bands.setdefault(min(int(d0 // 50), 4) * 50, []).append(o in detected)
    first = [detected[o] for o in vis if o in detected]
    return dict(fp=fp, frames=tot, recall=float(np.mean([o in detected for o in vis])) if vis else np.nan,
                bands={k: (float(np.mean(v)), len(v)) for k, v in sorted(bands.items())},
                n_vis=len(vis), max_first_stop=float(max(first)) if first else 0.0,
                far_first_stops=sorted(first)[-5:])


def choose(policy, p, train, budget):
    best = None
    taus = np.linspace(0.05, 0.95, 19)
    grid = [(t, None, None) for t in taus] if policy == 'measured' else \
        [(t, tf, k) for t in taus for tf in np.linspace(0.3, 0.95, 14) for k in (3, 4, 5, 6)]
    if policy == 'extended_sparse':
        grid = [(t, tf, k) for t in taus[::2] for tf in np.linspace(0.5, 0.95, 10) for k in (4, 5, 6)]
    for t, tf, k in grid:
        m = metrics(stop_mask(policy, p, t, tf, k), train)
        if m['fp'] / m['frames'] <= budget and (best is None or m['recall'] > best[1] + 1e-9):
            best = ((t, tf, k), m['recall'])
    return best[0] if best else (0.95, 0.95, 6)


def evaluate(policy, p, budget):
    fp = frames = 0
    per, bands, vis_n, det_n, firsts = {}, {}, 0, 0, []
    for b in BAGS:
        train = [int(x) for x in BAGS if x != b]
        t, tf, k = choose(policy, p, train, budget)
        m = metrics(stop_mask(policy, p, t, tf, k), [int(b)])
        per[int(b)] = dict(tau=round(float(t), 2), tau_far=None if tf is None else round(float(tf), 2), k_far=k, fp=m['fp'],
                           recall=round(m['recall'], 3))
        fp += m['fp']; frames += m['frames']
        vis_n += m['n_vis']; det_n += m['recall'] * m['n_vis']
        firsts += m['far_first_stops']
        for kk, (v, n) in m['bands'].items():
            bands.setdefault(kk, [0.0, 0])
            bands[kk][0] += v * n; bands[kk][1] += n
    return dict(policy=policy, fp=fp, fp_rate=round(fp / frames, 4), recall=round(det_n / max(vis_n, 1), 3),
                bands={k: (round(v / max(n, 1), 3), n) for k, (v, n) in sorted(bands.items())},
                farthest_first_stops=sorted(firsts)[-5:], per_bag=per)


if __name__ == '__main__':
    t0 = time.time()
    all_cols = list(range(len(NAMES)))
    inv_cols = [i for i, n in enumerate(NAMES) if n not in DIST_FEATS]
    results = {}
    for fs_name, cols in [('all_features', all_cols), ('distance_invariant', inv_cols)]:
        o_lgb = loro(fit_lgb_mono, cols)
        o_cat = loro(fit_cat, cols)
        blend = sig((lg(o_lgb) + lg(o_cat)) / 2)
        far = s_feat >= 150
        print(f'{fs_name}: AP {average_precision_score(y, blend):.4f} AUC {roc_auc_score(y, blend):.4f} '
              f'| far(>=150 m) AP {average_precision_score(y[far], blend[far]):.4f} [{time.time() - t0:.0f}s]', flush=True)
        ps = smooth(blend, 5)
        np.save(f'oof2_{fs_name}.npy', ps)
        for budget in (0.003, 0.005):
            for policy in ('measured', 'extended', 'extended_sparse'):
                r = evaluate(policy, ps, budget)
                results[f'{fs_name}|{policy}|{budget}'] = r
                print(f'  budget {budget} {policy:9s} fp {r["fp"]:3d} ({100 * r["fp_rate"]:.2f}%) recall {r["recall"]:.3f} '
                      f'bands {r["bands"]} farthest {r["farthest_first_stops"]}', flush=True)
    json.dump(results, open('train_v2_results.json', 'w'), indent=1, default=float)
    print('done', time.time() - t0)
