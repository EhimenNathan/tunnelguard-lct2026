"""Train and evaluate the second-stage obstacle scorer with leave-one-recording-out (LORO) cross-validation.

Models: LightGBM (plain and physics-constrained = monotone), XGBoost, CatBoost, MLP, physics-informed MLP (soft
monotonicity penalty on physically-signed features), logistic regression; blends of their out-of-fold predictions.
Decision-level evaluation (what matters operationally), simulated offline from saved candidates:
  * false STOP frames on clean frames of the held-out recording
  * obstacle recall (object ever confirmed STOP) and first-STOP distance on injected sequences of the held-out recording
Thresholds are chosen on the training recordings only (nested), never on the held-out one.
"""
import sys, json, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

STAGE = sys.argv[1] if len(sys.argv) > 1 else 'all'
dsA = np.load('ds_dsA.npz', allow_pickle=True)
dsB = np.load('ds_dsB.npz', allow_pickle=True)
X = np.vstack([dsA['X'], dsB['X']]).astype(np.float32)
M = np.vstack([dsA['meta'], dsB['meta']])
OBJ = np.concatenate([dsA['objs'], dsB['objs']])
NAMES = list(dsA['names'])
bag, pas, seq, frame, ftid, y, objid, rule_kept, rule_in, in_raw = [M[:, i] for i in range(10)]
X = np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
print('rows', len(X), 'positives', int(y.sum()), 'per bag', np.bincount(bag), 'pos per bag', np.bincount(bag, weights=y).astype(int))

# physically signed features: +1 more plausible obstacle when larger, -1 less plausible
SIGN = {'n_norm': 1, 'h_ext_beams': 1, 'in_frac': 1, 'depth_rel': 1, 'contained': 1, 'hits': 1,
        'shell_pts': -1, 'gravity_fail': -1, 'shape_fail': -1, 'sigma_l': -1, 'l_std': -1}
mono = [SIGN.get(n, 0) for n in NAMES]


# ----------------------------------------------------------------------------------------- models
def fit_lgb(Xtr, ytr, params, monotone=False):
    import lightgbm as lgb
    p = dict(objective='binary', verbose=-1, n_jobs=4, **params)
    if monotone:
        p['monotone_constraints'] = mono
        p['monotone_constraints_method'] = 'advanced'
    m = lgb.LGBMClassifier(**p)
    m.fit(Xtr, ytr)
    return m, lambda Z: m.predict_proba(Z)[:, 1]


def fit_xgb(Xtr, ytr, params):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_jobs=4, tree_method='hist', eval_metric='logloss', **params)
    m.fit(Xtr, ytr)
    return m, lambda Z: m.predict_proba(Z)[:, 1]


def fit_cat(Xtr, ytr, params):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(verbose=0, thread_count=4, **params)
    m.fit(Xtr, ytr)
    return m, lambda Z: m.predict_proba(Z)[:, 1]


def fit_lr(Xtr, ytr, params):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=params.get('C', 1.0), max_iter=2000, class_weight='balanced').fit(sc.transform(Xtr), ytr)
    return (sc, m), lambda Z: m.predict_proba(sc.transform(Z))[:, 1]


def fit_mlp(Xtr, ytr, params, physics=False):
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(4)
    sc = StandardScaler().fit(Xtr)
    Xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    h = params.get('hidden', 64)
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], h), torch.nn.SiLU(), torch.nn.Dropout(params.get('drop', 0.1)),
                              torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=params.get('lr', 2e-3), weight_decay=params.get('wd', 1e-3))
    pos_w = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    sign = torch.tensor(mono, dtype=torch.float32)
    n = len(Xt)
    for ep in range(params.get('epochs', 30)):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            xb = Xt[idx].clone().requires_grad_(physics)
            out = net(xb).squeeze(1)
            loss = lossf(out, yt[idx])
            if physics:
                # physics-informed soft constraint: d(logit)/d(feature) must have the physical sign
                g, = torch.autograd.grad(out.sum(), xb, create_graph=True)
                loss = loss + params.get('lam', 1.0) * torch.relu(-g * sign).mean() * len(mono)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    def pred(Z):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(sc.transform(Z), dtype=torch.float32)).squeeze(1)).numpy()
    return (sc, net), pred


MODELS = {
    'lgb': lambda Xt, yt, p: fit_lgb(Xt, yt, p),
    'lgb_mono': lambda Xt, yt, p: fit_lgb(Xt, yt, p, monotone=True),
    'xgb': fit_xgb,
    'cat': fit_cat,
    'lr': fit_lr,
    'mlp': lambda Xt, yt, p: fit_mlp(Xt, yt, p),
    'pimlp': lambda Xt, yt, p: fit_mlp(Xt, yt, p, physics=True),
}
DEFAULTS = {
    'lgb': dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0),
    'lgb_mono': dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0),
    'xgb': dict(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
    'cat': dict(iterations=400, learning_rate=0.05, depth=5, l2_leaf_reg=3.0),
    'lr': dict(C=1.0),
    'mlp': dict(hidden=64, drop=0.1, lr=2e-3, wd=1e-3, epochs=25),
    'pimlp': dict(hidden=64, drop=0.1, lr=2e-3, wd=1e-3, epochs=25, lam=1.0),
}
BAGS = np.unique(bag)


def loro_oof(name, params):
    oof = np.zeros(len(X))
    for b in BAGS:
        tr, te = bag != b, bag == b
        _, pred = MODELS[name](X[tr], y[tr], params)
        oof[te] = pred(X[te])
    return oof


# ------------------------------------------------------------------------------ decision simulation
order = np.lexsort((frame, ftid, seq, pas, bag))


def smooth_scores(p, k=3):
    """per feature-track running mean of the last k scores (temporal post-processing)"""
    ps = p.copy()
    o = order
    key = np.stack([bag[o], pas[o], seq[o], ftid[o]], 1)
    new = np.r_[True, np.any(key[1:] != key[:-1], axis=1)]
    grp = np.cumsum(new) - 1
    pv = p[o]
    out = np.empty_like(pv)
    start = 0
    for g_end in np.r_[np.nonzero(new)[0][1:], len(o)]:
        seg = pv[start:g_end]
        c = np.cumsum(np.r_[0.0, seg])
        idx = np.arange(1, len(seg) + 1)
        lo = np.maximum(idx - k, 0)
        out[start:g_end] = (c[idx] - c[lo]) / (idx - lo)
        start = g_end
    ps[o] = out
    return ps


hits = X[:, NAMES.index('hits')]
s_feat = X[:, NAMES.index('s')]


def stop_mask(policy, p, tau):
    base = (in_raw >= 2) & (hits >= 3)
    rule = base & (rule_kept == 1) & (rule_in == 1)
    if policy == 'rules':
        return rule
    if policy == 'ml':
        return base & (p >= tau)
    if policy == 'ml_safe':          # ML decides; near-field well-supported objects always STOP (safety override)
        return (base & (p >= tau)) | (rule & (s_feat < 30))
    if policy == 'and':               # rules AND ml (fewer false alarms)
        return rule & (p >= tau)
    if policy == 'or':                # rules OR confident ml (recall recovery)
        return rule | (base & (p >= tau))
    raise KeyError(policy)


def decision_metrics(mask, sel_bags):
    clean = (pas == 0) & np.isin(bag, sel_bags)
    frames_clean = {(b, f) for b, f in zip(bag[clean], frame[clean])}
    # clean frames without any candidate are also frames: count all frames of these bags
    n_frames = {0: 252, 1: 877, 2: 345, 3: 545, 4: 268}
    tot = sum(n_frames[b] for b in sel_bags)
    fp_frames = len({(b, f) for b, f in zip(bag[clean & mask], frame[clean & mask])})
    inj = (pas == 1) & np.isin(bag, sel_bags)
    visible = {}
    for r in OBJ:
        if r[0] in sel_bags and r[7] >= 3:
            visible.setdefault(r[4], []).append(float(r[5]))
    detected = {}
    for o_id, s_ in zip(objid[inj & mask & (y == 1)], s_feat[inj & mask & (y == 1)]):
        detected[o_id] = max(detected.get(o_id, 0), s_)
    rec = np.mean([o in detected for o in visible]) if visible else np.nan
    # recall by distance band of the object's first position
    bands = {}
    for o, ds in visible.items():
        b_ = min(int(ds[0] // 50), 3)
        bands.setdefault(b_, []).append(o in detected)
    return dict(fp_frames=fp_frames, frames=tot, fp_rate=fp_frames / tot, recall=rec,
                recall_bands={int(k) * 50: (float(np.mean(v)), len(v)) for k, v in sorted(bands.items())})


def choose_tau(policy, p, train_bags, fp_budget):
    """largest recall threshold whose false-STOP rate on the TRAINING recordings stays within budget"""
    best = None
    for tau in np.linspace(0.05, 0.99, 48):
        m = decision_metrics(stop_mask(policy, p, tau), train_bags)
        if m['fp_rate'] <= fp_budget:
            if best is None or m['recall'] > best[1] + 1e-9:
                best = (tau, m['recall'])
    return best[0] if best else 0.99


def loro_decision(policy, p, fp_budget):
    fp, frames, det_all = 0, 0, []
    per = {}
    for b in BAGS:
        tau = choose_tau(policy, p, [x for x in BAGS if x != b], fp_budget) if policy != 'rules' else 0.5
        m = decision_metrics(stop_mask(policy, p, tau), [b])
        per[int(b)] = (round(float(tau), 2), m['fp_frames'], round(m['recall'], 3))
        fp += m['fp_frames']; frames += m['frames']
    allm = decision_metrics(stop_mask(policy, p, 0.5), list(BAGS)) if policy == 'rules' else None
    # pooled recall with per-fold thresholds
    recs = []
    bands = {}
    for b in BAGS:
        tau = per[int(b)][0]
        m = decision_metrics(stop_mask(policy, p, tau), [b])
        recs.append((m['recall'], sum(1 for r in OBJ if r[0] == b and r[3] == 0 and r[7] >= 3)))
        for k, (v, n) in m['recall_bands'].items():
            bands.setdefault(k, []).append((v, n))
    pooled_rec = sum(r * n for r, n in recs if not np.isnan(r)) / max(sum(n for r, n in recs if not np.isnan(r)), 1)
    pooled_bands = {k: round(sum(v * n for v, n in vs) / max(sum(n for v, n in vs), 1), 3) for k, vs in sorted(bands.items())}
    return dict(fp_frames=fp, fp_rate=round(fp / frames, 4), recall=round(pooled_rec, 3), bands=pooled_bands, per_bag=per)


if __name__ == '__main__':
    t0 = time.time()
    oofs = {}
    for name in ['lr', 'lgb', 'lgb_mono', 'xgb', 'cat', 'mlp', 'pimlp']:
        tt = time.time()
        oofs[name] = loro_oof(name, DEFAULTS[name])
        ap = average_precision_score(y, oofs[name]); auc = roc_auc_score(y, oofs[name])
        print(f'{name:9s} LORO AP {ap:.4f} AUC {auc:.4f}  [{time.time() - tt:.0f}s]', flush=True)
    np.savez('oof_scores.npz', **oofs)
    # blends of out-of-fold predictions (rank average)
    # blends = mean of member logits (exactly what the deployed numpy scorer computes)
    def lg(v): v = np.clip(v, 1e-6, 1 - 1e-6); return np.log(v / (1 - v))
    def sig(z): return 1 / (1 + np.exp(-z))
    blends = {
        'blend_gbdt': sig((lg(oofs['lgb']) + lg(oofs['xgb']) + lg(oofs['cat'])) / 3),
        'blend_all': sig(np.mean([lg(oofs[k]) for k in ['lgb', 'lgb_mono', 'xgb', 'cat', 'mlp', 'pimlp']], axis=0)),
        'blend_phys': sig((lg(oofs['lgb_mono']) + lg(oofs['pimlp']) + lg(oofs['cat'])) / 3),
    }
    for k, v in blends.items():
        print(f'{k:11s} LORO AP {average_precision_score(y, v):.4f} AUC {roc_auc_score(y, v):.4f}')
    oofs.update(blends)
    np.savez('oof_scores.npz', **oofs)
    base = loro_decision('rules', np.zeros(len(X)), 0.0)
    print('\nRULES baseline:', base)
    for budget in (0.005, 0.01):
        for name in ['lgb', 'lgb_mono', 'cat', 'pimlp', 'blend_gbdt', 'blend_phys', 'blend_all']:
            p = smooth_scores(oofs[name])
            for policy in ('ml', 'ml_safe', 'and', 'or'):
                r = loro_decision(policy, p, budget)
                print(f'budget {budget:.3f} {name:11s} {policy:8s} fp {r["fp_frames"]:4d} ({100 * r["fp_rate"]:.2f}%) recall {r["recall"]:.3f} bands {r["bands"]}', flush=True)
    print('total', time.time() - t0)
