"""Train the domain-robust scorer (monotone LightGBM + CatBoost, mean of logits) on the original recordings and the
first three blocks of the new drive, choose the threshold on out-of-fold scores, export to JSON and check parity.
The sealed final-test block (group 13) is not used.   usage (scratchpad):  python final_scorer_v3.py [budget]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_scorer_v3 as T
from export_scorer import lgb_member, cat_member

sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard'))
from tunnel_guard.core.scorer import ObstacleScorer

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.003
SUF = os.environ.get('DS_SUFFIX', '')
OUT = os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard', 'config',
                   'obstacle_scorer_v4.json' if SUF == '4' else 'obstacle_scorer_v3.json')
CV = [0, 1, 2, 3, 4, 10, 11, 12]
train = np.isin(T.grp, CV)

oof = np.load(f'oof_v3{SUF}.npz')['rob']
ps = T.smooth(oof)
tau = T.choose_tau(ps, CV, BUDGET)
m = T.metrics(T.stop_mask(ps, tau), CV)
print('threshold %.3f  OOF decision on all CV groups: false STOP %d/%d (%.2f %%), recall %.1f %%' %
      (tau, m['fp'], m['frames'], 100 * m['fp_rate'], 100 * m['recall']))

m_l, _ = T.fit_lgb_mono(T.X[train], T.y[train])
m_c, _ = T.fit_cat(T.X[train], T.y[train])
doc = dict(features=T.NAMES, members=[lgb_member(m_l), cat_member(m_c, 'tmp_cat_v3.json')], threshold=float(tau),
           smooth_k=5, near_override_s=30.0,
           info=dict(models='monotone LightGBM + CatBoost (mean of logits), domain-robust' + (', hazard-space positives' if SUF == '4' else ''),
                     trained_on='5 original obstacle-free recordings + new 20-min drive t < 900 s (real negatives verified by '
                                'traversal) + ray-cast obstacles in real beams of both',
                     sealed_test='new drive t >= 900 s', fp_budget=BUDGET,
                     oof_decision=dict(fp=m['fp'], frames=m['frames'], recall=m['recall'])))
json.dump(doc, open(OUT, 'w'))


def lg(v):
    v = np.clip(v, 1e-9, 1 - 1e-9)
    return np.log(v / (1 - v))


ref = (lg(m_l.predict_proba(T.X)[:, 1]) + lg(m_c.predict_proba(T.X)[:, 1])) / 2
sc = ObstacleScorer(OUT)
print('parity max |logit diff|: %.2e   model size %d KB   -> %s' % (float(np.abs(sc.logit(T.X) - ref).max()),
                                                                 len(json.dumps(doc)) // 1024, OUT))
