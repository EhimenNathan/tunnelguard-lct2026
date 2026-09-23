"""Train the final physics blend on all obstacle-free recordings, pick the threshold from out-of-fold predictions,
export the dependency-free JSON model into the ROS package and verify numerical parity."""
import os
import json, sys
import numpy as np
import train_scorer as T
from export_scorer import lgb_member, cat_member, mlp_member

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'tunnel_guard'))
from tunnel_guard.core.scorer import ObstacleScorer

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005
SMOOTH_K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'tunnel_guard', 'config', 'obstacle_scorer.json')
params = json.load(open('tuned_params.json'))
oof = np.load('oof_tuned.npz')

# threshold: largest-recall value meeting the false-STOP budget on ALL recordings, from out-of-fold scores
p_s = T.smooth_scores(oof['blend'], k=SMOOTH_K)
tau = T.choose_tau('ml_safe', p_s, list(T.BAGS), BUDGET)
m_all = T.decision_metrics(T.stop_mask('ml_safe', p_s, tau), list(T.BAGS))
print('threshold', round(float(tau), 3), 'OOF decision on all recordings:', m_all)

members, fitted = [], {}
m_l, _ = T.fit_lgb(T.X, T.y, params['lgb_mono'], monotone=True)
m_c, _ = T.fit_cat(T.X, T.y, params['cat'])
m_p, _ = T.fit_mlp(T.X, T.y, params['pimlp'], physics=True)
members = [lgb_member(m_l), cat_member(m_c, 'tmp_cat_final.json'), mlp_member(m_p)]
doc = dict(features=T.NAMES, members=members, threshold=float(tau), smooth_k=SMOOTH_K, near_override_s=30.0,
           info=dict(models='monotone LightGBM + CatBoost + physics-informed MLP (mean of logits)', trained_on='5 obstacle-free recordings + ray-cast obstacles',
                     fp_budget=BUDGET, oof_decision=m_all, params=params))
json.dump(doc, open(OUT, 'w'))

# parity check: exported numpy scorer vs library models (mean of logits)
def lg(v): v = np.clip(v, 1e-9, 1 - 1e-9); return np.log(v / (1 - v))
ref = (lg(m_l.predict_proba(T.X)[:, 1]) + lg(m_c.predict_proba(T.X)[:, 1]) +
       lg(__import__('torch').sigmoid(m_p[1](__import__('torch').tensor(m_p[0].transform(T.X), dtype=__import__('torch').float32)).squeeze(1)).detach().numpy())) / 3
sc = ObstacleScorer(OUT)
print('parity max |logit diff|:', float(np.abs(sc.logit(T.X) - ref).max()), 'model size KB', round(len(json.dumps(doc)) / 1024))
