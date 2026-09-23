"""Optuna tuning (LORO average precision) of the physics blend members, then decision-level re-evaluation."""
import json, time, warnings
import numpy as np
import optuna
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
import train_scorer as T
from sklearn.metrics import average_precision_score

t0 = time.time()
best = {}


def objective_factory(name):
    def obj(trial):
        if name == 'lgb_mono':
            p = dict(n_estimators=trial.suggest_int('n_estimators', 100, 600, step=50),
                     learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                     num_leaves=trial.suggest_int('num_leaves', 4, 31),
                     min_child_samples=trial.suggest_int('min_child_samples', 10, 120),
                     subsample=trial.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
                     colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
                     reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 30, log=True))
        elif name == 'cat':
            p = dict(iterations=trial.suggest_int('iterations', 150, 700, step=50),
                     learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                     depth=trial.suggest_int('depth', 3, 7),
                     l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.5, 30, log=True))
        else:
            p = dict(hidden=trial.suggest_categorical('hidden', [32, 64, 128]),
                     drop=trial.suggest_float('drop', 0.0, 0.3),
                     lr=trial.suggest_float('lr', 5e-4, 5e-3, log=True),
                     wd=trial.suggest_float('wd', 1e-5, 1e-2, log=True),
                     epochs=trial.suggest_int('epochs', 15, 40, step=5),
                     lam=trial.suggest_float('lam', 0.1, 5.0, log=True))
        oof = T.loro_oof(name, p)
        trial.set_user_attr('params', p)
        return average_precision_score(T.y, oof)
    return obj


for name, n_trials in [('lgb_mono', 25), ('cat', 14), ('pimlp', 10)]:
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    st.enqueue_trial({k: v for k, v in T.DEFAULTS[name].items() if k not in ('subsample_freq',)})
    st.optimize(objective_factory(name), n_trials=n_trials)
    best[name] = (st.best_value, st.best_trial.user_attrs['params'])
    print(f'{name}: default AP {st.trials[0].value:.4f} -> best AP {st.best_value:.4f} params {st.best_trial.user_attrs["params"]}  [{time.time() - t0:.0f}s]', flush=True)
json.dump({k: v[1] for k, v in best.items()}, open('tuned_params.json', 'w'), indent=1)

oofs = {k: T.loro_oof(k, best[k][1]) for k in best}
def lg(v): v = np.clip(v, 1e-6, 1 - 1e-6); return np.log(v / (1 - v))
blend = 1 / (1 + np.exp(-(lg(oofs['lgb_mono']) + lg(oofs['cat']) + lg(oofs['pimlp'])) / 3))
np.savez('oof_tuned.npz', blend=blend, **oofs)
print('tuned blend AP', round(average_precision_score(T.y, blend), 4))
for budget in (0.003, 0.005, 0.01):
    for k in [3, 5]:
        p = T.smooth_scores(blend, k=k)
        for policy in ('and', 'ml_safe'):
            r = T.loro_decision(policy, p, budget)
            print(f'tuned blend budget {budget:.3f} smooth {k} {policy:8s} fp {r["fp_frames"]:3d} ({100 * r["fp_rate"]:.2f}%) recall {r["recall"]:.3f} bands {r["bands"]} per_bag {r["per_bag"]}', flush=True)
print('done', time.time() - t0)
