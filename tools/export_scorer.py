"""Convert trained models into the dependency-free JSON format of tunnel_guard.core.scorer."""
import json
import numpy as np


def lgb_member(model, weight=1.0):
    booster = model.booster_ if hasattr(model, 'booster_') else model
    dump = booster.dump_model()
    trees = []
    for t in dump['tree_info']:
        feat, thr, left, right, val, dleft = [], [], [], [], [], []

        def add(node):
            i = len(feat)
            feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); val.append(0.0); dleft.append(True)
            if 'leaf_value' in node:
                val[i] = node['leaf_value']
                return i
            assert node['decision_type'] == '<='
            feat[i] = node['split_feature']; thr[i] = node['threshold']; dleft[i] = bool(node['default_left'])
            l = add(node['left_child']); r = add(node['right_child'])
            left[i], right[i] = l, r
            return i
        add(t['tree_structure'])
        trees.append(dict(feature=feat, threshold=thr, left=left, right=right, value=val, default_left=dleft))
    return dict(type='trees', weight=weight, base=0.0, trees=trees)


def xgb_member(model, weight=1.0):
    booster = model.get_booster()
    base = float(json.loads(booster.save_config())['learner']['learner_model_param']['base_score'])
    base_logit = float(np.log(base / (1 - base)))
    trees = []
    for js in booster.get_dump(dump_format='json'):
        root = json.loads(js)
        feat, thr, left, right, val, dleft = [], [], [], [], [], []

        def add(node):
            i = len(feat)
            feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); val.append(0.0); dleft.append(True)
            if 'leaf' in node:
                val[i] = node['leaf']
                return i
            f = int(node['split'][1:]) if isinstance(node['split'], str) and node['split'].startswith('f') else int(node['split'])
            # xgboost: x < thr goes to 'yes'; convert to x <= nextafter(thr, -inf)
            feat[i] = f; thr[i] = float(np.nextafter(np.float32(node['split_condition']), np.float32(-np.inf)))
            children = {c['nodeid']: c for c in node['children']}
            dleft[i] = node['missing'] == node['yes']
            l = add(children[node['yes']]); r = add(children[node['no']])
            left[i], right[i] = l, r
            return i
        add(root)
        trees.append(dict(feature=feat, threshold=thr, left=left, right=right, value=val, default_left=dleft))
    return dict(type='trees', weight=weight, base=base_logit, trees=trees)


def cat_member(model, path_tmp, weight=1.0):
    model.save_model(path_tmp, format='json')
    doc = json.load(open(path_tmp))
    borders = [ff['borders'] for ff in doc['features_info']['float_features']]
    fidx = [ff['feature_index'] for ff in doc['features_info']['float_features']]
    trees = []
    for t in doc['oblivious_trees']:
        feats, bs = [], []
        for sp in t['splits']:
            feats.append(fidx[sp['float_feature_index']])
            bs.append(sp['border'])
        trees.append(dict(features=feats, borders=bs, values=t['leaf_values']))
    sb = doc.get('scale_and_bias', [1.0, [0.0]])
    bias = sb[1][0] if isinstance(sb[1], list) else sb[1]
    return dict(type='oblivious', weight=weight, base=float(bias), scale=float(sb[0]), trees=trees)


def mlp_member(scaler_net, weight=1.0):
    sc, net = scaler_net
    layers = [dict(W=m.weight.detach().numpy().tolist(), b=m.bias.detach().numpy().tolist())
              for m in net if hasattr(m, 'weight')]
    return dict(type='mlp', weight=weight, mean=sc.mean_.tolist(), std=sc.scale_.tolist(), layers=layers)
