"""Dependency-free (numpy) evaluation of the learned obstacle scorer.

The model file (JSON) is produced offline by tools/export_scorer.py from LightGBM / XGBoost / CatBoost / MLP models.
Members are combined by a weighted mean of logits; the decision threshold and temporal smoothing are stored with it.
Format:
  {"features": [...], "members": [member, ...], "threshold": float, "smooth_k": int, "near_override_s": float}
  tree member      {"type": "trees", "weight": w, "base": b, "trees": [{"feature": [...], "threshold": [...],
                     "left": [...], "right": [...], "value": [...], "default_left": [...]}]}   (leaf: feature = -1)
  oblivious member {"type": "oblivious", "weight": w, "base": b, "scale": s,
                     "trees": [{"features": [...], "borders": [...], "values": [...]}]}
  mlp member       {"type": "mlp", "weight": w, "mean": [...], "std": [...], "layers": [{"W": [[...]], "b": [...]}]}
"""
import json

import numpy as np


class _Trees:
    """All trees packed into padded arrays and traversed simultaneously (one numpy step per depth level)."""

    def __init__(self, m):
        self.base = float(m.get('base', 0.0))
        trees = m['trees']
        nt = len(trees)
        mx = max(len(t['feature']) for t in trees)
        self.feat = np.full((nt, mx), -1, np.int64)
        self.thr = np.zeros((nt, mx))
        self.left = np.zeros((nt, mx), np.int64)
        self.right = np.zeros((nt, mx), np.int64)
        self.val = np.zeros((nt, mx))
        self.dleft = np.ones((nt, mx), bool)
        for i, t in enumerate(trees):
            k = len(t['feature'])
            self.feat[i, :k] = t['feature']; self.thr[i, :k] = t['threshold']
            self.left[i, :k] = t['left']; self.right[i, :k] = t['right']
            self.val[i, :k] = t['value']; self.dleft[i, :k] = t['default_left']
        self.depth = self._max_depth(trees)

    @staticmethod
    def _max_depth(trees):
        best = 0
        for t in trees:
            stack = [(0, 0)]
            while stack:
                node, d = stack.pop()
                best = max(best, d)
                if t['feature'][node] >= 0:
                    stack.append((t['left'][node], d + 1)); stack.append((t['right'][node], d + 1))
        return best

    def raw(self, X):
        ns = len(X)
        nt = self.feat.shape[0]
        ti = np.arange(nt)[None, :]
        node = np.zeros((ns, nt), np.int64)
        rows = np.arange(ns)[:, None]
        for _ in range(self.depth):
            f = self.feat[ti, node]
            leaf = f < 0
            x = X[rows, np.maximum(f, 0)]
            go_left = np.where(np.isnan(x), self.dleft[ti, node], x <= self.thr[ti, node])
            node = np.where(leaf, node, np.where(go_left, self.left[ti, node], self.right[ti, node]))
        return self.base + self.val[ti, node].sum(axis=1)


class _Oblivious:
    def __init__(self, m):
        self.base = float(m.get('base', 0.0))
        self.scale = float(m.get('scale', 1.0))
        self.groups = {}
        for t in m['trees']:
            d = len(t['features'])
            self.groups.setdefault(d, []).append(t)
        self.packed = []
        for d, ts in self.groups.items():
            F = np.array([t['features'] for t in ts], np.int64).reshape(len(ts), d)
            B = np.array([t['borders'] for t in ts], np.float64).reshape(len(ts), d)
            V = np.array([t['values'] for t in ts], np.float64)
            self.packed.append((F, B, V))

    def raw(self, X):
        out = np.zeros(len(X))
        for F, B, V in self.packed:
            bits = (X[:, F] > B[None, :, :]).astype(np.int64)             # (samples, trees, depth)
            idx = (bits << np.arange(F.shape[1])[None, None, :]).sum(axis=2)
            out += V[np.arange(V.shape[0])[None, :], idx].sum(axis=1)
        return self.scale * out + self.base


class _MLP:
    def __init__(self, m):
        self.mean = np.asarray(m['mean'], np.float64)
        self.std = np.asarray(m['std'], np.float64)
        self.layers = [(np.asarray(l['W'], np.float64), np.asarray(l['b'], np.float64)) for l in m['layers']]

    def raw(self, X):
        h = (X - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = h * (1.0 / (1.0 + np.exp(-h)))          # SiLU
        return h[:, 0]


class ObstacleScorer:
    def __init__(self, path):
        with open(path) as f:
            doc = json.load(f)
        self.features = doc['features']
        self.threshold = float(doc['threshold'])
        self.smooth_k = int(doc.get('smooth_k', 3))
        self.near_override_s = float(doc.get('near_override_s', 30.0))
        kinds = {'trees': _Trees, 'oblivious': _Oblivious, 'mlp': _MLP}
        self.members = [(float(m.get('weight', 1.0)), kinds[m['type']](m)) for m in doc['members']]
        self.wsum = sum(w for w, _ in self.members)

    def logit(self, X):
        X = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=1e3, neginf=-1e3)
        if len(X) == 0:
            return np.zeros(0)
        return sum(w * m.raw(X) for w, m in self.members) / self.wsum

    def prob(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(self.logit(X), -30, 30)))
