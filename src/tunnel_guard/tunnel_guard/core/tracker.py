"""Multi-frame confirmation of obstacle candidates in track coordinates.

Each track holds along-track distance s, lateral offset l and closing speed v (alpha-beta filter).
A candidate becomes CONFIRMED after `confirm_hits` detections within the last `window` frames; this
suppresses single-frame artefacts (spurious returns, geometry glitches) at a cost of ~0.2-0.3 s latency.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackerConfig:
    window: int = 5
    confirm_hits: int = 3
    max_misses_tentative: int = 2
    max_misses_confirmed: int = 5
    gate_s_base: float = 1.2
    gate_s_rel: float = 0.02
    gate_l: float = 0.8
    max_closing_speed: float = 30.0   # [m/s] bound used to gate new tracks
    alpha: float = 0.5
    beta: float = 0.2


@dataclass
class Track:
    id: int
    s: float
    l: float
    v: float = 0.0
    hits: list = field(default_factory=list)
    misses: int = 0
    age: int = 1
    confirmed: bool = False
    last: dict = None
    ls: list = field(default_factory=list)
    stop_hist: list = field(default_factory=list)   # recent frames with a STOP-eligible detection (1/0)
    obs: list = field(default_factory=list)   # (odometry segment, train position x, s) of matched detections

    @property
    def hit_count(self):
        return sum(self.hits)


class Tracker:
    def __init__(self, cfg: TrackerConfig = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks = []
        self.next_id = 1
        self.last_t = None

    def reset(self):
        self.tracks = []
        self.last_t = None

    def update(self, clusters, t):
        cfg = self.cfg
        dt = 0.1 if self.last_t is None else float(np.clip(t - self.last_t, 0.02, 1.0))
        self.last_t = t
        used = np.zeros(len(clusters), bool)
        # predicted positions
        for tr in sorted(self.tracks, key=lambda k: (not k.confirmed, k.s)):
            s_pred = tr.s - tr.v * dt
            unc = cfg.max_closing_speed * dt if tr.age < 3 else 0.35 * max(abs(tr.v), 3.0) * dt
            best, bd = -1, np.inf
            for j, c in enumerate(clusters):
                if used[j]:
                    continue
                ds = c['s_min'] - s_pred
                lo = -(cfg.gate_s_base + cfg.gate_s_rel * tr.s + unc)
                hi = cfg.gate_s_base + cfg.gate_s_rel * tr.s + (unc if tr.age < 3 else 0.5 * unc)
                if not (lo <= ds <= hi) or abs(c['l_mean'] - tr.l) > cfg.gate_l:
                    continue
                d = abs(ds) / (cfg.gate_s_base + cfg.gate_s_rel * tr.s) + abs(c['l_mean'] - tr.l) / cfg.gate_l
                if d < bd:
                    best, bd = j, d
            if best >= 0:
                c = clusters[best]
                used[best] = True
                r = c['s_min'] - s_pred
                tr.s = s_pred + cfg.alpha * r
                tr.v = tr.v - cfg.beta * r / dt
                tr.l = 0.7 * tr.l + 0.3 * c['l_mean']
                tr.hits.append(1)
                tr.misses = 0
                tr.last = c
                tr.ls.append(c['l_mean'])
                tr.ls = tr.ls[-10:]
                c['track'] = tr
            else:
                tr.s = s_pred
                tr.hits.append(0)
                tr.misses += 1
                tr.last = None
            tr.hits = tr.hits[-cfg.window:]
            tr.age += 1
            if tr.hit_count >= cfg.confirm_hits:
                tr.confirmed = True
        self.tracks = [k for k in self.tracks
                       if k.misses <= (cfg.max_misses_confirmed if k.confirmed else cfg.max_misses_tentative) and k.s > -2.0]
        for j, c in enumerate(clusters):
            if not used[j]:
                t = Track(self.next_id, c['s_min'], c['l_mean'], hits=[1], last=c, ls=[c['l_mean']])
                c['track'] = t
                self.tracks.append(t)
                self.next_id += 1
        return self.tracks
