"""Online self-calibration: the train is its own labeller.

1. Lidar odometry.  The tunnel wall (outside the envelope) carries discrete features - brackets, lamps, ring joints.
   Their along-track profile, detrended and Poisson-normalised, is matched between consecutive frames; the shift is the
   distance travelled (0.1 m grid, parabolic refinement, weak constant-speed prior against periodic structure).
   Validated: exactly 0 on the stationary recording, 1.69 m/frame on a moving one, 13.0 km for the 20-min drive.
2. Proof of clutter.  Every scored candidate beyond the near field is remembered at its tunnel coordinate
   X = x_train + s.  When the train later drives through X, nothing solid can have been there: its score joins a
   rolling reservoir of proven clutter of *this* line.  Real obstacles never enter it - the train stops before them.
3. Adaptive threshold.  tau = clip(max(tau_trained, q_(1-alpha)(reservoir)), tau_trained, tau_max): the alarm
   threshold rises only as far as the local clutter demands, never above tau_max (strong evidence always passes),
   and only after enough proof has accumulated.  The near-field physics rules are not affected.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptConfig:
    enabled: bool = True
    quantile: float = 0.995        # clutter score quantile the threshold must exceed
    tau_max: float = 0.70          # hard cap: scores above this always pass
    min_samples: int = 200         # proof needed before adapting
    reservoir: int = 400           # most recent proven-clutter scores kept
    min_s: float = 30.0            # only candidates beyond the near field
    traverse_margin: float = 3.0   # the train must be this far past X
    max_pending_s: float = 60.0    # forget unconfirmed candidates after this time (train stopped, data gap)


SIG = np.arange(6.0, 90.0, 0.1)


def signature(s, l, h):
    m = (np.abs(l) > 1.6) & (np.abs(l) < 4.5) & (h > 0.3) & (h < 4.0) & (s > SIG[0]) & (s < SIG[-1])
    hist = np.histogram(s[m], bins=np.r_[SIG, SIG[-1] + 0.1])[0].astype(float)
    trend = np.convolve(hist, np.ones(21) / 21, mode='same')
    hist = (hist - trend) / np.sqrt(trend + 1.0)
    hist = np.convolve(hist, np.array([0.25, 0.5, 0.25]), mode='same')
    n = np.linalg.norm(hist)
    return hist / n if n > 0 else hist


def shift_between(prev, cur, prior=None, max_shift=45):
    cs = np.empty(max_shift + 1)
    for d in range(max_shift + 1):
        a, b = prev[d:], cur[:len(cur) - d]
        cs[d] = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
    score = cs if prior is None else cs - 0.5 * ((np.arange(max_shift + 1) * 0.1 - prior) / 0.6) ** 2 * 0.05
    k = int(np.argmax(score))
    frac = 0.0
    if 0 < k < max_shift:
        y0, y1, y2 = cs[k - 1], cs[k], cs[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    return 0.1 * (k + frac), float(cs[k])


class ClutterCalibrator:
    def __init__(self, cfg: AdaptConfig = None):
        self.cfg = cfg or AdaptConfig()
        self.reset()

    def reset(self):
        self.x = 0.0
        self.prev_sig = None
        self.prev_t = None
        self.steps = deque(maxlen=5)
        self.pending = []                     # (X, score, t)
        self.proven = deque(maxlen=self.cfg.reservoir)
        self.speed = float('nan')
        self.segment = 0                      # increments whenever the odometry chain breaks

    def update_motion(self, s, l, h, t):
        sig = signature(s, l, h)
        if self.prev_sig is not None and self.prev_t is not None and 0 < t - self.prev_t < 0.3:
            prior = float(np.median(self.steps)) if self.steps else None
            d, _ = shift_between(self.prev_sig, sig, prior)
            self.x += d
            self.steps.append(d)
            self.speed = d / (t - self.prev_t)
        else:
            self.pending = []                 # odometry chain broken: unconfirmed positions are unusable
            self.segment += 1
            self.steps.clear()
        self.prev_sig, self.prev_t = sig, t

    def observe(self, candidates, t):
        """candidates: iterable of (s_min, score) of scored far-field candidates of this frame."""
        keep = []
        for X, sc, t0 in self.pending:
            if self.x > X + self.cfg.traverse_margin:
                self.proven.append(sc)
            elif t - t0 < self.cfg.max_pending_s:
                keep.append((X, sc, t0))
        for s_min, sc in candidates:
            if s_min >= self.cfg.min_s:
                keep.append((self.x + s_min, float(sc), t))
        self.pending = keep[-5000:]

    def threshold(self, tau_trained):
        if not self.cfg.enabled or len(self.proven) < self.cfg.min_samples:
            return tau_trained
        q = float(np.quantile(np.asarray(self.proven), self.cfg.quantile))
        return float(min(max(tau_trained, q), max(self.cfg.tau_max, tau_trained)))
