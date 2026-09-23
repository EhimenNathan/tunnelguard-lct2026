"""Dynamic envelope (kinematic gauge) of the train in track coordinates, with uncertainty-aware decisions.

l: lateral offset from the track centreline [m] (+left), h: height above the rail-head plane [m].
The envelope is symmetric and described by a piecewise-linear half-width profile w(h) given as (h, w) vertices.

Decision levels use the per-distance geometry uncertainty (sigma_l, sigma_v from the geometry estimator):
  IN_GAUGE    the point lies inside the envelope shrunk by k*sigma on every side: it is inside the train's
              envelope with high confidence -> may trigger STOP.
  NEAR_GAUGE  the point lies inside the nominal envelope (or within near_margin of it) but not confidently:
              contact with the envelope edge within the measurement uncertainty -> CAUTION.
"""
from dataclasses import dataclass

import numpy as np

OUTSIDE, NEAR_GAUGE, IN_GAUGE = 0, 1, 2


@dataclass
class GaugeConfig:
    # (h, half_width) vertices of the envelope, bottom to top.  Derived from the free space measured in all
    # recordings (see docs): contact rails |l|=1.22 m at h 0.2-0.45 m, platform edges |l|>=1.37 m at h 0.6-1.4 m,
    # tunnel walls |l|>=1.48 m, round-tunnel crown |l|=1.18 m at h=3.43 m, square-tunnel ceiling h=3.8 m.
    # Roof tapered to 3.25 m: ceiling fixtures (lamps, cable trays) hang 0.4-0.6 m below the 3.8 m square-tunnel ceiling.
    profile: tuple = ((0.0, 1.10), (0.50, 1.10), (0.55, 1.25), (1.60, 1.25), (1.70, 1.35), (2.90, 1.35),
                      (3.10, 1.05), (3.25, 0.70))
    h_min: float = 0.15              # objects lower than this above the rail head are ignored [m]
    sigma_k: float = 2.0             # confidence multiplier for IN_GAUGE decisions
    near_margin: float = 0.08        # width of the near-gauge band beyond the nominal envelope [m]


def half_width(h, cfg: GaugeConfig):
    prof = np.asarray(cfg.profile, float)
    return np.interp(h, prof[:, 0], prof[:, 1], left=-1.0, right=-1.0)


def classify(x, l, h, cfg: GaugeConfig, sigma_l=0.0, sigma_v=0.0):
    """Vectorised zone label for points (x along track, l lateral, h height) given per-point uncertainties."""
    top = max(v[0] for v in cfg.profile)
    al = np.abs(l)
    label = np.zeros(len(l), np.uint8)
    nominal = (h >= cfg.h_min) & (al < half_width(h, cfg) + cfg.near_margin)
    label[nominal] = NEAR_GAUGE
    ks_l = cfg.sigma_k * np.asarray(sigma_l)
    ks_v = cfg.sigma_k * np.asarray(sigma_v)
    # shrunken envelope: raise the floor, lower the roof (by stretching the profile), narrow the sides
    top_c = np.maximum(top - ks_v, 0.5)
    h_scaled = h * top / top_c
    confident = nominal & (h >= cfg.h_min + ks_v) & (h <= top_c) & (al < half_width(h_scaled, cfg) - ks_l)
    label[confident] = IN_GAUGE
    return label


def envelope_polygon(cfg: GaugeConfig, x=0.0, sigma_l=0.0, sigma_v=0.0):
    """Closed (l, h) outline of the confident envelope (for visualisation)."""
    prof = np.asarray(cfg.profile, float)
    top = prof[:, 0].max()
    top_c = max(top - cfg.sigma_k * sigma_v, 0.5)
    hs = np.unique(np.r_[cfg.h_min + cfg.sigma_k * sigma_v, prof[:, 0] * top_c / top])
    hs = hs[hs >= cfg.h_min + cfg.sigma_k * sigma_v]
    ws = half_width(hs * top / top_c, cfg) - cfg.sigma_k * sigma_l
    right = np.stack([ws, hs], 1)
    left = right[::-1] * np.array([-1, 1])
    return np.vstack([right, left, right[:1]])
