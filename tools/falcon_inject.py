"""Physically ray-cast synthetic obstacles into real Falcon frames (evaluation only).

Sensor model measured from the recording itself: in the region of interest (the dense band, azimuth -10..30 deg,
elevation -4..2 deg) the Falcon samples ~156 rays/deg^2 (0.097 deg x 0.066 deg), outside it 23-47 rays/deg^2
(0.17 deg x 0.113 deg); the scan phase changes from frame to frame.  Each ray that crosses an object returns from the
first surface with a range-dependent probability p = 1 / (1 + exp((r - R50) / (0.08 R50))), R50 = 1.1 * 250 m *
sqrt(rho / 0.1) (datasheet: 250 m at 10 % reflectivity, POD > 90 %); the real returns behind the object along those
rays are removed (occlusion).  Calibration: a synthetic person (rho 0.10) at 260 m gives ~4.5 returns per frame, the
real walking person at 260 m gave 4.6.

Aerosol (dust, smoke, spray) is volumetric: a ray crossing a cloud of extinction sigma over a chord L returns with
probability 1 - exp(-sigma L), from an exponentially distributed depth inside the cloud.

Objects are given in the levelled track frame (x ahead, y left, z up, floor at z = FLOOR) and converted to the
sensor frame with the measured mounting pitch.
"""
import numpy as np

PITCH = np.radians(2.4)          # Falcon mounting pitch measured from the floor (lidar tilted up)
FLOOR = -2.05                    # track bed in the levelled frame [m]
TRACK_Y = 0.35                   # our track centre (lateral) in the levelled frame [m]
ROI = (-10.0, 30.0, -4.0, 2.0)   # az0, az1, el0, el1 [deg]


def level_to_sensor(p):
    c, s = np.cos(PITCH), np.sin(PITCH)
    return np.column_stack([p[:, 0] * c + p[:, 2] * s, p[:, 1], -p[:, 0] * s + p[:, 2] * c])


def sensor_to_level(p):
    c, s = np.cos(PITCH), np.sin(PITCH)
    return np.column_stack([p[:, 0] * c - p[:, 2] * s, p[:, 1], p[:, 0] * s + p[:, 2] * c])


def p_detect(r, rho):
    r50 = 1.1 * 250.0 * np.sqrt(rho / 0.1)
    return 1.0 / (1.0 + np.exp((r - r50) / (0.08 * r50)))


def ray_grid(az_lo, az_hi, el_lo, el_hi, rng):
    """Ray directions (deg) of one frame inside an angular window, Falcon spacing, random scan phase."""
    in_roi = ROI[0] <= 0.5 * (az_lo + az_hi) <= ROI[1] and ROI[2] <= 0.5 * (el_lo + el_hi) <= ROI[3]
    daz, de = (0.097, 0.066) if in_roi else (0.17, 0.113)
    az = np.arange(az_lo - daz + rng.uniform(0, daz), az_hi + daz, daz)
    el = np.arange(el_lo - de + rng.uniform(0, de), el_hi + de, de)
    A, E = np.meshgrid(az, el)
    return A.ravel(), E.ravel()


def _dirs(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.column_stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def box_hits(d, lo, hi):
    """Ray (from the origin, unit directions d) / axis-aligned box intersection: entry distance or inf."""
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = lo[None, :] / d
        t2 = hi[None, :] / d
    tmin = np.nanmax(np.minimum(t1, t2), 1)
    tmax = np.nanmin(np.maximum(t1, t2), 1)
    ok = (tmax >= tmin) & (tmax > 0)
    return np.where(ok, np.maximum(tmin, 0), np.inf), np.where(ok, tmax, np.inf)


def inject(xyz, inten, objects, rng):
    """xyz: real frame (sensor frame). objects: list of dicts kind=person|box|dust, x (distance ahead, levelled),
    y (lateral, default track centre), size (w, d, h) and rho / sigma.  Returns new (xyz, intensity, n_object_returns)."""
    if not objects:
        return xyz, inten, []
    lev = sensor_to_level(xyz)
    r_real = np.linalg.norm(xyz, axis=1)
    keep = np.ones(len(xyz), bool)
    new_pts, new_int, counts = [], [], []
    for ob in objects:
        w, dp, h = ob.get('size', (0.5, 0.3, 1.75))
        y0 = ob.get('y', TRACK_Y)
        z0 = ob.get('z0', FLOOR)
        lo_l = np.array([ob['x'], y0 - w / 2, z0])
        hi_l = np.array([ob['x'] + dp, y0 + w / 2, z0 + h])
        # angular window of the object (sensor frame)
        corners = np.array([[x, y, z] for x in (lo_l[0], hi_l[0]) for y in (lo_l[1], hi_l[1]) for z in (lo_l[2], hi_l[2])])
        cs = level_to_sensor(corners)
        caz = np.degrees(np.arctan2(cs[:, 1], cs[:, 0]))
        cel = np.degrees(np.arctan2(cs[:, 2], np.hypot(cs[:, 0], cs[:, 1])))
        az, el = ray_grid(caz.min(), caz.max(), cel.min(), cel.max(), rng)
        d_s = _dirs(az, el)
        d_l = sensor_to_level(d_s)                       # rotation only: directions in the levelled frame
        t_in, t_out = box_hits(d_l, lo_l, hi_l)
        hit = np.isfinite(t_in)
        if ob['kind'] == 'dust':
            L = np.where(hit, t_out - t_in, 0.0)
            ret = hit & (rng.random(len(L)) < 1.0 - np.exp(-ob.get('sigma', 0.6) * L))
            depth = t_in + np.minimum(rng.exponential(1.0 / ob.get('sigma', 0.6), len(L)), L)
            ret &= rng.random(len(L)) < p_detect(depth, ob.get('rho', 0.05) * 4)
            r_hit = depth
        else:
            r_hit = t_in
            ret = hit & (rng.random(len(r_hit)) < p_detect(r_hit, ob.get('rho', 0.10)))
        # occlusion: real returns behind the object along the rays that hit it
        if hit.any():
            raz = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
            rel = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
            inwin = (raz > caz.min()) & (raz < caz.max()) & (rel > cel.min()) & (rel < cel.max()) & \
                (r_real > np.nanmin(np.where(hit, r_hit, np.inf)))
            if ob['kind'] != 'dust':
                keep &= ~inwin
        pts = d_s[ret] * r_hit[ret][:, None]
        new_pts.append(pts.astype(np.float32))
        new_int.append(np.full(len(pts), 25.0 if ob['kind'] != 'dust' else 3.0, np.float32))
        counts.append(int(ret.sum()))
    out = np.vstack([xyz[keep]] + new_pts)
    oi = np.concatenate([inten[keep]] + new_int)
    return out, oi, counts


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    for d in (100, 200, 260, 300):
        n = [inject(np.zeros((1, 3), np.float32) + 1, np.zeros(1, np.float32),
                    [dict(kind='person', x=float(d))], rng)[2][0] for _ in range(200)]
        print(f'synthetic person at {d} m: {np.mean(n):.1f} returns/frame (zero in {100 * np.mean(np.array(n) == 0):.0f}% of frames)')
