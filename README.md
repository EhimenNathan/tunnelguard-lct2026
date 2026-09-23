# TunnelGuard — foreign-object detection for a driverless metro train (3D lidar, ROS 2 Humble)

TunnelGuard looks down the tunnel with the train's 3D lidar and answers one question every frame:

> **"Is anything inside the train's dynamic envelope ahead — and up to what distance is the path verified clear?"**

It does **not** try to recognise object classes. It learns *what the normal tunnel looks like* from the lidar
itself (rails, cross-section, curvature, grade) and declares an obstacle when something occupies the space the
train is about to sweep. Because of that it needs no labelled data and generalises to new tunnel sections,
new obstacle types and even a different lidar mounting.

| Output (per lidar frame) | Topic | Type |
|---|---|---|
| Decision: CLEAR / CAUTION / STOP, nearest obstacle distance, time-to-collision, verified clear distance, all obstacles | `/tunnel_guard/status` | `tunnel_guard_msgs/ObstacleStatus` |
| Obstacles as standard 3D detections | `/tunnel_guard/detections` | `vision_msgs/Detection3DArray` |
| Alarm level (0/1/2) | `/tunnel_guard/alarm` | `std_msgs/UInt8` |
| Along-track distance to nearest in-gauge obstacle (NaN = none) | `/tunnel_guard/nearest_distance` | `std_msgs/Float32` |
| Envelope, track centreline, obstacle boxes + distance labels | `/tunnel_guard/markers` | `visualization_msgs/MarkerArray` |
| Points of confirmed obstacles / all envelope points | `/tunnel_guard/obstacle_points`, `/tunnel_guard/envelope_points` | `sensor_msgs/PointCloud2` |

---

## 1. Quick start (docker build → docker run → ros2 bag play → see result)

```bash
# 1. build (Ubuntu 22.04 + ROS 2 Humble inside, all dependencies installed automatically)
docker build -t tunnel_guard .

# 2. run the detector + RViz2 and play a bag in one command (bags mounted at /data)
xhost +local:root   # allow the container to open RViz on the host display
docker run --rm -it --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /path/to/for_hackathon:/data tunnel_guard \
    ros2 launch tunnel_guard tunnel_guard.launch.py rviz:=true bag:=/data/doubleT_obstacle
```

Other ways to run:

```bash
# detector only (subscribes to the first PointCloud2 topic it finds; the bag is played from anywhere on the network)
docker run --rm -it --net=host tunnel_guard
ros2 bag play /path/to/for_hackathon/doubleT_obstacle          # in another shell / container
ros2 topic echo /tunnel_guard/status                            # decisions

# offline, faster than real time: per-frame CSV + JSON summary
docker run --rm -v /path/to/for_hackathon:/data tunnel_guard \
    ros2 run tunnel_guard evaluate_bag --bag /data/doubleT_obstacle --out /data/results/doubleT_obstacle

# unit + end-to-end tests (synthetic curved tunnel with ground truth)
docker run --rm tunnel_guard python3 -m pytest -q /ws/src/tunnel_guard/test
```

With `bag:=...` the launch file reads the PointCloud2 topic from the bag's `metadata.yaml`, starts playback only after
the detector reports it is ready, and replays the bag with reliable QoS on both ends: large clouds (9–24 MB) sent
best-effort over DDS lose fragments, which would silently drop frames. A live lidar keeps sensor-data (best-effort) QoS.

Verified end to end on Ubuntu 22.04 + ROS 2 Humble (WSL2): `colcon build`, 14 pytest tests,
`ros2 launch tunnel_guard tunnel_guard.launch.py bag:=<real frames of doubleT_obstacle>` → `/tunnel_guard/status` reports
STOP, and `evaluate_bag` on the exported test bag → first STOP at 55.5 m with the deployed model. The Docker image installs OpenBLAS (Ubuntu's default reference BLAS made the geometry solver 4× slower)
and pre-compiles the numba kernels at build time.

The input topic is discovered automatically (`input_topic:=auto`), so both recorded topic names
(`/lidar_points`, `/sensing/lidar/hesai128/pointcloud`) work without configuration. The forward axis of the lidar is
also detected automatically (`forward_axis:=auto`: the horizontal axis with the most far returns).

---

### Windows (Docker Desktop, WSL2 engine)

Run in PowerShell from the solution folder (no Docker account is needed; ~6 GB free disk for the image):

```powershell
docker build -t tunnel_guard .
docker run --rm tunnel_guard python3 -m pytest -q /ws/src/tunnel_guard/test
docker run --rm -v "C:\path\to\bags:/data" tunnel_guard ros2 launch tunnel_guard tunnel_guard.launch.py bag:=/data/doubleT_obstacle_7s copy_bag:=true
docker run --rm -v "C:\path\to\bags:/data" tunnel_guard ros2 run tunnel_guard evaluate_bag --bag /data/doubleT_obstacle_7s --out /data/results
```

`copy_bag:=true` matters on Docker Desktop (Windows/macOS): `ros2 bag play` reads the bag's SQLite file through the
host-folder mount far slower than real time, so most frames never reach the detector. The option copies the bag into
the container first. On a Linux host the mount is native and the option is not needed.

Verified with Docker Desktop 4.91 (WSL2 engine) on the exported test bag: 14 tests pass; `ros2 launch ... copy_bag:=true`
→ 66 frames processed, `/tunnel_guard/status` CLEAR → CAUTION → STOP, 58 STOP frames, first STOP at 55.5 m, ~100 ms per
frame on a 15 W laptop CPU (921 600-ray clouds of the obstacle recording; 56–70 ms on the 307 200-ray recordings).

RViz needs a display: run the Linux command of step 2 above from a WSL (Ubuntu) terminal, where WSLg provides it.

**Test bag without the original archive.** `tools/export_bag.py` writes a standard ROS 2 Humble bag (sqlite3, CDR
`sensor_msgs/PointCloud2`, fields x, y, z, intensity) from the range-image cache, with no ROS installation needed:

```bash
python tools/export_bag.py doubleT_obstacle test_bags/doubleT_obstacle_7s 0 69 --topic /sensing/lidar/hesai128/pointcloud --frame lidar_livox
```

Checked with ROS 2 Humble: `ros2 bag info` reads 70 messages; `evaluate_bag` gives the first STOP at frame 10, 55.5 m,
73 ms per frame.  (Range is quantised to 1 cm in the cache; invalid returns are dropped.)

## 2. Architecture

```
 ROS 2 bag / live lidar
        │  sensor_msgs/PointCloud2 (any field layout, any topic)
        ▼
 ┌───────────────────────── tunnel_guard/detector_node (rclpy) ──────────────────────────┐
 │  decode (zero-copy numpy) → axis detection → FOV crop / near-field decimation        │
 │        │                                                                            │
 │        ▼                        core (pure numpy/scipy, ROS-independent)            │
 │  ┌──────────────┐   ┌───────────────────────┐   ┌──────────────┐   ┌─────────────┐  │
 │  │ Track        │ → │ Envelope test with     │ → │ Range-adaptive│ → │ Multi-frame │  │
 │  │ geometry     │   │ uncertainty (IN / NEAR)│   │ clustering +  │   │ confirmation│  │
 │  │ estimator    │   │ + sight horizon        │   │ containment   │   │ + TTC       │  │
 │  └──────────────┘   └───────────────────────┘   └──────────────┘   └─────────────┘  │
 │        │                                                                   │        │
 └────────┼───────────────────────────────────────────────────────────────────┼────────┘
          ▼                                                                   ▼
   markers (RViz2)                         ObstacleStatus · Detection3DArray · alarm · distance
```

Source layout:

```
src/tunnel_guard_msgs/            Obstacle.msg, ObstacleStatus.msg
src/tunnel_guard/
  tunnel_guard/core/cloud.py      PointCloud2 decoding, forward-axis detection
  tunnel_guard/core/geometry.py   rails DP, cross-section template, global DP + dense Gauss-Newton alignment, uncertainty
  tunnel_guard/core/gauge.py      envelope profile, uncertainty-aware zone classification
  tunnel_guard/core/cluster.py    track-aligned range-adaptive clustering
  tunnel_guard/core/tracker.py    M-of-N confirmation, closing speed
  tunnel_guard/core/detector.py   pipeline orchestration, sight horizon, containment check, decision
  tunnel_guard/core/adapt.py      lidar odometry (wall signature) and online self-calibration (off by default)
  tunnel_guard/core/synth.py      ray-cast synthetic obstacles (evaluation only)
  tunnel_guard/detector_node.py   ROS 2 node
  tunnel_guard/evaluate_bag.py    offline bag evaluation
  config/tunnel_guard.yaml        all parameters (generated from the code defaults)
  launch/, rviz/, test/
```

---

## 3. Algorithm

### 3.1 Why geometry first
In a metro tunnel "obstacle" has no appearance — it is *anything inside the space the train will sweep*.
That space is defined by the track, which curves (R ≈ 300–1000 m), climbs and falls (station humps, ±30‰),
is superelevated, and is seen by a lidar whose mounting differs between trains (the obstacle recording has a
rolled lidar mounted 0.5 m higher). A straight "box in front of the train" either floods with false alarms from
walls in curves or misses objects on the actual track. TunnelGuard therefore reconstructs the track geometry
**every frame, from the lidar alone**, out to the lidar horizon.

### 3.2 Track geometry estimator (`geometry.py`)
1. **Rails (3–45 m).** BEV max-height map (0.5–1 m × 3 cm cells) → lateral white top-hat keeps narrow ridges
   6–30 cm high (rail heads) → gauge-pair evidence `S(x,c) = min(R(x,c−0.80), R(x,c+0.80))` for 1520 mm gauge →
   **Viterbi dynamic programming** finds the globally best smooth centre path. Gives centreline, rail-head height and
   cross-level (roll) with cm precision, independent of the scan pattern. Far rail detections are gated by temporal
   consistency (gate guide rails and switches cannot hijack the track).
2. **Cross-section template.** A tunnel keeps its cross-section along the track. Near-field points in track
   coordinates `(l, h)` are accumulated (temporal memory) into an occupancy image = *the model of the normal tunnel*;
   its truncated distance transforms are the alignment targets.
3. **Global coarse search.** For 3–5 m slabs out to 300 m, DP over lateral-offset hypotheses with template-distance cost
   and a stiff continuity cost (a track cannot detour metres sideways). Globally optimal — resolves "only the outer wall
   is visible" in curves and never lets a compact object bend the track.
4. **Dense direct alignment.** Centreline `y_c(x)` and rail height `z_r(x)` (1 m grid) are refined by **Gauss–Newton**:
   every lidar point should lie on a template surface; each point constrains the profile at its own distance (oblique
   walls in curves included). Banded Jacobian → one banded solve per iteration; coarse-to-fine truncation; Tukey weights.
   Priors: rails and a **clothoid prior** penalising curvature *change* (railway transition curves).
5. **Obstacle invariance.** Points inside the envelope space never influence the geometry used to judge them
   (lateral: corridor core excluded; vertical: only track-bed level or structure outside the envelope footprint).
   Validated with ray-cast obstacles: a person at 160 m no longer moves the estimated track (error < 10 cm).
6. **Temporal Kalman fusion.** The previous profile is a prior weighted by its own uncertainty + process noise;
   uncertainty σ(x) fuses only *measured* information (an extrapolation is not a measurement), grows linearly beyond
   the last measured node, and keeps a distance-growing systematic term that fusion never reduces.
7. **Sight distance.** Through a curved tunnel the corridor beyond the chord that clears the tunnel wall
   (≈√(8RW)) is physically invisible; evaluation stops there.

### 3.3 Envelope, decision and confirmation
* **Envelope profile** `w(h)` measured from the data (free space of all recordings in track coordinates):
  contact rails at |l| = 1.22 m (h 0.2–0.45 m), platform edges ≥ 1.37 m, walls ≥ 1.48 m, round-tunnel crown 1.18 m
  at 3.43 m, square-tunnel ceiling 3.8 m (fixtures below it → roof tapered to 3.25 m). Objects lower than 0.15 m above
  the rail head are ignored.
* **Uncertainty-aware zones.** `IN_GAUGE` requires the point to be inside the envelope shrunk by 2σ;
  points inside the nominal envelope only are `NEAR_GAUGE` (contact within the measurement error → CAUTION).
* **Two-tier range.** STOP only where this frame measured the track geometry; beyond it (temporally trusted geometry)
  an in-envelope object is an early warning (CAUTION) that upgrades to STOP as measurement reaches it.
* **3D clustering** on a track-aligned grid (along-track cell grows with range), split at vertical gaps larger than
  the lidar's vertical beam spacing (Pandar128: 0.125°).
* **Physical plausibility tests** (each is a statement about the world, not a tuned threshold):
  - *containment* — a foreign object can never be the outermost structure on its side; required only where a wall
    displacement into the cluster's position is statistically plausible (within 3σ);
  - *shell attachment* — tall clusters that continue upward into the tunnel shell are infrastructure (columns, masts);
  - *gravity* — beyond 60 m only floor-supported objects can trigger STOP (suspended fixtures → CAUTION);
  - *shape* — a physical object spans at least ~1.2 vertical beam spacings at its range (or has ≥ 15 points).
* **Confirmation** in track coordinates: 3 detections within 5 frames (≈0.2–0.3 s), gating by the maximum
  closing speed; α-β filter gives closing speed and time-to-collision.
* **Decision:** STOP if a confirmed object is confidently inside; CAUTION if one touches the envelope or is only
  early-warned; CLEAR otherwise. `clear_distance` = distance up to which the envelope was actually verified.

### 3.4 Learned plausibility scorer (physics-first hybrid)
The geometry and the envelope stay analytic; a small learned model only answers *"is this in-envelope cluster a real
object or a piece of tunnel?"* for clusters beyond 30 m, where the rule tests are least certain.
* **Features (38, `core/features.py`)** — only physical, sensor-invariant quantities: extent in beam spacings, points per
  vertical beam, depth inside the envelope, margins to the measured/trusted/sight range, σ of the geometry, wall
  evidence on both sides, shell attachment, lateral stability of the track. Intensity, ring index, track age and recording
  identity are deliberately **excluded** (they would learn the recording, not the physics).
* **Data without leakage** — negatives: every candidate cluster of the 5 obstacle-free recordings (5 864 rows);
  positives: ray-cast obstacles of random shape, size, reflectivity, lateral position, range and speed inserted into the
  real beams. The real obstacle recording is never seen.
* **Validation = leave-one-recording-out** (train on 4 tunnels, test on the 5th) at the *decision* level: false-STOP
  frames on clean frames and object recall, threshold chosen by nested CV under a false-alarm budget.
* **Models compared** (LORO average precision): logistic regression 0.877, LightGBM 0.888, monotone LightGBM 0.887,
  XGBoost 0.884, CatBoost 0.890, MLP 0.864, physics-informed MLP (monotonicity penalty on the gradient w.r.t. physical
  evidence) 0.858; Optuna tuning: monotone LGB 0.893, CatBoost 0.894, PI-MLP 0.867.
* **Deployed:** mean of logits of *monotone LightGBM + CatBoost + physics-informed MLP* (diversity beats the best single
  AP at the decision level), 5-frame score smoothing along each track, threshold 0.37. Hard physical veto: a cluster
  attached to the tunnel shell can never be promoted. Within 30 m the rule decision is kept unchanged.
* **Inference without ML libraries** — models are exported to JSON (`config/obstacle_scorer.json`, 0.4 MB) and evaluated by
  a packed numpy traversal (`core/scorer.py`), bit-exact to LightGBM/CatBoost. Physics rules only: `-p scorer_model:=none`
  or `evaluate_bag --no-scorer`.

### 3.5 Speed
Vectorised coarse DP cost (bincount, exactly equal to the loop), numba kernels for the rail DP, coarse DP and the
Gauss–Newton normal equations (numpy fallback, identical results), column-selected rotation, a distance-bucketed
neighbourhood index and a 2 m sight grid: **268 → ~70 ms** per frame on the laptop (921 600-point clouds 552 → 89 ms),
i.e. faster than the 10 Hz lidar on a 15 W CPU, single thread. Kernels are JIT-compiled at node start-up.

---

## 4. Parameters (`config/tunnel_guard.yaml`)

All parameters are ROS parameters (`--ros-args -p group.name:=value`), generated from the dataclass defaults.

| Parameter | Default | Meaning |
|---|---|---|
| `input_topic` | `auto` | PointCloud2 topic; `auto` = taken from the bag, else first one found |
| `input_reliability` | `best_effort` | subscription QoS; the launch file sets `reliable` for bag playback |
| launch `play_delay`, `rate` | 1.0 s, 1.0 | bag playback starts `play_delay` after the detector is ready |
| `forward_axis` | `auto` | `x`, `-x`, `y`, `-y` or auto-detection |
| `min_range`, `min_distance` | 1.5 m, 3 m | ignore own-train returns |
| `half_fov_deg` | 70 | forward field of view used |
| `gauge.profile` | see file | envelope half-width vs height above rail head (flat list h, w, h, w …) |
| `gauge.h_min` | 0.15 m | minimum object height above the rail head |
| `gauge.sigma_k` | 2.0 | confidence multiplier for STOP decisions |
| `geometry.smooth_lat / smooth_vert` | (1e3, 1e8) | curvature / curvature-change penalties |
| `geometry.sigma_sys_l / sigma_sys_v` | (0.04, 0.10) / (0.03, 0.05) | systematic geometry error model |
| `tracker.confirm_hits / window` | 3 / 5 | M-of-N confirmation |
| `sight_clearance` | 1.8 m | line-of-sight room inside the tunnel |
| `containment_check` | true | lateral containment test |
| `scorer_model` | package `config/obstacle_scorer.json` | learned plausibility scorer (empty = rules only) |
| `scorer_near_override_s` | 30 m | below this distance the rule decision is kept |

---

## 5. Results (details and reproduction: `docs/EXPERIMENTS.md`, `tools/`)

Final model: physics layer + **½·LightGBM (monotone) + ½·CatBoost + ego-motion test** (internal name v3; the previous
model v1 was ⅓·LightGBM + ⅓·CatBoost + ⅓·physics-informed MLP).  Ensemble weights: fixed equal (mean of logits);
a grouped out-of-fold sweep of w over 0…1 changes recall only 67.3–67.6 % and false STOP 0.30–0.33 %
(`tools/weight_sweep.py`), so equal weights were kept.  The two data sets
are always reported separately.

**Dataset 1: original recordings (5 obstacle-free tunnels + 1 held-out recording with people)**

| What | Result |
|---|---|
| Held-out real recording (never used for development, training or tuning): person inside the gauge | STOP in **58 / 60** frames, first STOP at **55.5 m**; **0** spurious STOP (precision 100 %, recall 96.7 %, accuracy 99.0 %) |
| False STOP on all 2 287 frames of the 5 obstacle-free tunnels | 0–3 frames (≤ 0.13 %) over 3 geometry seeds |
| Ray-cast person / 0.5 m box in real beams, first second of approach | person **100 / 78 / 33 %** at 80 / 120 / 160 m; box 90 / 71 / 0 % |

**Dataset 2: new line, 20-min drive, 13 km, no obstacles (not used to develop the detector)**

| What | Result |
|---|---|
| Self-labelling by traversal | all 91 alarms of the previous model proven false (the train later drove through each place) |
| Sealed final test (last 5 min, 2.8 km, never used for training or tuning), mean of 3 seeds | **1.7 false STOP events / km** (previous model 10.3), 0.43 % of frames (was 2.06 %) |
| Ray-cast person standing on the line, train at its real speed (60 km/h) | first STOP at **142 m** (demo video) |

Median latency: **56–70 ms** per frame on one CPU core (laptop i5-8250U), below the 100 ms of the 10 Hz lidar.

Further analysis in `docs/EXPERIMENTS.md`: precision / recall / accuracy (§6), leakage controls and generalization (§7),
long-range study (§8), ablation study (§9), Pandar128 files (§10), the new line: self-labelling, domain-robust scorer,
ego-motion test, sealed test (§12).

Demo video (105 s, Russian captions): `docs/demo_tunnelguard.mp4`:
- dataset 1: held-out real people (STOP at 55.5 m), and a ray-cast person approaching from 200 m (first STOP at 150 m);
- dataset 2: the train at line speed with lidar odometry, a ray-cast person standing on the new line (STOP at 142 m),
  and the same seconds with the previous model (12 false STOP frames) and the deployed one (0).

Every frame is the live output of the detector on real recorded lidar frames. Synthetic objects are marked
«СИНТЕТИКА». Presentation: `presentation/TunnelGuard_LCT2026.pptx` (figures: `tools/make_deck_figures.py`,
`tools/make_ds2_figures.py`; video: `tools/make_video.py`).

## 6. Limitations (honest)
* **Latency.** ≈ 70 ms median on a 15 W laptop CPU (single thread, numba) — within the 100 ms lidar period with little
  margin on p95 frames; the node always processes the newest cloud (QoS depth 2, best effort), so it never falls behind.
  A C++ port of `geometry.py` (≈ 45 % of the time) is the next step for a hard real-time guarantee.
* **The learned scorer is trained on ray-cast positives.** Real obstacles in real tunnels are needed to calibrate it
  further; the physics rules remain in force within 30 m and the shell veto is absolute.
* **Range is limited by physics.** The Pandar128 returns ≈ 200 m at 10 % reflectivity; at 160 m a person yields ≈ 10
  points and the vertical beam spacing is 0.35 m. In curves the corridor is only visible up to ≈ √(8RW).
* **Small objects.** A 0.3 m box is reliably confirmed only within ≈ 40–60 m; beyond that it yields CAUTION at most
  (shape plausibility rule). This is a deliberate trade-off against false stops.
* Objects lower than 0.15 m above the rail head are ignored; suspended objects beyond 60 m raise CAUTION, not STOP.
* CAUTION is frequent near infrastructure that touches the envelope within the measurement error (switches,
  platforms): it is informational and never commands braking.
* Only one real obstacle recording was available; range figures rely on physically ray-cast obstacles.
