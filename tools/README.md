# Evaluation tools (outside ROS; reproduce every number in docs/EXPERIMENTS.md)

1. `convert.py full` — stream-parses the rosbag2 SQLite files directly from `for_hackathon.zst` (no extraction, no
   sqlite; handles the SQLite lock-byte page) into compact range-image caches (`cache/`, ~600 MB for all bags).
   `loader.py` reads them (and completes never-returned beam directions with the exact Pandar128 separable model).
2. `eval_empty.py <bags> <tag>` — every frame of the obstacle-free recordings -> false alarms, latency.
3. `eval_synth.py shapes=... d0=... starts=3 seq=12 tag=...` — ray-cast obstacles (real beam directions, occlusion,
   Pandar128 range-vs-reflectivity) into real frames -> recall vs distance.
4. `eval_holdout.py` — held-out real obstacle recording vs an independent background-subtraction reference.
5. `make_report.py`, `make_figures.py`, `build_deck.py` — docs/EXPERIMENTS.md, figures, demo video, presentation.
6. `trace_far.py bag frame distance shape` — per-frame trace of every pipeline stage for one synthetic obstacle.
7. Learned scorer (all validation is leave-one-recording-out; the obstacle recording is never used):
   `gen_dataset.py` (candidate clusters of the empty bags + ray-cast positives -> 38 physical features),
   `train_scorer.py` (LR / LightGBM / monotone LightGBM / XGBoost / CatBoost / MLP / physics-informed MLP, blends,
   decision-level simulation), `tune_scorer.py` (Optuna), `export_scorer.py` (models -> dependency-free JSON, parity
   check), `final_scorer.py` (trains the deployed blend on all data -> `config/obstacle_scorer.json`),
   `make_hybrid_figure.py`. The eval scripts use the scorer when `TG_SCORER=<path to json>` is set.
8. Presentation and video: `make_deck_figures.py` (Russian figures in the template palette → docs/figures/deck),
   `make_video.py` (docs/demo_tunnelguard.mp4 + key frames), `build_deck.py` (presentation from the official template),
   `preview_deck.py deck.pptx out_dir` (renders slides to PNG and reports overflowing text — QA without PowerPoint).
