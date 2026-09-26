# Инструменты оценки (вне ROS; воспроизводят каждое число из docs/EXPERIMENTS.md)

1. `convert.py full` — потоково разбирает SQLite-файлы rosbag2 прямо из `for_hackathon.zst` (без распаковки и без
   sqlite; учитывает служебную страницу блокировки SQLite) в компактные кэши дальностных изображений (`cache/`,
   ~600 МБ на все бэги). `loader.py` читает их (и достраивает направления лучей, не давших отражения, по точной
   разделимой модели Pandar128).
2. `eval_empty.py <bags> <tag>` — каждый кадр записей без препятствий -> ложные тревоги, задержка.
3. `eval_synth.py shapes=... d0=... starts=3 seq=12 tag=...` — препятствия лучевым моделированием (реальные
   направления лучей, затенение, дальность Pandar128 в зависимости от отражательной способности) в реальных кадрах ->
   полнота в зависимости от дистанции.
4. `eval_holdout.py` — отложенная реальная запись с препятствием против независимого эталона (вычитание фона).
5. `make_report.py`, `make_figures.py`, `build_deck.py` — docs/EXPERIMENTS.md, рисунки, демо-видео, презентация.
6. `trace_far.py bag frame distance shape` — покадровая трассировка каждого этапа конвейера для одного синтетического
   препятствия.
7. Обучаемый классификатор (вся валидация — «оставь одну запись»; запись с препятствием никогда не используется):
   `gen_dataset.py` (кластеры-кандидаты пустых бэгов + положительные примеры лучевого моделирования -> 38 физических
   признаков), `train_scorer.py` (LR / LightGBM / монотонный LightGBM / XGBoost / CatBoost / MLP / физически
   информированная MLP, смеси, моделирование на уровне решения), `tune_scorer.py` (Optuna), `export_scorer.py`
   (модели -> JSON без зависимостей, проверка совпадения), `final_scorer.py` (обучает развёрнутую смесь на всех
   данных -> `config/obstacle_scorer.json`), `make_hybrid_figure.py`. Скрипты оценки используют классификатор, если
   задано `TG_SCORER=<путь к json>`.
8. Новая линия и датасет 3: `run_drive.py`, `selflabel.py` (саморазметка проездом), `gen_dataset_v3.py`,
   `train_scorer_v3.py`, `final_scorer_v3.py` (`DS_SUFFIX=4` — эксперт для висящих объектов), `weight_sweep.py`,
   `ablation.py`, `sealed_summary.py`, `stream_eval_ds3.py`, `ds3_stream.py`, `replica_ds3.py`.
9. Презентация и видео: `make_deck_figures.py` (рисунки на русском в палитре шаблона → docs/figures/deck),
   `make_video.py` (docs/demo_tunnelguard.mp4 + ключевые кадры), `build_deck.py` (презентация по официальному
   шаблону), `preview_deck.py deck.pptx out_dir` (рендерит слайды в PNG и сообщает о вылезающем тексте — проверка без
   PowerPoint), `export_web.py`, `export_web_ds3.py` (данные для сайта `web/`).
