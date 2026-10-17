# Запуск TunnelGuard на компьютере организаторов

Два варианта: готовый Docker с рабочим столом в браузере либо сборка из исходников. Для пользователя с заполненным Windows-диском рекомендуется [облачный Codespaces](FINAL_HANDOFF_RU.md#10-запуск-в-браузере-без-docker-desktop-и-wsl-на-windows): Docker и WSL на его ПК не нужны.

## 1. Готовый контейнер с RViz и графическим интерфейсом

Нужны уже установленный и запущенный Docker Engine + Compose v2 (Linux) либо Docker Desktop в режиме Linux containers (Windows/macOS), интернет для загрузки образа, браузер и каталог стандартных ROS 2 бэгов. Запрашивайте не менее 4 CPU и 8 ГБ RAM. Частота зависит от CPU, облака, размеров кадров и программного OpenGL; приведённый облачный тест не гарантирует ту же скорость на любом ноутбуке.

Получите исходники (нужны Compose и документация):

```bash
git clone https://github.com/EhimenNathan/tunnelguard-lct2026.git
cd tunnelguard-lct2026
docker version
docker compose version
```

### Windows PowerShell

Укажите **свой существующий каталог с распакованными бэгами**. Не указывайте вместо него архив `.zst`: для проигрывания нужен подкаталог с `metadata.yaml` и `.db3`/`.mcap`.

```powershell
Set-Location 'C:\путь\к\tunnelguard-lct2026'
$env:TG_BAGS_DIR='D:\данные\бэги'
Test-Path -LiteralPath $env:TG_BAGS_DIR
docker compose -f compose.gui.yaml config
docker compose -f compose.gui.yaml pull
docker compose -f compose.gui.yaml up -d
docker compose -f compose.gui.yaml ps
```

`Test-Path` должен вернуть `True`. Если данных мало места для распаковки, используйте облако или короткий бэг, созданный потоковым инструментом. Не переносите пример `D:` буквально, если такого диска нет.

### Linux/macOS

```bash
cd /путь/к/tunnelguard-lct2026
export TG_BAGS_DIR=/абсолютный/путь/к/бэгам
test -d "$TG_BAGS_DIR"
docker compose -f compose.gui.yaml config
docker compose -f compose.gui.yaml pull
docker compose -f compose.gui.yaml up -d
docker compose -f compose.gui.yaml ps
```

Откройте **[http://localhost:6080/vnc.html](http://localhost:6080/vnc.html)** → Connect → пароль `tunnelguard`. RViz и терминал находятся в том же контейнере. Не нужны внешний X-сервер, `xhost`, проброс `/tmp/.X11-unix` или установка ROS на хост. Compose слушает только 127.0.0.1. При необходимости задайте `TG_VNC_PASSWORD` до `up`; не публикуйте этот демонстрационный рабочий стол в интернет.

В терминале рабочего стола:

```bash
tg bags
tg play /data/ИМЯ_БЭГА 1.0
```

Если `metadata.yaml` находится в `/data/каталог/бэг`, используйте именно этот путь. `tg play` сам читает PointCloud2-топик, настраивает reliable QoS, сбрасывает память и запускает штатный `ros2 bag play` после готовности детектора.

### Пауза, скорость, шаг и другой бэг

Терминал player должен иметь фокус:

- Пробел — пауза/продолжение.
- ↑ / ↓ — увеличить/уменьшить скорость.
- → — один следующий кадр во время паузы.
- Ctrl+C — остановить только player.

```bash
# Старт на паузе
tg play /data/ИМЯ_БЭГА 1.0 --start-paused

# После Ctrl+C: замедлить или открыть другой бэг, контейнер не перезапускается
tg play /data/ДРУГОЙ_БЭГ 0.5

tg status
tg reset
```

Проверены старт на паузе, Resume, Pause, PlayNext, GetRate/SetRate (0,5 → 0,25), новая запись с другим PointCloud2-топиком и неизменный PID детектора. Для сценариев без интерактивного терминала доступны соответствующие сервисы `/rosbag2_player/*`; типы: `rosbag2_interfaces/srv/Pause`, `Resume`, `PlayNext`, `GetRate`, `SetRate`. Посмотрите актуальные имена через `ros2 service list -t`.

### Остановка и диагностика

```bash
docker compose -f compose.gui.yaml logs --tail=100
docker compose -f compose.gui.yaml exec tunnelguard tg status
docker compose -f compose.gui.yaml exec tunnelguard cat /tmp/tunnel_guard.log
docker compose -f compose.gui.yaml down
```

`down` удаляет контейнер, не каталог данных на хосте. Том с бэгами подключён только для чтения. Снимки/результаты сохраняйте в отдельный подключённый writable-каталог, если они нужны на хосте.

## 2. Сборка на ПК без скачивания готового образа

В каталоге репозитория:

```bash
docker build -t tunnel_guard .
docker build --build-arg TG_BASE=tunnel_guard -f Dockerfile.gui -t tunnel_guard_gui .
docker run --rm -it -p 127.0.0.1:6080:6080 \
  -v /абсолютный/путь/к/бэгам:/data:ro tunnel_guard_gui
```

Windows PowerShell (одна строка, путь берётся из переменной):

```powershell
docker build -t tunnel_guard .
docker build --build-arg TG_BASE=tunnel_guard -f Dockerfile.gui -t tunnel_guard_gui .
docker run --rm -it -p 127.0.0.1:6080:6080 -v "${env:TG_BAGS_DIR}:/data:ro" tunnel_guard_gui
```

Сборка требует места для образов; не запускайте её на заполненном пользовательском диске. Готовый GUI-образ называется `ghcr.io/ehimennathan/tunnel_guard_gui:latest`, базовый — `ghcr.io/ehimennathan/tunnel_guard:latest`.

## 3. Без графики: offline или ROS

```bash
docker run --rm -it -v /путь/к/бэгам:/data:ro \
  ghcr.io/ehimennathan/tunnel_guard:latest tg play /data/ИМЯ_БЭГА 1.0

# Offline: отдельный writable-том результатов
mkdir -p results
docker run --rm --network none -v /путь/к/бэгам:/data:ro -v "$PWD/results:/results" \
  ghcr.io/ehimennathan/tunnel_guard:latest \
  ros2 run tunnel_guard evaluate_bag --bag /data/ИМЯ_БЭГА --out /results/ИМЯ_БЭГА
```

В Windows для writable-тома используйте абсолютный путь, например `-v "C:\результаты:/results"`. Offline-оценка не требует ни интернета после загрузки образа, ни RViz.

## 4. Короткий тестовый бэг без полной распаковки

На машине/облаке с Python и пакетами `numpy`, `zstandard`:

```bash
mkdir -p bags
python tools/falcon_export_bag.py bags/falcon_test 0 300
```

Это читает первые 301 кадр с публичного Яндекс.Диска и материализует небольшой бэг. Полный архив не сохраняется. При существующем локальном кеше:

```bash
python tools/falcon_export_bag.py bags/falcon_test 0 300 --cache /путь/falcon_cache.bin
```

Для уже скачанного `.zst` можно применять потоковый анализ `finals_archive_eval.py` без распаковки, но он сохраняет диагностические JSON, не является player. Полный `tg unpack` выполняйте на диске достаточного размера: сжатый размер не равен распакованному. Облако предпочтительно для крупных архивов.

## 5. Другой LiDAR и проверка осей

При запуске неизвестного сенсора сначала визуально проверьте оси. Нативный Falcon: X вверх, Y вправо, Z вперёд; многие ROS-драйверы уже преобразуют его в X вперёд/Z вверх. Нельзя применять преобразование дважды.

`forward_axis` — параметр узла, а не аргумент штатного launch. Для явного режима остановите совместный запуск и в отдельном терминале выполните:

```bash
tg stop
ros2 run tunnel_guard detector_node --ros-args -p forward_axis:=falcon
```

Для предоставленной записи Falcon нужен её фактический ROS-режим или `auto`, а не принудительный native Falcon. Доступные launch-аргументы проверьте `ros2 launch tunnel_guard tunnel_guard.launch.py --show-args`. YAML находится в `src/tunnel_guard/config/tunnel_guard.yaml`, генератор — `src/tunnel_guard/tools/gen_config.py`.

## 6. Если изображение или сообщения не появились

1. `docker compose ... ps`: контейнер должен работать, health перейти в healthy. Проверьте лог, свободный порт 6080 и доступ к образу GHCR.
2. В GUI `tg bags`: должны находиться реальные metadata.yaml; пустой список обычно означает неверный том/каталог.
3. `ros2 topic list -t`: есть ли PointCloud2 и `/tunnel_guard/status`? Проверьте `/tmp/tunnel_guard.log`.
4. Нет СТОП на каждом кадре — не обязательно ошибка запуска: решение зависит от объекта, габарита, прогрева и подтверждения. Дождитесь нескольких секунд записи, проверьте оси и облако.
5. Низкая скорость: проверьте выделенные CPU/RAM, software OpenGL, размер кадров. Не путайте частоту решений 10 Гц и графику 5 Гц; `callback_ms` измеряет полный callback.
6. При намеренной остановке RViz в Xvfb/Mesa наблюдалась ошибка его завершения; она отдельно отражена в протоколе. Во время обработки финального облачного теста детектор и RViz работали, сообщения получены без потерь.

Подтверждённые результаты, конфигурация и ограничения: [METRICS_RU.md](finals/METRICS_RU.md), [FINAL_HANDOFF_RU.md](FINAL_HANDOFF_RU.md). Нельзя гарантировать отсутствие ошибок на неизвестном железе; для передачи организаторам фиксируйте digest образа и выполняйте короткий smoke test на их машине.
