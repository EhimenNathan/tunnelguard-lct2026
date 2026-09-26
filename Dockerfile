# TunnelGuard — детектор препятствий в тоннеле метро (ROS 2 Humble)
# docker build -t tunnel_guard .
FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-numpy \
        libopenblas0-pthread \
        python3-scipy \
        python3-pytest \
        python3-matplotlib \
        python3-yaml \
        ros-humble-vision-msgs \
        ros-humble-rosbag2-storage-default-plugins \
        ros-humble-sensor-msgs-py \
    && rm -rf /var/lib/apt/lists/*

# numba ускоряет ядра геометрии ~в 2 раза (нужно для 10 Гц на одном ядре); пакет Ubuntu совместим с системным numpy.
# Если он не ставится, пробуется pip, а если и это не удалось — используется проверенный (более медленный) путь на numpy.
RUN (apt-get update && apt-get install -y --no-install-recommends python3-numba && rm -rf /var/lib/apt/lists/* \
        && python3 -c "import numba") \
    || (apt-get update && apt-get install -y --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/* \
        && pip3 install --no-cache-dir "numba==0.56.4" && python3 -c "import numba") \
    || echo "numba unavailable - numpy fallback will be used"

WORKDIR /ws
COPY src ./src
RUN source /opt/ros/humble/setup.bash \
    && colcon build --event-handlers console_direct+ \
    && rm -rf build log

# один раз компилируем ядра numba в дисковый кэш, чтобы узел был готов через ~1 с после старта
RUN source /ws/install/setup.bash \
    && python3 -c "from tunnel_guard.core.geometry import warmup; print('numba kernels cached:', warmup())"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "tunnel_guard", "tunnel_guard.launch.py"]
