#!/bin/bash
# Codespaces: окружение ROS в каждом терминале и тестовый бэг из записи Falcon организаторов (потоково, ~30 с записи).
set -e
grep -q "tunnel_guard" ~/.bashrc || cat >> ~/.bashrc <<'EOF'
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash
export TG_DATA=/workspaces/bags
echo "TunnelGuard: откройте вкладку PORTS -> 6080 (рабочий стол, пароль tunnelguard), там терминал: tg ; tg play falcon_test"
EOF
mkdir -p /workspaces/bags
if [ ! -f /workspaces/bags/falcon_test/metadata.yaml ]; then
    python3 -m pip install --quiet zstandard 2>/dev/null || (apt-get update -qq && apt-get install -y -qq python3-pip > /dev/null && python3 -m pip install --quiet zstandard)
    python3 tools/falcon_export_bag.py /workspaces/bags/falcon_test 0 300 || echo "тестовый бэг не создан (нет сети?)"
fi
