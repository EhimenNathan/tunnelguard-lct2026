#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash
export DISPLAY=${DISPLAY:-:99}
cleanup() {
    trap - EXIT INT TERM
    tg stop || true
    jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM
Xvfb "$DISPLAY" -screen 0 1600x900x24 -nolisten tcp &
for i in $(seq 1 100); do
    xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && break
    sleep 0.1
done
xdpyinfo -display "$DISPLAY" >/dev/null
openbox >/tmp/openbox.log 2>&1 &
umask 077
x11vnc -storepasswd "$TG_VNC_PASSWORD" /tmp/tg-vnc.pass >/dev/null
x11vnc -display "$DISPLAY" -rfbauth /tmp/tg-vnc.pass -forever -shared -localhost -rfbport 5900 >/tmp/x11vnc.log 2>&1 &
websockify --web=/usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
if [ "${TG_GUI_AUTOSTART:-1}" = 1 ]; then tg start true; fi
cat > /tmp/tg-bashrc <<'EOF'
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash
export TG_DATA=/data
echo 'tg bags; tg play /data/ИМЯ_БЭГА 1.0'
EOF
xterm -geometry 100x18+20+600 -fa 'DejaVu Sans Mono' -fs 11 -title 'TunnelGuard — tg play /data/бэг 1.0' \
    -e bash --rcfile /tmp/tg-bashrc &
echo 'TunnelGuard GUI: http://localhost:6080/vnc.html (пароль из TG_VNC_PASSWORD)'
wait -n
