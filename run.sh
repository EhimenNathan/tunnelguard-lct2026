#!/bin/bash
# Start TunnelGuard with RViz in Docker (Linux X11, or WSL2 Ubuntu with WSLg on Windows).
#   ./run.sh /path/to/bags            -> detector + RViz + console; then:  tg bags ; tg play <bag>
#   ./run.sh /path/to/bags --no-gui   -> no RViz (headless machine)
# Requires the image: docker build -t tunnel_guard .
set -e
BAGS=$(realpath "${1:-$PWD}")
GUI=1; [ "$2" = "--no-gui" ] && GUI=0
ARGS=(--rm -it --net=host --ipc=host -v "$BAGS:/data")
if [ $GUI = 1 ]; then
    if [ -d /mnt/wslg ]; then            # Windows 11 / WSL2: WSLg provides X11 and Wayland sockets
        ARGS+=(-e DISPLAY=:0 -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
               -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg)
    else                                  # Linux desktop (X11)
        xhost +local:docker >/dev/null 2>&1 || true
        ARGS+=(-e DISPLAY="${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix)
    fi
    [ -e /dev/dri ] && ARGS+=(--device /dev/dri)     # GPU rendering for RViz when available
fi
exec docker run "${ARGS[@]}" tunnel_guard tg
