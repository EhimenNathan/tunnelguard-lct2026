# Start TunnelGuard with RViz from Windows PowerShell (Docker Desktop, WSL2 engine).
#   .\run_windows.ps1 C:\path\to\bags
# The GUI is shown through WSLg (Windows 11 or Windows 10 with WSL 2 + WSLg): the container is started from the default
# WSL distribution, which owns the WSLg display.  Without WSL, start an X server (VcXsrv: "Disable access control") and
# use:  .\run_windows.ps1 C:\path\to\bags -X
param([string]$Bags = (Get-Location).Path, [switch]$X)
$Bags = (Resolve-Path $Bags).Path
if ($X) {
    docker run --rm -it -e DISPLAY=host.docker.internal:0.0 -e LIBGL_ALWAYS_INDIRECT=0 -v "${Bags}:/data" tunnel_guard tg
    exit $LASTEXITCODE
}
$drive = $Bags.Substring(0, 1).ToLower()
$wslPath = "/mnt/$drive" + ($Bags.Substring(2) -replace '\\', '/')
wsl.exe -e bash -lc "docker run --rm -it -e DISPLAY=:0 -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg -v '$wslPath':/data tunnel_guard tg"
