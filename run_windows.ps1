# Browser desktop with RViz using the pinned release.
param([Parameter(Mandatory=$true)][string]$Bags)
$ErrorActionPreference = 'Stop'
$resolvedBags = (Resolve-Path -LiteralPath $Bags).Path
if (-not (Test-Path -LiteralPath $resolvedBags -PathType Container)) {
    throw 'Bags must point to an existing directory of extracted ROS 2 recordings.'
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Start Docker Desktop with Linux containers, or use the documented Codespaces option.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:TG_BAGS_DIR = $resolvedBags
    $dockerOs = docker info --format '{{.OSType}}'
    if ($LASTEXITCODE -ne 0 -or $dockerOs -ne 'linux') { throw 'Docker Engine with Linux containers is required.' }
    docker compose -f compose.repro.yaml config --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Compose configuration failed.' }
    docker compose -f compose.repro.yaml pull
    if ($LASTEXITCODE -ne 0) { throw 'Image download failed. Check network and disk space.' }
    docker compose -f compose.repro.yaml up -d
    if ($LASTEXITCODE -ne 0) { throw 'Container startup failed. Run docker compose -f compose.repro.yaml logs.' }
    Write-Host 'RViz desktop: http://localhost:6080/vnc.html (password: tunnelguard, unless TG_VNC_PASSWORD is set).'
    Write-Host 'Console: docker compose -f compose.repro.yaml exec tunnelguard bash'
}
finally { Pop-Location }
