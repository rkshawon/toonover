# Replaced: Toonover now uses project-local tools on E:\ (no global installs).
# Run scripts\setup_local.bat instead.
Start-Process -FilePath (Join-Path $PSScriptRoot "setup_local.bat") -Wait
