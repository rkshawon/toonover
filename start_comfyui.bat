@echo off
REM Start ComfyUI (AMD ROCm) from E:\ProgramData\Toonover\ComfyUI, keeping its caches on E:\
setlocal
set "TOOLS=E:\ProgramData\Toonover"
set "CACHE=%TOOLS%\cache"
set "TEMP=%TOOLS%\tmp"
set "TMP=%TOOLS%\tmp"
set "HF_HOME=%CACHE%\huggingface"
set "TORCH_HOME=%CACHE%\torch"
set "MIOPEN_USER_DB_PATH=%CACHE%\miopen"
set "MIOPEN_CUSTOM_CACHE_DIR=%CACHE%\miopen"
set "TRITON_CACHE_DIR=%CACHE%\triton"
set "PYTHONNOUSERSITE=1"
if not exist "%TEMP%" mkdir "%TEMP%"
set "CDIR="
for /d %%D in ("%TOOLS%\ComfyUI\*") do if exist "%%D\run_amd_gpu.bat" set "CDIR=%%D"
if not defined CDIR (
  echo ComfyUI not found. Download ComfyUI_windows_portable_amd.7z into E:\ToonoverDownloads
  echo and run scripts\setup_local.bat
  pause
  exit /b 1
)
cd /d "%CDIR%"
call run_amd_gpu.bat
