@echo off
REM Run Toonover with the project-local tools on E:\ (no global install).
REM   toonover.bat make "https://youtube.com/shorts/..." -s cartoon
REM   toonover.bat serve
set "TOOLS=E:\ProgramData\Toonover"
set "PATH=%TOOLS%\ffmpeg\bin;%TOOLS%\venv\Scripts;%PATH%"
set "PIP_CACHE_DIR=%TOOLS%\cache\pip"
set "TEMP=%TOOLS%\tmp"
set "TMP=%TOOLS%\tmp"
set "PYTHONNOUSERSITE=1"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%TOOLS%\venv\Scripts\python.exe" (
  echo Run scripts\setup_local.bat first.
  exit /b 1
)
"%TOOLS%\venv\Scripts\python.exe" -m toonover %*
