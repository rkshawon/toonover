@echo off
REM ===================================================================
REM  Toonover - project-local setup (nothing installed globally)
REM  Tools live in   E:\ProgramData\Toonover
REM  Downloads in    E:\ToonoverDownloads
REM  Safe to run again any time (e.g. after the big downloads finish).
REM ===================================================================
setlocal EnableExtensions
set "TOOLS=E:\ProgramData\Toonover"
set "DL=E:\ToonoverDownloads"
set "PROJ=%~dp0.."
set "PIP_CACHE_DIR=%TOOLS%\cache\pip"
set "TEMP=%TOOLS%\tmp"
set "TMP=%TOOLS%\tmp"
set "PYTHONNOUSERSITE=1"
if not exist "%TEMP%" mkdir "%TEMP%"

if not exist "%TOOLS%\python\python.exe" (
  echo [X] Portable Python not found in %TOOLS%\python
  goto :end
)

echo.
echo == 1/4  Python virtual environment: %TOOLS%\venv
if not exist "%TOOLS%\venv\Scripts\python.exe" "%TOOLS%\python\python.exe" -m venv "%TOOLS%\venv"

echo.
echo == 2/4  Installing Toonover packages offline from %DL%\wheels
"%TOOLS%\venv\Scripts\python.exe" -m pip install -q --no-index --find-links "%DL%\wheels" --upgrade pip
"%TOOLS%\venv\Scripts\python.exe" -m pip install -q --no-index --find-links "%DL%\wheels" -e "%PROJ%"
if errorlevel 1 (
  echo [X] Package install failed
  goto :end
)
echo     OK

echo.
echo == 3/4  ComfyUI for AMD
call :find_comfy
if defined CDIR (
  echo     already extracted: %CDIR%
) else if exist "%DL%\ComfyUI_windows_portable_amd.7z" (
  echo     extracting ComfyUI_windows_portable_amd.7z - takes a few minutes ...
  "%TOOLS%\7zip\7zr.exe" x "%DL%\ComfyUI_windows_portable_amd.7z" -o"%TOOLS%\ComfyUI" -y >nul
  call :find_comfy
) else (
  echo     not downloaded yet - download it with Brave into %DL% and run this again
)

echo.
echo == 4/4  Models
if not defined CDIR goto :models_skip
set "MD=%CDIR%\ComfyUI\models"
if exist "%DL%\sd_xl_base_1.0.safetensors" move /Y "%DL%\sd_xl_base_1.0.safetensors" "%MD%\checkpoints\" >nul
if exist "%DL%\diffusion_pytorch_model_promax.safetensors" move /Y "%DL%\diffusion_pytorch_model_promax.safetensors" "%MD%\controlnet\controlnet-union-sdxl-1.0.safetensors" >nul
for %%F in ("%DL%\*.safetensors") do move /Y "%%F" "%MD%\loras\" >nul
if exist "%MD%\checkpoints\sd_xl_base_1.0.safetensors" (echo     [OK] SDXL base) else (echo     [ ] SDXL base - still missing)
if exist "%MD%\controlnet\controlnet-union-sdxl-1.0.safetensors" (echo     [OK] ControlNet union) else (echo     [ ] ControlNet union - still missing)
goto :models_done
:models_skip
echo     skipped - ComfyUI not extracted yet
:models_done

echo.
"%TOOLS%\venv\Scripts\python.exe" -m toonover styles
echo.
echo Done. Use  toonover.bat  in the project folder, and  start_comfyui.bat  for GPU styles.
goto :end

:find_comfy
set "CDIR="
for /d %%D in ("%TOOLS%\ComfyUI\*") do if exist "%%D\run_amd_gpu.bat" set "CDIR=%%D"
exit /b 0

:end
echo.
pause
