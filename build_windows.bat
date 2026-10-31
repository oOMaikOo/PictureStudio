@echo off
REM Build Picture Studio for Windows (PyInstaller + optional Inno Setup).
REM Prerequisites: Python venv with project deps, PyInstaller, Inno Setup 6 (optional).

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Create .venv and install requirements first.
  exit /b 1
)

echo === PyInstaller ===
.venv\Scripts\python.exe -m PyInstaller --noconfirm Picture.spec
if errorlevel 1 exit /b 1

if exist "installer\windows\PictureStudio.iss" (
  where iscc >nul 2>&1
  if not errorlevel 1 (
    echo === Inno Setup ===
    iscc "installer\windows\PictureStudio.iss"
  ) else (
    echo Inno Setup (iscc) not on PATH — skipping installer. Dist folder is ready under dist\.
  )
) else (
  echo No Inno script found — PyInstaller output only.
)

echo Done.
endlocal
