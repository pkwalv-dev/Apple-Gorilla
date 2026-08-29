@echo off
title Apple-Gorilla
setlocal

REM ============================================================================
REM  Apple-Gorilla one-click launcher (Windows).
REM  Double-click to open AG's web app in your browser. You can copy this file
REM  anywhere (e.g. your Desktop) - the first run asks where the AG folder is and
REM  remembers it in AG_path.txt next to this launcher.
REM ============================================================================

set "AGDIR="

REM 1) launcher sitting inside the AG repo
if exist "%~dp0ag\__init__.py" (
  set "AGDIR=%~dp0"
REM 2) an AG clone sitting next to the launcher
) else if exist "%~dp0Apple-Gorilla\ag\__init__.py" (
  set "AGDIR=%~dp0Apple-Gorilla"
REM 3) a path we remembered from a previous run
) else if exist "%~dp0AG_path.txt" (
  set /p AGDIR=<"%~dp0AG_path.txt"
)

REM 4) ask once if we still don't know
if not defined AGDIR (
  echo.
  echo   Could not find the Apple-Gorilla folder next to this launcher.
  echo   Tip: drag the AG folder onto this window, then press Enter.
  echo.
  set /p "AGDIR=Paste the full path to your Apple-Gorilla folder: "
)

REM strip any surrounding quotes a drag-and-drop may have added
set AGDIR=%AGDIR:"=%

if not exist "%AGDIR%\ag\__init__.py" (
  echo.
  echo   That folder does not contain AG ^(no ag\__init__.py^):
  echo     %AGDIR%
  echo   Delete AG_path.txt next to this launcher and try again.
  echo.
  pause
  exit /b 1
)

REM remember the location for next time
> "%~dp0AG_path.txt" echo %AGDIR%
cd /d "%AGDIR%"

REM pick a Python: prefer the "py" launcher, fall back to "python"
set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo.
  echo   Python was not found. Install it from https://www.python.org/downloads/
  echo   and tick "Add python.exe to PATH" during setup, then run this again.
  echo.
  pause
  exit /b 1
)

echo Checking dependencies ^(first run may take a moment^)...
%PY% -m pip install -q -r requirements.txt >nul 2>nul

echo.
echo   Apple-Gorilla is starting - a browser tab will open.
echo   Keep this window open while you use AG; close it to stop.
echo.
%PY% -m ag serve --open

echo.
echo   AG has stopped.
pause
