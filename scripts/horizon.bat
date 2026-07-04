@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
title Horizon Signal - close this window to stop
cd /d "%~dp0.."

REM 选 Python：本机装了依赖的解释器优先（默认 conda），否则用 PATH 里的 python
set "PY=python"
if exist "D:\conda\python.exe" set "PY=D:\conda\python.exe"

REM Foreground run: closing this window stops the signal.
"%PY%" -m src.realtime.live_signal_v4 --threshold 0.66 --board-min 30

echo.
echo [stopped] If there is a red error above, please screenshot it.
pause
