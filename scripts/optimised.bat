@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
title Horizon Optimised - BTC/ETH coordination (close to stop)
cd /d "%~dp0.."

REM 选 Python：本机装了依赖的解释器优先（默认 conda），否则用 PATH 里的 python
set "PY=python"
if exist "D:\conda\python.exe" set "PY=D:\conda\python.exe"

REM Optimised layer: only signals when BTC and ETH agree; flat-bet paper P&L + Kelly hint.
"%PY%" -m src.realtime.optimised_signal --threshold 0.62 --board-min 30

echo.
echo [stopped] If there is a red error above, please screenshot it.
pause
