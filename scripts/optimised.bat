@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
title Horizon Optimised - BTC/ETH coordination (close to stop)
cd /d "D:\Horizon-Incident"

REM Optimised layer: only signals when BTC and ETH agree; flat-bet paper P&L + Kelly hint.
D:\conda\python.exe -m src.realtime.optimised_signal --threshold 0.62 --board-min 30

echo.
echo [stopped] If there is a red error above, please screenshot it.
pause
