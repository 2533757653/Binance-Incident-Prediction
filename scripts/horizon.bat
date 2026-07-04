@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
title Horizon Signal - close this window to stop
cd /d "D:\Horizon-Incident"

REM Foreground run: closing this window stops the signal.
D:\conda\python.exe -m src.realtime.live_signal_v4 --threshold 0.66 --board-min 30

echo.
echo [stopped] If there is a red error above, please screenshot it.
pause
