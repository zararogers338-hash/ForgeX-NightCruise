@echo off
chcp 65001 >nul
title ForgeX NightCruise v7.1 Industrial

echo ========================================
echo   ForgeX NightCruise v7.1 Industrial
echo   Synthetic Dataset Factory
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10 or 3.11.
    pause
    exit /b 1
)

echo [INFO] Checking core dependencies...
pip show customtkinter >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies from requirements.txt ...
    pip install -r requirements.txt
)

echo [INFO] Checking GPU visibility...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARN] NVIDIA GPU not detected by nvidia-smi. NightCruise can still run in CPU/API mode.
) else (
    echo [OK] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

echo.
echo [INFO] Launching NightCruise...
echo.
python main.py

pause
