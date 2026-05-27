#!/usr/bin/env bash
set -e

echo "========================================"
echo "  ForgeX NightCruise v7.1 Industrial"
echo "  Synthetic Dataset Factory"
echo "========================================"
echo ""

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[ERROR] Python not found. Please install Python 3.10 or 3.11."
  exit 1
fi

echo "[INFO] Using: $($PYTHON_BIN --version)"
echo "[INFO] Checking core dependencies..."
if ! $PYTHON_BIN -c "import customtkinter" >/dev/null 2>&1; then
  echo "[INFO] Installing dependencies from requirements.txt ..."
  $PYTHON_BIN -m pip install -r requirements.txt
fi

echo "[INFO] Checking GPU visibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[OK] NVIDIA GPU detected:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
elif [[ "$(uname -m)" == "arm64" && "$(uname)" == "Darwin" ]]; then
  echo "[OK] Apple Silicon detected. Metal may be available for supported backends."
else
  echo "[WARN] No GPU detected. NightCruise can still run in CPU/API mode."
fi

echo ""
echo "[INFO] Launching NightCruise..."
$PYTHON_BIN main.py
