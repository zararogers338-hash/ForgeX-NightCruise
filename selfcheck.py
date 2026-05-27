#!/usr/bin/env python3
"""Lightweight self-check for ForgeX NightCruise.
This does not start the GUI and does not run an LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    ROOT / "main.py",
    ROOT / "core" / "gpu_monitor.py",
    ROOT / "core" / "inference.py",
    ROOT / "core" / "text_processing.py",
    ROOT / "core" / "prompts.py",
    ROOT / "requirements.txt",
]

missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
if missing:
    print("[FAIL] Missing files:", ", ".join(missing))
    sys.exit(1)

from core.prompts import VERSION, DEFAULT_SYSTEM_PROMPT
from core.text_processing import SUPPORTED_EXTS, classify_paper, heuristic_abstract, quality_guard
from core.gpu_monitor import GPUMonitor

if "[CHINESE VERSION]" not in DEFAULT_SYSTEM_PROMPT or "[ENGLISH VERSION]" not in DEFAULT_SYSTEM_PROMPT:
    print("[FAIL] Prompt is missing bilingual requirements")
    sys.exit(1)

sample = "Abstract: This note describes a machine learning workflow for local AI dataset generation."
abstract = heuristic_abstract(sample, 500)
major, sub = classify_paper(abstract)

monitor = GPUMonitor(poll_interval=10)
monitor.poll_once()

print("[OK] ForgeX NightCruise self-check passed")
print(f"[OK] Version: {VERSION}")
print(f"[OK] Supported extensions: {', '.join(sorted(SUPPORTED_EXTS))}")
print(f"[OK] Sample classification: {major} / {sub}")
print(f"[OK] Monitor backend: {monitor.backend}")
