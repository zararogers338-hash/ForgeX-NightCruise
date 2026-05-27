#!/usr/bin/env python3
"""Smoke test for the document-processing and output-format path.
This does not call a real model.
"""
from __future__ import annotations

import json
from pathlib import Path
from core.text_processing import extract_text, heuristic_abstract, classify_paper, quality_guard

ROOT = Path(__file__).resolve().parent
sample = ROOT / "examples" / "sample_input_paper.txt"
if not sample.exists():
    raise SystemExit("[FAIL] examples/sample_input_paper.txt missing")

text = extract_text(sample)
abstract = heuristic_abstract(text, 1000)
major, sub = classify_paper(abstract)

fake_output = """
[CHINESE VERSION]
[S1] STUDY OVERVIEW
[S2] METHODS AND DATA
[S3] AUTHOR CLAIMS
[S4] CLAIM–EVIDENCE ALIGNMENT
[S5] UNCERTAINTIES AND LIMITATIONS
[S6] REPRODUCTION REQUIREMENTS
[S7] QA BLOCK — ANTI-HALLUCINATION TRAINING
""" + ("structured output " * 80) + """
[ENGLISH VERSION]
[S1] STUDY OVERVIEW
[S2] METHODS AND DATA
[S3] AUTHOR CLAIMS
[S4] CLAIM–EVIDENCE ALIGNMENT
[S5] UNCERTAINTIES AND LIMITATIONS
[S6] REPRODUCTION REQUIREMENTS
[S7] QA BLOCK — ANTI-HALLUCINATION TRAINING
"""

ok, reasons = quality_guard(fake_output)
if not ok:
    raise SystemExit(f"[FAIL] quality_guard rejected smoke output: {reasons}")

record = {"text": fake_output, "validation": {"score": 9.0, "issues": [], "confidence": "high"}, "discipline": {"major": major, "sub": sub}}
json.dumps(record, ensure_ascii=False)
print("[OK] Smoke test passed")
print(f"[OK] Extracted chars: {len(text)}")
print(f"[OK] Abstract chars: {len(abstract)}")
print(f"[OK] Discipline: {major} / {sub}")
