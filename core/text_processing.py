# -*- coding: utf-8 -*-
"""
Text Processing - Document extraction, classification, and quality control
"""

import re
import json
from pathlib import Path
from typing import Tuple, List, Dict
from collections import Counter

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".doc", ".json", ".jsonl", ".html", ".htm"}

# Optional imports
try:
    import textract
    HAS_TEXTRACT = True
except ImportError:
    HAS_TEXTRACT = False

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx as python_docx
except ImportError:
    python_docx = None


# ====================== Discipline Keywords ======================
DISCIPLINES = {
    "Natural Sciences": {
        "Mathematics": ["math", "mathematics", "algebra", "geometry", "calculus", "statistics", "probability", "optimization", "graph theory"],
        "Physics": ["physics", "quantum", "relativity", "particle", "thermodynamics", "mechanics", "optics", "astrophysics", "condensed matter"],
        "Chemistry": ["chemistry", "organic", "inorganic", "physical chemistry", "biochemistry", "polymer", "catalysis", "spectroscopy"],
        "Biology": ["biology", "genetics", "evolution", "ecology", "cell", "molecular biology", "microbiology", "neuroscience", "botany"],
        "Medicine": ["medicine", "clinical", "disease", "therapy", "pharmacology", "pathology", "surgery", "epidemiology", "immunology"],
        "Computer Science": ["computer", "algorithm", "ai", "machine learning", "deep learning", "software", "network", "database", "cybersecurity", "llm", "transformer", "neural network"],
        "Engineering": ["engineering", "mechanical", "electrical", "civil", "chemical engineering", "robotics", "materials science", "aerospace"],
        "Environmental Science": ["environment", "climate", "sustainability", "pollution", "ecosystem", "conservation", "renewable energy"],
        "Geosciences": ["geology", "earth science", "seismology", "oceanography", "meteorology", "geophysics"],
        "Astronomy": ["astronomy", "cosmology", "planet", "galaxy", "space", "telescope", "exoplanet"],
    },
    "Social Sciences": {
        "Economics": ["economy", "economic", "market", "finance", "trade", "macroeconomics", "microeconomics", "development economics"],
        "Psychology": ["psychology", "cognitive", "behavior", "mental health", "psychotherapy", "social psychology", "developmental psychology"],
        "Sociology": ["sociology", "society", "social", "culture", "inequality", "urban sociology", "family"],
        "Political Science": ["politics", "government", "policy", "international relations", "democracy", "election", "governance"],
        "Law": ["law", "legal", "justice", "rights", "constitution", "criminal law", "international law"],
        "Education": ["education", "teaching", "learning", "school", "curriculum", "pedagogy", "educational psychology"],
        "Anthropology": ["anthropology", "cultural anthropology", "ethnography", "archaeology", "ethnology"],
        "History": ["history", "historical", "ancient", "modern history", "medieval", "archival", "historiography"],
        "Linguistics": ["linguistics", "language", "syntax", "phonetics", "semantics", "sociolinguistics"],
        "Philosophy": ["philosophy", "ethics", "logic", "metaphysics", "epistemology", "aesthetics"],
        "Geography": ["geography", "human geography", "urban", "gis", "regional", "physical geography"],
        "Demography": ["population", "demography", "migration", "fertility", "mortality"],
        "Communication": ["communication", "media", "journalism", "public relations", "digital media"],
        "Management": ["management", "business", "organization", "leadership", "strategy", "marketing"],
        "Library Science": ["library", "information science", "bibliometrics", "archival science"],
        "Criminology": ["criminology", "crime", "deviance", "penology"],
    }
}


def extract_text(path: Path) -> str:
    """Extract text from various document formats"""
    suf = path.suffix.lower()
    try:
        if suf in (".txt", ".md", ".json", ".jsonl", ".html", ".htm"):
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        if suf == ".pdf":
            if PdfReader is None:
                return "[ERROR] pypdf not installed - pip install pypdf"
            reader = PdfReader(str(path))
            text = "\n\n".join([pg.extract_text() or "" for pg in reader.pages])
            return text.strip() or "[ERROR] PDF empty"
        if suf == ".docx":
            if python_docx is None:
                return "[ERROR] python-docx not installed - pip install python-docx"
            d = python_docx.Document(str(path))
            text = "\n".join([p.text for p in d.paragraphs if p.text.strip()])
            return text.strip() or "[ERROR] DOCX empty"
        if suf == ".doc":
            if HAS_TEXTRACT:
                return textract.process(str(path), encoding="utf-8").decode("utf-8").strip()
            return "[ERROR] textract not installed for .doc"
    except Exception as e:
        return f"[ERROR] {suf.upper()} extract failed: {e}"
    return f"[ERROR] unsupported {suf}"


def heuristic_abstract(text: str, max_chars: int) -> str:
    """Extract abstract or first portion of text"""
    t = re.sub(r"\r\n?", "\n", text)
    patterns = [
        r"(?is)\babstract\b\s*[:\-]?\s*(.+?)(?=\n\s*\b(introduction|background|methods?|materials|results?|conclusions?|keywords?)\b|\Z)",
        r"(?is)\b摘要\b\s*[:：\-]?\s*(.+?)(?=\n\s*\b(引言|背景|方法|材料|结果|结论|关键词)\b|\Z)",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            s = re.sub(r"\n{3,}", "\n\n", m.group(1).strip())
            return s[:max_chars]
    return t[:max_chars]


def classify_paper(text: str) -> Tuple[str, str]:
    """Classify paper by discipline"""
    if not text:
        return "Unknown", "Unknown"
    text_lower = text.lower()
    best_sub = best_major = "Unknown"
    best_score = 0
    for major, subs in DISCIPLINES.items():
        for sub, keywords in subs.items():
            score = sum(keyword in text_lower for keyword in keywords)
            if score > best_score:
                best_score = score
                best_sub = sub
                best_major = major
    return best_major, best_sub


def quality_guard(output: str) -> Tuple[bool, List[str]]:
    """Check output quality and format compliance"""
    reasons = []
    if not output or len(output.strip()) < 800:
        reasons.append("output_too_short")
    if "[CHINESE VERSION]" not in output or "[ENGLISH VERSION]" not in output:
        reasons.append("missing_bilingual_blocks")
    required = ["[S1]", "[S2]", "[S3]", "[S4]", "[S5]", "[S6]", "[S7]"]
    missing = [s for s in required if s not in output]
    if missing:
        reasons.extend([f"missing_section_{s.strip('[]')}" for s in missing])
    return len(reasons) == 0, reasons


def parse_validator_output(output: str) -> Dict:
    """Parse validator JSON output"""
    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError
        data = json.loads(output[start:end])
        score = float(data.get("score", 0.0))
        issues = data.get("issues", [])
        confidence = data.get("confidence", "low" if score < 6 else "medium" if score < 8 else "high")
        return {"score": score, "issues": issues, "confidence": confidence}
    except Exception:
        return {"score": 0.0, "issues": ["Validator JSON parse failed"], "confidence": "low"}
