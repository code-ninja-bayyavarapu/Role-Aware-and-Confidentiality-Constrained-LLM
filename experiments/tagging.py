"""
Rule-based sensitivity tagging. Reproducible; no ML.
Levels: Public < Internal < Confidential < Restricted.
"""
import re
import csv
import os
from collections import Counter

CLEARANCES_ORDER = ["Public", "Internal", "Confidential", "Restricted"]
LEVEL_INDEX = {c: i for i, c in enumerate(CLEARANCES_ORDER)}

# Lexicons (reproducible)
RESTRICTED_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # phone
    re.compile(r"\b(attorney|confidential|privileged|subpoena|litigation)\b", re.I),
]
CONFIDENTIAL_TERMS = [
    "revenue", "budget", "salary", "margin", "earnings", "project alpha", "project beta",
    "acquisition", "merger", "pricing", "forecast", "financial",
]
INTERNAL_TERMS = [
    "oncall", "incident", "roadmap", "strategy", "sla", "internal", "draft",
    "preliminary", "not for distribution", "need-to-know",
]


def _max_level(text: str) -> str:
    text_lower = (text or "").lower()
    for pat in RESTRICTED_PATTERNS:
        if pat.search(text or ""):
            return "Restricted"
    for t in CONFIDENTIAL_TERMS:
        if t in text_lower:
            return "Confidential"
    for t in INTERNAL_TERMS:
        if t in text_lower:
            return "Internal"
    return "Public"


def doc_label(text: str) -> str:
    return _max_level(text or "")


def segment_label(text: str) -> str:
    return _max_level(text or "")


def build_token_min_clearance(tokenizer) -> dict:
    """token_id -> min clearance name. Decode each token; apply lexicon."""
    vocab = tokenizer.get_vocab()
    token_min = {}
    for tok, idx in vocab.items():
        try:
            s = tokenizer.decode([idx]).strip().lower()
        except Exception:
            s = ""
        level = "Public"
        for pat in RESTRICTED_PATTERNS:
            if pat.search(tokenizer.decode([idx])):
                level = "Restricted"
                break
        if level != "Restricted":
            for t in CONFIDENTIAL_TERMS:
                if t in s:
                    level = "Confidential"
                    break
        if level not in ("Confidential", "Restricted"):
            for t in INTERNAL_TERMS:
                if t in s:
                    level = "Internal"
                    break
        token_min[idx] = level
    return token_min


def clearance_allows(clearance: str, label: str) -> bool:
    return LEVEL_INDEX.get(label, 0) <= LEVEL_INDEX.get(clearance, 0)


def write_manual_review_sample(docs_with_labels: list, output_path: str, n_per_level: int = 10):
    """Write stratified sample for manual review."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    by_level = {}
    for d in docs_with_labels:
        lvl = d.get("label", "Public")
        by_level.setdefault(lvl, []).append(d)
    rows = []
    for lvl in CLEARANCES_ORDER:
        for d in (by_level.get(lvl, []))[:n_per_level]:
            rows.append({"doc_id": d.get("doc_id", ""), "label": lvl, "text_preview": (d.get("text", ""))[:200]})
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "label", "text_preview"])
        w.writeheader()
        w.writerows(rows)
