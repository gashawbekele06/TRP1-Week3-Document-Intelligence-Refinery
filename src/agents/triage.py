"""
Stage 1: The Triage Agent (Document Classifier)
Produces DocumentProfile with all five required dimensions.
Uses pdfplumber heuristics only - no external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import sys
from pathlib import Path

# Add project root to PYTHONPATH (temporary fix)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


from typing import Dict, List, Optional, Tuple

import pdfplumber
import yaml

from src.models.document_profile import DocumentProfile

# ─── Paths ────────────────────────────────────────────────────────────────
RULES_PATH = Path(__file__).parent.parent.parent / "rubric" / "extraction_rules.yaml"
PROFILES_DIR = Path(".refinery") / "profiles"


def load_rules() -> dict:
    """Load configurable thresholds from yaml."""
    if not RULES_PATH.exists():
        raise FileNotFoundError(f"Rules file not found: {RULES_PATH}")
    with open(RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_doc_id(file_path: Path) -> str:
    """Deterministic short ID: hash of absolute path + size."""
    raw = f"{file_path.resolve()}:{file_path.stat().st_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ─── Signal Collection Helpers ─────────────────────────────────────────────

def char_density(page) -> float:
    """Characters per 1000 pt²."""
    try:
        text = page.extract_text() or ""
        area = (page.width or 1) * (page.height or 1) / 1000.0
        return len(text) / area if area > 0 else 0.0
    except:
        return 0.0


def image_ratio(page) -> float:
    """Fraction of page area covered by images."""
    try:
        page_area = (page.width or 1) * (page.height or 1)
        img_area = sum(
            (im.get("x1", 0) - im.get("x0", 0)) * (im.get("y1", 0) - im.get("y0", 0))
            for im in (page.images or [])
        )
        return min(img_area / page_area, 1.0) if page_area > 0 else 0.0
    except:
        return 0.0


def has_font_metadata(pdf) -> bool:
    """True if any page has font information."""
    try:
        for page in pdf.pages[:5]:
            if page.chars and any("fontname" in c for c in page.chars):
                return True
    except:
        pass
    return False


def is_form_fillable(pdf) -> bool:
    """Detect AcroForm or XFA forms."""
    try:
        trailer = pdf.doc.trailer
        root = trailer.get("/Root", {})
        if isinstance(root, dict) and (root.get("/AcroForm") or root.get("/XFA")):
            return True
    except:
        pass
    return False


def detect_language_heuristic(text: str) -> Tuple[str, float]:
    """Amharic + English focused heuristic."""
    if not text.strip():
        return "und", 0.0

    amharic = sum(1 for c in text if 0x1200 <= ord(c) <= 0x139F)
    latin = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    total = amharic + latin or 1

    am_ratio = amharic / total
    en_ratio = latin / total

    if am_ratio > 0.45:
        return "am", min(0.95, 0.55 + am_ratio * 0.45)
    if en_ratio > 0.70:
        return "en", min(0.92, 0.50 + en_ratio * 0.45)
    if am_ratio > 0.15 or en_ratio > 0.35:
        return "mixed", 0.65
    return "und", 0.30


def infer_domain(text: str, rules: dict) -> str:
    """Keyword-based domain hint."""
    text_lower = text.lower()
    scores: Dict[str, int] = {d: 0 for d in rules.get("domain_keywords", {})}

    for domain, keywords in rules.get("domain_keywords", {}).items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                scores[domain] += 1

    if not any(scores.values()):
        return "general"

    best = max(scores, key=scores.get)
    return best


# ─── Classification Functions ──────────────────────────────────────────────

def detect_origin_type(
    densities: List[float],
    ratios: List[float],
    has_fonts: bool,
    is_form: bool,
    rules: dict
) -> str:
    if is_form:
        return "form_fillable"

    mean_density = sum(densities) / max(1, len(densities))
    mean_image = sum(ratios) / max(1, len(ratios))

    cfg = rules["origin_detection"]
    if mean_density < cfg["scanned_max_char_density"] * 0.25 and not has_fonts:
        return "scanned_image"
    if mean_image >= cfg["scanned_min_image_ratio"]:
        return "scanned_image"
    if mean_density < cfg["scanned_max_char_density"] and mean_image > cfg["scanned_min_image_ratio"]:
        return "scanned_image"
    if cfg["mixed_image_ratio_lower"] <= mean_image < cfg["scanned_min_image_ratio"]:
        return "mixed"
    if has_fonts and mean_image < cfg["mixed_image_ratio_lower"]:
        return "native_digital"
    if mean_density >= cfg["digital_min_char_density"]:
        return "native_digital"
    return "mixed"


def detect_layout_complexity(
    sample_pages: List,
    rules: dict
) -> str:
    table_area_total = 0.0
    figure_area_total = 0.0
    page_area_total = 0.0
    x_centers: List[float] = []

    for page in sample_pages:
        try:
            w, h = page.width or 600, page.height or 800
            page_area = w * h
            page_area_total += page_area

            # Tables
            for t in page.find_tables() or []:
                if bbox := t.bbox:
                    table_area_total += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            # Images / figures
            for im in page.images or []:
                try:
                    figure_area_total += (im["x1"] - im["x0"]) * (im["y1"] - im["y0"])
                except:
                    pass

            # Column detection via x-centers
            words = page.extract_words() or []
            for w in words:
                if x0 := w.get("x0"):
                    if x1 := w.get("x1"):
                        x_centers.append((x0 + x1) / 2)

        except:
            continue

    table_ratio = table_area_total / max(page_area_total, 1)
    figure_ratio = figure_area_total / max(page_area_total, 1)

    cfg = rules["layout_detection"]
    if table_ratio > cfg["table_heavy_min_ratio"]:
        return "table_heavy"
    if figure_ratio > cfg["figure_heavy_min_ratio"]:
        return "figure_heavy"

    # Simple column count via x-gaps
    if x_centers and len(sample_pages) > 0:
        page_w = sample_pages[0].width or 600
        sorted_x = sorted(x_centers)
        gaps = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x)-1)]
        big_gaps = sum(1 for g in gaps if g > page_w * 0.15)
        if big_gaps >= cfg["multi_column_min_clusters"] - 1:
            return "multi_column"

    return "single_column"


# ─── Main Triage Logic ─────────────────────────────────────────────────────

class TriageAgent:
    def __init__(self):
        self.rules = load_rules()
        self.profiles_dir = PROFILES_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def triage(self, file_path: str | Path) -> DocumentProfile:
        file_path = Path(file_path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = compute_doc_id(file_path)
        profile_path = self.profiles_dir / f"{doc_id}.json"

        # Return cached result if exists
        if profile_path.exists():
            return DocumentProfile.model_validate_json(profile_path.read_text())

        start = time.time()

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                raise ValueError("PDF has no pages")

            # Smart sampling: start + middle + end
            indices = [0, page_count // 3, page_count // 2, page_count - 1]
            sample_pages = [pdf.pages[i] for i in set(indices) if i < page_count]

            # Collect signals
            densities = [char_density(p) for p in sample_pages]
            img_ratios = [image_ratio(p) for p in sample_pages]
            has_fonts = has_font_metadata(pdf)
            is_form = is_form_fillable(pdf)

            # Text sample for domain & language
            text_sample = " ".join(
                (p.extract_text() or "")[:600]
                for p in sample_pages
            )

            origin_type = detect_origin_type(densities, img_ratios, has_fonts, is_form, self.rules)
            layout_complexity = detect_layout_complexity(sample_pages, self.rules)
            language_code, language_confidence = detect_language_heuristic(text_sample)
            domain_hint = infer_domain(text_sample, self.rules)
            estimated_extraction_cost = (
                "fast_text_sufficient"
                if origin_type == "native_digital" and layout_complexity == "single_column"
                else "needs_vision_model" if origin_type in ("scanned_image", "form_fillable") else
                "needs_layout_model"
            )

            mean_density = sum(densities) / max(1, len(densities))
            mean_image = sum(img_ratios) / max(1, len(img_ratios))

            profile = DocumentProfile(
                doc_id=doc_id,
                filename=file_path.name,
                file_path=str(file_path),
                page_count=page_count,
                origin_type=origin_type,
                layout_complexity=layout_complexity,
                language_code=language_code,
                language_confidence=language_confidence,
                domain_hint=domain_hint,
                estimated_extraction_cost=estimated_extraction_cost,
                char_density_mean=round(mean_density, 3),
                image_ratio_mean=round(mean_image, 3),
                table_count_total=0,  # can be extended
                has_font_metadata=has_fonts,
                is_form_fillable=is_form,
            )

        # Save
        profile_path.write_text(profile.model_dump_json(indent=2))

        duration = time.time() - start
        print(f"✓ {file_path.name} → {doc_id[:8]}... ({duration:.1f}s)")
        print(f"  {origin_type:16} | {layout_complexity:16} | {estimated_extraction_cost}")
        print(f"  lang: {language_code} ({language_confidence:.2f}) | domain: {domain_hint}")

        return profile


def main():
    agent = TriageAgent()
    for pdf in Path("data").rglob("*.pdf"):
        try:
            agent.triage(pdf)
        except Exception as e:
            print(f"✗ {pdf.name}: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()