"""src.agents.triage
====================

Module purpose
--------------
This module implements the Stage 1 **Triage Agent**: a lightweight classifier
that inspects a PDF and emits a :class:`src.models.document_profile.DocumentProfile`.

The profile describes the document along the rubric-required dimensions:

- ``origin_type`` (native vs scanned vs mixed vs form)
- ``layout_complexity`` (single/multi-column, table/figure heavy)
- ``language_code`` + confidence
- ``domain_hint``
- ``estimated_extraction_cost`` (Strategy A/B/C routing hint)

Why this exists
---------------
Real-world “native” PDFs (annual reports, statements) often contain many images
(logos, charts). If we treat “image-heavy” as “scanned”, we misroute digital
documents to vision unnecessarily.

This triage implementation therefore:

- prioritizes **digital signals** (extractable text density and font metadata)
  before considering image coverage;
- uses **spread sampling** across the document (not just first/last pages);
- invalidates cached profiles when the triage logic version changes.

Testing hooks
-------------
Unit tests monkeypatch three wrapper helpers (kept intentionally stable):

- :func:`_compute_char_density`
- :func:`_compute_image_ratio`
- :func:`_has_font_metadata`
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
import sys
# Add project root to PYTHONPATH (temporary fix)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from typing import Dict, List, Tuple

import yaml

from src.models.document_profile import DocumentProfile

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None


TRIAGE_VERSION = "1.7"

# ─── Paths ────────────────────────────────────────────────────────────────
RULES_PATH = Path(__file__).parent.parent.parent / "rubric" / "extraction_rules.yaml"
PROFILES_DIR = Path(".refinery") / "profiles"


def load_rules() -> dict:
    """Load configurable thresholds from the rubric YAML."""
    if not RULES_PATH.exists():
        raise FileNotFoundError(f"Rules file not found: {RULES_PATH}")
    with open(RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_doc_id(file_path: Path) -> str:
    """Compute a deterministic short document id (truncated SHA-256)."""
    raw = f"{file_path.resolve()}:{file_path.stat().st_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2)


def _safe_mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _sample_page_indices(page_count: int, max_samples: int) -> List[int]:
    """Return spread indices across the document.

    We intentionally include more interior pages to avoid cover/back-cover bias
    (common in annual reports).
    """
    if page_count <= 0:
        return []

    max_samples = max(1, int(max_samples))
    if page_count <= max_samples:
        return list(range(page_count))

    # Spread points in [0, page_count-1]
    step = (page_count - 1) / float(max_samples - 1)
    indices = {int(round(i * step)) for i in range(max_samples)}

    # Nudge away from edge-only sampling for longer docs
    if page_count >= 10:
        indices.update({1, 2, page_count - 2, page_count - 3})

    return sorted(i for i in indices if 0 <= i < page_count)


# ─── Signal Collection Helpers ─────────────────────────────────────────────

def char_density(page) -> float:
    """Estimate character density on a page (chars per 1000 pt²)."""
    try:
        text = page.extract_text() or ""
        area = (page.width or 1) * (page.height or 1) / 1000.0
        return len(text) / area if area > 0 else 0.0
    except Exception:
        return 0.0


def image_ratio(page) -> float:
    """Estimate fraction of page area covered by images (0.0–1.0)."""
    try:
        page_area = (page.width or 1) * (page.height or 1)
        img_area = sum(
            (im.get("x1", 0) - im.get("x0", 0)) * (im.get("y1", 0) - im.get("y0", 0))
            for im in (page.images or [])
        )
        return min(img_area / page_area, 1.0) if page_area > 0 else 0.0
    except Exception:
        return 0.0


def has_font_metadata(pdf) -> bool:
    """Detect embedded font metadata (strong signal of native digital text)."""
    try:
        for page in pdf.pages[:5]:
            if page.chars and any("fontname" in c for c in page.chars):
                return True
    except Exception:
        pass
    return False


def is_form_fillable(pdf) -> bool:
    """Detect AcroForm/XFA interactive form metadata."""
    try:
        trailer = pdf.doc.trailer
        root = trailer.get("/Root", {})
        if isinstance(root, dict) and (root.get("/AcroForm") or root.get("/XFA")):
            return True
    except Exception:
        pass
    return False


# Compatibility wrapper functions used by tests/legacy callers
def _compute_char_density(page) -> float:
    return char_density(page)


def _compute_image_ratio(page) -> float:
    return image_ratio(page)


def _has_font_metadata(pdf) -> bool:
    return has_font_metadata(pdf)


def detect_language_heuristic(text: str) -> Tuple[str, float]:
    """Lightweight heuristic optimized for Amharic and English detection."""
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
    """Infer a coarse domain hint using keyword matching."""
    text_lower = text.lower()
    scores: Dict[str, int] = {d: 0 for d in rules.get("domain_keywords", {})}

    for domain, keywords in rules.get("domain_keywords", {}).items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                scores[domain] += 1

    if not any(scores.values()):
        return "general"

    return max(scores, key=scores.get)


# ─── Classification Functions ──────────────────────────────────────────────

def detect_origin_type(
    densities: List[float],
    ratios: List[float],
    has_fonts: bool,
    is_form: bool,
    rules: dict,
) -> str:
    """Classify origin: native_digital / scanned_image / mixed / form_fillable.

    Key behavior change vs naive heuristics:
    - **Font metadata + some high-density pages** => native_digital (even if image-heavy).
    - “Mixed” is reserved for documents where most sampled pages look scanned but
      there is some clear digital text present.
    """
    if is_form:
        return "form_fillable"

    if not densities:
        return "mixed"

    cfg = rules.get("origin_detection", {})
    scanned_max_density = float(cfg.get("scanned_max_char_density", 0.05))
    scanned_min_image = float(cfg.get("scanned_min_image_ratio", 0.92))
    digital_min_density = float(cfg.get("digital_min_char_density", 0.15))
    low_frac_threshold = float(cfg.get("scanned_low_density_fraction", 0.6))
    high_frac_threshold = float(cfg.get("digital_high_density_fraction", 0.3))

    n = len(densities)
    mean_density = _safe_mean(densities)
    max_density = max(densities)
    low_frac = sum(1 for d in densities if d < scanned_max_density) / max(1, n)
    high_frac = sum(1 for d in densities if d >= digital_min_density) / max(1, n)

    img_median = _median(ratios) if ratios else 0.0

    # Strong digital signal: font metadata implies selectable text.
    # This is intentionally prioritized so annual reports (often image-heavy)
    # remain "native_digital".
    if has_fonts:
        # Override to scanned when signals are overwhelmingly scanned
        # (e.g., OCR layer exists but pages are image-dominant).
        scanned_support_image = float(cfg.get("scanned_support_image_ratio", 0.50))
        if img_median >= scanned_min_image and mean_density <= scanned_max_density:
            return "scanned_image"
        if img_median >= scanned_support_image and low_frac >= low_frac_threshold and mean_density <= scanned_max_density:
            return "scanned_image"

        # Mixed origin: heterogeneous density across pages (some sparse, some dense).
        mixed_low_density_threshold = float(cfg.get("mixed_low_density_threshold", 1.0))
        mixed_high_density_threshold = float(cfg.get("mixed_high_density_threshold", 3.0))
        mixed_low_density_fraction = float(cfg.get("mixed_low_density_fraction", 0.15))
        mixed_high_density_fraction = float(cfg.get("mixed_high_density_fraction", 0.40))

        low_frac_mixed = sum(1 for d in densities if d < mixed_low_density_threshold) / max(1, n)
        high_frac_mixed = sum(1 for d in densities if d >= mixed_high_density_threshold) / max(1, n)

        if low_frac_mixed >= mixed_low_density_fraction and high_frac_mixed >= mixed_high_density_fraction:
            return "mixed"

        return "native_digital"

    # No fonts: fall back to density/image heuristics.
    if img_median >= scanned_min_image:
        return "scanned_image"

    if low_frac >= low_frac_threshold and max_density < digital_min_density:
        return "scanned_image"

    # Some PDFs have extractable text but lack fontname metadata in pdfplumber;
    # treat those as mixed (conservative routing).
    if mean_density >= digital_min_density or high_frac >= high_frac_threshold:
        return "mixed"

    return "mixed"


def detect_layout_complexity(sample_pages: List, rules: dict) -> str:
    """Estimate layout complexity (single/multi-column, table-heavy, etc.)."""
    table_area_total = 0.0
    figure_area_total = 0.0
    page_area_total = 0.0
    # Multi-column evidence is evaluated per-page to avoid a few outlier pages
    # (e.g., full-page photos/covers) dominating the signal.
    multi_col_pages = 0
    pages_with_words = 0

    for page in sample_pages:
        try:
            w, h = page.width or 600, page.height or 800
            page_area = w * h
            page_area_total += page_area

            for t in page.find_tables() or []:
                bbox = getattr(t, "bbox", None)
                if bbox:
                    table_area_total += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            for im in page.images or []:
                try:
                    figure_area_total += (im["x1"] - im["x0"]) * (im["y1"] - im["y0"])
                except Exception:
                    pass

            # Multi-column detection: count words on left and right halves.
            # This catches annual reports where many interior pages are 2-column,
            # even if cover/back-cover pages have no words.
            words = page.extract_words() or []
            x_centers: List[float] = []
            for word in words:
                x0 = word.get("x0")
                x1 = word.get("x1")
                if x0 is not None and x1 is not None:
                    x_centers.append((x0 + x1) / 2)

            if x_centers:
                pages_with_words += 1
                mid = (page.width or 600) / 2.0
                # Ignore a small gutter around the center to reduce false positives
                # from headings spanning the full width.
                gutter_frac = float(rules.get("layout_detection", {}).get("multi_column_center_gutter_frac", 0.05))
                left_max = mid * (1.0 - gutter_frac)
                right_min = mid * (1.0 + gutter_frac)

                left = sum(1 for x in x_centers if x < left_max)
                right = sum(1 for x in x_centers if x > right_min)
                min_each_side = int(rules.get("layout_detection", {}).get("multi_column_min_words_each_side", 50))
                if left >= min_each_side and right >= min_each_side:
                    multi_col_pages += 1
        except Exception:
            continue

    table_ratio = table_area_total / max(page_area_total, 1.0)
    figure_ratio = figure_area_total / max(page_area_total, 1.0)

    cfg = rules.get("layout_detection", {})
    table_heavy_min = float(cfg.get("table_heavy_min_ratio", 0.18))
    mixed_table_min = float(cfg.get("mixed_table_min_ratio", 0.05))
    fig_heavy_min = float(cfg.get("figure_heavy_min_ratio", 0.30))

    # Table-heavy dominates when tables occupy substantial area.
    if table_ratio >= table_heavy_min:
        return "table_heavy"

    # Multi-column evidence across a fraction of text-bearing pages.
    min_page_fraction = float(cfg.get("multi_column_min_page_fraction", 0.25))
    multi_col_fraction = (multi_col_pages / pages_with_words) if pages_with_words > 0 else 0.0
    is_multi_column = multi_col_fraction >= min_page_fraction

    # Mixed layout: multi-column structure + meaningful table presence,
    # and a higher multi-column prevalence (to separate Class A vs C).
    mixed_min_fraction = float(cfg.get("mixed_multi_column_min_fraction", 0.70))
    if is_multi_column and multi_col_fraction >= mixed_min_fraction and table_ratio >= mixed_table_min:
        return "mixed"

    if is_multi_column:
        return "multi_column"

    if figure_ratio >= fig_heavy_min:
        return "figure_heavy"

    return "single_column"


class TriageAgent:
    """Stage 1 agent that computes and caches DocumentProfiles."""

    def __init__(self, refinery_dir: Path | None = None):
        self.rules = load_rules()
        self.refinery_dir = Path(refinery_dir) if refinery_dir is not None else PROFILES_DIR
        self.refinery_dir.mkdir(parents=True, exist_ok=True)

    def triage(self, file_path: str | Path) -> DocumentProfile:
        file_path = Path(file_path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = compute_doc_id(file_path)
        profile_path = self.refinery_dir / f"{doc_id}.json"

        if profile_path.exists():
            cached = DocumentProfile.model_validate_json(profile_path.read_text())
            # Cache invalidation: recompute when triage logic changes.
            if getattr(cached, "triage_version", None) == TRIAGE_VERSION:
                return cached

        start = time.time()

        if pdfplumber is None:
            raise ImportError("pdfplumber is required for TriageAgent; install it to run triage")

        try:
            pdf = pdfplumber.open(file_path)
        except Exception:
            profile = DocumentProfile(
                doc_id=doc_id,
                filename=file_path.name,
                file_path=str(file_path),
                page_count=0,
                origin_type="mixed",
                layout_complexity="mixed",
                language_code="und",
                language_confidence=0.0,
                domain_hint="general",
                estimated_extraction_cost="needs_vision_model",
                char_density_mean=0.0,
                image_ratio_mean=0.0,
                table_count_total=0,
                has_font_metadata=False,
                is_form_fillable=False,
                confidence=0.0,
                triage_version=TRIAGE_VERSION,
            )
            profile_path.write_text(profile.model_dump_json(indent=2))
            return profile

        with pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                profile = DocumentProfile(
                    doc_id=doc_id,
                    filename=file_path.name,
                    file_path=str(file_path),
                    page_count=0,
                    origin_type="mixed",
                    layout_complexity="mixed",
                    language_code="und",
                    language_confidence=0.0,
                    domain_hint="general",
                    estimated_extraction_cost="needs_vision_model",
                    char_density_mean=0.0,
                    image_ratio_mean=0.0,
                    table_count_total=0,
                    has_font_metadata=False,
                    is_form_fillable=False,
                    confidence=0.0,
                    triage_version=TRIAGE_VERSION,
                )
                profile_path.write_text(profile.model_dump_json(indent=2))
                return profile

            origin_cfg = self.rules.get("origin_detection", {})
            max_samples = int(origin_cfg.get("sample_pages_max", 8))
            indices = _sample_page_indices(page_count, max_samples=max_samples)
            sample_pages = [pdf.pages[i] for i in indices]

            densities = [_compute_char_density(p) for p in sample_pages]
            img_ratios = [_compute_image_ratio(p) for p in sample_pages]
            has_fonts = _has_font_metadata(pdf)
            is_form = is_form_fillable(pdf)

            text_sample = " ".join((p.extract_text() or "")[:600] for p in sample_pages)

            origin_type = detect_origin_type(densities, img_ratios, has_fonts, is_form, self.rules)
            layout_complexity = detect_layout_complexity(sample_pages, self.rules)
            language_code, language_confidence = detect_language_heuristic(text_sample)
            domain_hint = infer_domain(text_sample, self.rules)

            estimated_extraction_cost = (
                "fast_text_sufficient"
                if origin_type == "native_digital" and layout_complexity == "single_column"
                else "needs_vision_model"
                if origin_type in ("scanned_image", "form_fillable")
                else "needs_layout_model"
            )

            mean_density = _safe_mean(densities)
            mean_image = _safe_mean(img_ratios)

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
                table_count_total=0,
                has_font_metadata=has_fonts,
                is_form_fillable=is_form,
                confidence=round(language_confidence, 3),
                triage_version=TRIAGE_VERSION,
            )

        profile_path.write_text(profile.model_dump_json(indent=2))

        # Keep prints minimal; callers (corpus runner) usually capture logs.
        _ = time.time() - start
        return profile


def main() -> None:  # pragma: no cover
    agent = TriageAgent()
    for pdf in Path("data").rglob("*.pdf"):
        try:
            profile = agent.triage(pdf)
            print(
                f"✓ {pdf.name} | {profile.origin_type:14} | {profile.layout_complexity:14} | {profile.estimated_extraction_cost}"  # noqa: E501
            )
        except Exception as e:
            print(f"✗ {pdf.name}: {type(e).__name__} - {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
