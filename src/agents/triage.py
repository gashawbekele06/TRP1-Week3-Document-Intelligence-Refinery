from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple
import pdfplumber
from langdetect import detect, DetectorFactory
from ..models.schemas import DocumentProfile

DetectorFactory.seed = 0


DATA_DIR = Path(__file__).resolve().parents[2]
REFINERY_DIR = DATA_DIR / ".refinery"
REFINERY_DIR.mkdir(exist_ok=True)


def _page_image_area(page) -> float:
    """Estimate total image area on a page using pdfplumber page.images entries."""
    area = 0.0
    try:
        for img in page.images:
            w = img.get("width") or (img.get("x1", 0) - img.get("x0", 0))
            h = img.get("height") or (img.get("y1", 0) - img.get("y0", 0))
            area += float(w) * float(h)
    except Exception:
        return 0.0
    return area


def _detect_columns(page, threshold: int = 2) -> int:
    """Rudimentary column detection: cluster word x0 positions.
    Returns number of clusters (approx columns)."""
    words = page.extract_words()
    if not words:
        return 1
    xs = [w.get("x0", 0) for w in words]
    xs_sorted = sorted(xs)
    # simple gap-based clustering
    clusters = 1
    gaps = []
    for a, b in zip(xs_sorted[:-1], xs_sorted[1:]):
        gaps.append(b - a)
    if not gaps:
        return 1
    avg_gap = sum(gaps) / len(gaps)
    large_gaps = [g for g in gaps if g > avg_gap * 3]
    clusters += len(large_gaps)
    return max(1, clusters)


def classify_document(path: str) -> DocumentProfile:
    path = Path(path)
    doc_id = path.name
    origin = "native_digital"
    layout = "single_column"
    domain = "general"
    estimated = "fast_text_sufficient"
    signals = {}

    with pdfplumber.open(path) as pdf:
        pages = pdf.pages
        total_pages = len(pages)
        total_chars = 0
        total_image_area = 0.0
        page_areas = 0.0
        column_counts = []
        table_count = 0

        for p in pages:
            text = p.extract_text() or ""
            total_chars += len(text)
            page_area = float(p.width * p.height)
            page_areas += page_area
            img_area = _page_image_area(p)
            total_image_area += img_area
            cols = _detect_columns(p)
            column_counts.append(cols)
            try:
                tables = p.extract_tables()
                if tables:
                    table_count += len(tables)
            except Exception:
                pass

        avg_chars_per_page = total_chars / max(1, total_pages)
        image_area_ratio = total_image_area / max(1, page_areas)
        median_cols = sorted(column_counts)[len(column_counts) // 2]

        # heuristics
        if avg_chars_per_page < 50 and image_area_ratio > 0.4:
            origin = "scanned_image"
        elif image_area_ratio > 0.6:
            origin = "mixed"

        if median_cols >= 2:
            layout = "multi_column"
        if table_count > max(1, total_pages // 10):
            layout = "table_heavy"

        if layout == "table_heavy" or origin in ("mixed", "scanned_image"):
            estimated = "needs_layout_model"
        else:
            estimated = "fast_text_sufficient"

        # language detection from first 5 pages
        sample_text = "".join([(p.extract_text() or "") for p in pages[:5]])
        lang = None
        lang_conf = None
        try:
            if sample_text.strip():
                lang = detect(sample_text)
                lang_conf = 0.8
        except Exception:
            lang = None

        # domain hint via keywords
        txt_all = "".join([(p.extract_text() or "") for p in pages[:10]]).lower()
        if any(k in txt_all for k in ["revenue", "balance sheet", "income", "assets", "liabilities"]):
            domain = "financial"
        elif any(k in txt_all for k in ["court", "plaintiff", "defendant", "act", "statute"]):
            domain = "legal"
        elif any(k in txt_all for k in ["experiment", "methodology", "assessment", "findings"]):
            domain = "technical"

        signals = {
            "total_pages": total_pages,
            "avg_chars_per_page": avg_chars_per_page,
            "image_area_ratio": image_area_ratio,
            "median_columns": median_cols,
            "table_count": table_count,
        }

    profile = DocumentProfile(
        doc_id=doc_id,
        origin_type=origin,
        layout_complexity=layout,
        language=lang,
        language_confidence=lang_conf,
        domain_hint=domain,
        estimated_extraction_cost=estimated,
        signals=signals,
    )

    out = REFINERY_DIR / "profiles"
    out.mkdir(exist_ok=True)
    with open(out / f"{doc_id}.json", "w", encoding="utf-8") as f:
        f.write(profile.model_dump_json(indent=2))

    return profile


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: triage.py path/to/doc.pdf")
        raise SystemExit(1)
    p = classify_document(sys.argv[1])
    print(p.json(indent=2))
