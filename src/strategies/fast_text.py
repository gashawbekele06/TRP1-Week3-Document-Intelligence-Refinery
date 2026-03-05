"""
Strategy A: Fast Text Extractor
==============================

Purpose in pipeline:
Low-cost, high-speed extraction for clean native-digital PDFs (single-column, no heavy layout).
Uses pdfplumber to extract text, tables, and basic image blocks.
Computes multi-signal confidence score → router escalates if < threshold.

Triggers (decided by router, not here):
- origin_type == "native_digital" AND layout_complexity == "single_column"

Confidence signals (justified in DOMAIN_NOTES.md):
- Text volume (chars per page > 100)
- Char density (chars / 1000 pt²)
- Image area ratio (< 50%)
- Font metadata presence
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pdfplumber

from src.models import (
    BBox,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    TableCell,
    TextBlock,
)
from src.strategies.base import ExtractionStrategy


# Load thresholds from config (should be moved to extraction_rules.yaml)
_MIN_CHARS_PER_PAGE = 100
_TARGET_CHAR_DENSITY = 150.0      # chars per 1000 pt² for clean digital
_MAX_IMAGE_RATIO = 0.50           # >50% image area → likely scanned
_CONFIDENCE_THRESHOLD = 0.65      # below this → escalate


def _compute_char_density(page) -> float:
    """
    Purpose: Estimate text density (chars per 1000 pt²).
    Uses chars list first, falls back to extract_text() for robustness.
    """
    try:
        chars = page.chars or []
        text = "".join(c.get("text", "") for c in chars if c.get("text", "").strip())
        if not text:
            text = page.extract_text() or ""
        area_1000 = (page.width or 1) * (page.height or 1) / 1000.0
        return len(text) / area_1000 if area_1000 > 0 else 0.0
    except Exception:
        return 0.0


def _compute_image_ratio(page) -> float:
    """Purpose: Fraction of page area covered by embedded images."""
    try:
        page_area = (page.width or 1) * (page.height or 1)
        img_area = sum(
            max(0, im.get("x1", 0) - im.get("x0", 0)) * max(0, im.get("y1", 0) - im.get("y0", 0))
            for im in (page.images or [])
        )
        return min(img_area / page_area, 1.0) if page_area > 0 else 0.0
    except Exception:
        return 0.0


def _has_font_metadata(pdf) -> bool:
    """Purpose: Detect embedded font info (strong signal of native digital text)."""
    try:
        for page in pdf.pages[:5]:
            if page.chars and any("fontname" in c for c in page.chars):
                return True
    except Exception:
        pass
    return False


class FastTextExtractor(ExtractionStrategy):
    """
    Strategy A: Fast Text Extraction using pdfplumber.

    Purpose:
    - Quick & cheap extraction for clean native-digital documents
    - Produces structured output with provenance
    - Computes confidence → enables escalation guard
    """

    name = "fast_text"

    def extract(self, file_path: Path) -> ExtractedDocument:
        """
        Main extraction method.

        Purpose:
        - Extract text blocks, tables, figures
        - Collect signals for confidence
        - Build normalized ExtractedDocument
        """
        start_time = time.time()
        text_blocks: List[TextBlock] = []
        tables: List[ExtractedTable] = []
        figures: List[ExtractedFigure] = []
        section_headings: List[TextBlock] = []
        full_text_parts: List[str] = []

        char_densities = []
        image_ratios = []
        has_fonts = False

        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                has_fonts = _has_font_metadata(pdf)

                for page_num, page in enumerate(pdf.pages, 1):
                    # ── Collect signals for confidence ────────────────────────
                    density = _compute_char_density(page)
                    img_ratio = _compute_image_ratio(page)
                    char_densities.append(density)
                    image_ratios.append(img_ratio)

                    # Early abort for clearly bad pages (scanned-like)
                    if density < 0.05 and img_ratio > 0.75:
                        continue  # skip this page

                    # ── Text Blocks ────────────────────────────────────────────
                    words = page.extract_words(extra_attrs=["fontname", "size"]) or []
                    page_text = page.extract_text() or ""
                    full_text_parts.append(page_text)

                    for word in words:
                        size = float(word.get("size", 12))
                        is_header = size > 14  # simple heuristic
                        block = TextBlock(
                            text=word.get("text", ""),
                            bbox=BBox(
                                x0=float(word.get("x0", 0)),
                                y0=float(word.get("top", 0)),
                                x1=float(word.get("x1", 0)),
                                y1=float(word.get("bottom", 0)),
                                page=page_num,
                            ),
                            page=page_num,
                            font_name=word.get("fontname"),
                            font_size=size,
                            is_header=is_header,
                            reading_order=len(text_blocks),  # basic sequential order
                        )
                        text_blocks.append(block)
                        if is_header:
                            section_headings.append(block)

                    # ── Tables ─────────────────────────────────────────────────
                    for t_idx, table in enumerate(page.find_tables() or []):
                        try:
                            raw_data = table.extract() or []
                            if len(raw_data) < 2:
                                continue

                            headers = [str(h or "") for h in raw_data[0]]
                            rows = [[str(c or "") for c in row] for row in raw_data[1:]]

                            cells = []
                            for ri, row in enumerate(raw_data):
                                for ci, val in enumerate(row):
                                    cells.append(
                                        TableCell(
                                            value=str(val or ""),
                                            row=ri,
                                            col=ci,
                                            is_header=(ri == 0),
                                        )
                                    )

                            bbox_raw = table.bbox
                            ext_table = ExtractedTable(
                                table_id=f"t_{page_num}_{t_idx}",
                                bbox=BBox(
                                    x0=bbox_raw[0], y0=bbox_raw[1],
                                    x1=bbox_raw[2], y1=bbox_raw[3],
                                    page=page_num
                                ) if bbox_raw else None,
                                page=page_num,
                                headers=headers,
                                rows=rows,
                                cells=cells,
                                reading_order=len(tables),
                            )
                            tables.append(ext_table)
                        except Exception:
                            continue

                    # ── Figures ────────────────────────────────────────────────
                    for img_idx, img in enumerate(page.images or []):
                        try:
                            fig = ExtractedFigure(
                                figure_id=f"f_{page_num}_{img_idx}",
                                bbox=BBox(
                                    x0=float(img.get("x0", 0)),
                                    y0=float(img.get("y0", 0)),
                                    x1=float(img.get("x1", 0)),
                                    y1=float(img.get("y1", 0)),
                                    page=page_num,
                                ),
                                page=page_num,
                                reading_order=len(figures),
                            )
                            figures.append(fig)
                        except Exception:
                            continue

        except Exception as e:
            error_msg = str(e)
        else:
            error_msg = None

        duration = time.time() - start_time

        doc_id = f"{file_path.stem}_{uuid.uuid4().hex[:8]}"

        result = ExtractedDocument(
            doc_id=doc_id,
            filename=file_path.name,
            strategy_used=self.name,
            confidence=0.0,
            page_count=page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            full_text="\n\n".join(full_text_parts),
            section_headings=section_headings,
            processing_time_sec=round(duration, 2),
            metadata={
                "char_density_mean": round(sum(char_densities) / max(len(char_densities), 1), 3),
                "image_ratio_mean": round(sum(image_ratios) / max(len(image_ratios), 1), 4),
                "has_font_metadata": has_fonts,
            },
        )

        # Final confidence computation
        result.confidence = self.confidence(result)
        return result

    def confidence(self, doc: ExtractedDocument) -> float:
        """
        Purpose: Multi-signal confidence score to decide escalation.

        Signals:
        1. Text volume (chars per page vs min threshold)
        2. Density (chars per 1000 pt² vs target)
        3. Image ratio penalty
        4. Font metadata bonus

        Returns: 0.0–1.0 score
        """
        if doc.page_count == 0:
            return 0.0

        total_chars = sum(len(b.text) for b in doc.text_blocks)
        chars_per_page = total_chars / doc.page_count

        meta = doc.metadata or {}
        density_mean = meta.get("char_density_mean", 0.0)
        image_mean = meta.get("image_ratio_mean", 0.0)
        has_fonts = meta.get("has_font_metadata", False)

        # Signal 1: Text volume
        text_score = min(1.0, chars_per_page / _MIN_CHARS_PER_PAGE)

        # Signal 2: Density
        density_score = min(1.0, density_mean / _TARGET_CHAR_DENSITY)

        # Signal 3: Image penalty
        image_penalty = max(0.0, 1.0 - (image_mean / _MAX_IMAGE_RATIO))

        # Signal 4: Font bonus
        font_bonus = 0.12 if has_fonts else 0.0

        score = 0.40 * text_score + 0.35 * density_score + 0.13 * image_penalty + font_bonus

        # Strong penalty for image-dominant pages
        if image_mean > 0.75:
            score *= 0.3

        return round(min(max(score, 0.0), 1.0), 3)