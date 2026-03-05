"""
Strategy B: Layout-Aware Extractor using Docling (IBM Research)
==============================================================

Purpose in pipeline:
Medium-cost extraction for documents with complex layout (multi-column, table-heavy, mixed).
Uses Docling to reconstruct reading order, extract structured tables, figures with captions,
and text blocks with bounding boxes.

Triggers (decided by router):
- layout_complexity in ["multi_column", "table_heavy", "figure_heavy", "mixed"]
- OR origin_type == "mixed"

Key features:
- DoclingDocumentAdapter normalizes DoclingDocument → ExtractedDocument
- Graceful fallback to FastTextExtractor if Docling unavailable
- Confidence scoring tuned for layout-aware success (high when structure extracted)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from src.models import (
    BBox,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    TableCell,
    TextBlock,
)
from src.strategies.base import ExtractionStrategy


class DoclingDocumentAdapter:
    """
    Purpose:
    Convert Docling's DoclingDocument model into our normalized ExtractedDocument schema.
    Maps Docling items (TextItem, TableItem, PictureItem, etc.) to TextBlock/Table/Figure.
    Preserves bbox, reading order, captions, table structure.
    """

    def __init__(self, doc):
        self.doc = doc

    def to_extracted(self, file_path: Path, processing_seconds: float) -> ExtractedDocument:
        """
        Main adapter method.

        Args:
            file_path: Original PDF path
            processing_seconds: Measured time taken

        Returns:
            ExtractedDocument: fully populated result
        """
        text_blocks: List[TextBlock] = []
        tables: List[ExtractedTable] = []
        figures: List[ExtractedFigure] = []
        section_headings: List[TextBlock] = []

        try:
            from docling.datamodel.document import (
                TextItem,
                TableItem,
                PictureItem,
                SectionHeaderItem,
            )
        except ImportError:
            # Docling not installed → caller should fallback
            raise RuntimeError("Docling not available — fallback to pdfplumber")

        reading_counter = 0  # simple global order

        for item in self.doc.iterate_items():
            # Get page & bbox from provenance (Docling format)
            page_num = 1
            bbox_obj = None
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0]
                page_num = getattr(prov, "page_no", 1)
                bb = getattr(prov, "bbox", None)
                if bb:
                    try:
                        bbox_obj = BBox(
                            x0=float(bb.l),
                            y0=float(bb.t),
                            x1=float(bb.r),
                            y1=float(bb.b),
                            page=page_num,
                        )
                    except Exception:
                        bbox_obj = None

            item_type = type(item).__name__

            # ── Text ─────────────────────────────────────────────────────────────
            if hasattr(item, "text") and getattr(item, "text", "").strip():
                text = item.text.strip()
                is_header = item_type in ("SectionHeaderItem", "DocTitle", "Heading")
                block = TextBlock(
                    text=text,
                    bbox=bbox_obj,
                    page=page_num,
                    is_header=is_header,
                    reading_order=reading_counter,
                )
                text_blocks.append(block)
                if is_header:
                    section_headings.append(block)
                reading_counter += 1

            # ── Tables ───────────────────────────────────────────────────────────
            elif item_type == "TableItem":
                try:
                    # Docling table → dataframe
                    df = item.export_to_dataframe()
                    headers = [str(h) for h in df.columns]
                    rows = [[str(c) for c in row] for _, row in df.iterrows()]

                    cells = []
                    for ri, row in enumerate(df.itertuples(index=False), start=1):
                        for ci, val in enumerate(row):
                            cells.append(
                                TableCell(
                                    value=str(val),
                                    row=ri,
                                    col=ci,
                                    is_header=(ri == 1),  # first row headers
                                )
                            )

                    ext_table = ExtractedTable(
                        table_id=f"t_{page_num}_{len(tables)}",
                        bbox=bbox_obj,
                        page=page_num,
                        headers=headers,
                        rows=rows,
                        cells=cells,
                        caption=item.caption if hasattr(item, "caption") else None,
                        reading_order=reading_counter,
                    )
                    tables.append(ext_table)
                    reading_counter += 1
                except Exception:
                    continue

            # ── Figures ──────────────────────────────────────────────────────────
            elif item_type == "PictureItem":
                fig = ExtractedFigure(
                    figure_id=f"f_{page_num}_{len(figures)}",
                    bbox=bbox_obj,
                    page=page_num,
                    caption=item.caption if hasattr(item, "caption") else None,
                    reading_order=reading_counter,
                )
                figures.append(fig)
                reading_counter += 1

        page_count = max((b.page for b in text_blocks if b.page > 0), default=1)
        full_text = "\n\n".join(b.text for b in text_blocks)

        result = ExtractedDocument(
            doc_id=f"{file_path.stem}_docling_{uuid.uuid4().hex[:6]}",
            filename=file_path.name,
            strategy_used="layout_aware",
            confidence=0.0,
            page_count=page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            full_text=full_text,
            section_headings=section_headings,
            processing_time_sec=round(processing_seconds, 2),
            metadata={"docling_used": True, "reading_order_count": reading_counter},
        )

        # Final confidence (Docling usually very reliable)
        result.confidence = self.confidence(result)
        return result


class LayoutExtractor(ExtractionStrategy):
    """
    Strategy B: Layout-Aware Extraction using Docling.

    Purpose:
    - Handle complex layouts (multi-column, tables, figures)
    - Preserve reading order, structure, captions
    - Fallback to FastTextExtractor if Docling import fails
    """

    name = "layout_aware"

    def __init__(self):
        self.docling_available = self._check_docling()

    def _check_docling(self) -> bool:
        """Purpose: Detect if Docling is installed/usable."""
        try:
            from docling.document_converter import DocumentConverter  # noqa
            return True
        except ImportError:
            return False

    def _extract_with_docling(self, file_path: Path) -> ExtractedDocument:
        """Real Docling extraction path."""
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        start = time.time()

        try:
            # Configure Docling pipeline
            pipeline_options = PdfPipelineOptions(
                do_ocr=False,              # we trust native text when possible
                do_table_structure=True,
                do_layout_detection=True,
            )

            converter = DocumentConverter()
            result = converter.convert(str(file_path), pipeline_options=pipeline_options)
            doc = result.document

            adapter = DoclingDocumentAdapter(doc)
            return adapter.to_extracted(file_path, processing_seconds=time.time() - start)

        except Exception as e:
            print(f"Docling failed: {e} — falling back to pdfplumber")
            return self._extract_fallback(file_path)

    def _extract_fallback(self, file_path: Path) -> ExtractedDocument:
        """Graceful fallback when Docling fails."""
        from src.strategies.fast_text import FastTextExtractor
        doc = FastTextExtractor().extract(file_path)
        doc.strategy_used = "layout_aware_fallback"
        doc.metadata["docling_used"] = False
        return doc

    def extract(self, file_path: Path) -> ExtractedDocument:
        """
        Main entry point.

        Purpose:
        - Try Docling first (preferred)
        - Fallback to fast_text on failure
        - Return normalized result
        """
        if self.docling_available:
            return self._extract_with_docling(file_path)
        return self._extract_fallback(file_path)

    def confidence(self, doc: ExtractedDocument) -> float:
        """
        Purpose: Score reliability of layout-aware extraction.

        Logic:
        - Docling usually high confidence if tables/figures extracted
        - Higher baseline than fast_text (0.75–0.95)
        """
        if doc.page_count == 0:
            return 0.0

        has_text = len(doc.text_blocks) > 0
        text_score = min(1.0, sum(len(b.text) for b in doc.text_blocks) / (doc.page_count * 200))
        table_score = min(0.4, len(doc.tables) * 0.1)
        figure_score = min(0.3, len(doc.figures) * 0.15)

        score = 0.45 * text_score + 0.35 * table_score + 0.20 * figure_score + 0.2 * int(has_text)

        # Docling baseline boost
        if "docling_used" in doc.metadata and doc.metadata["docling_used"]:
            score = min(score + 0.25, 0.98)

        return round(score, 3)