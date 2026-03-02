from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any
import pdfplumber
from ..models.schemas import ExtractedDocument, TextBlock, BoundingBox, TableObject, ExtractionLedgerEntry


class FastTextExtractor:
    """Fast text extraction using pdfplumber. Produces ExtractedDocument and confidence score."""

    def __init__(self, path: str):
        self.path = Path(path)

    def extract(self) -> Dict[str, Any]:
        start = time.time()
        doc = ExtractedDocument(doc_id=self.path.name, pages=0)
        total_chars = 0
        total_image_area = 0.0
        page_areas = 0.0
        tables = []

        with pdfplumber.open(self.path) as pdf:
            doc.pages = len(pdf.pages)
            for i, p in enumerate(pdf.pages, start=1):
                text = p.extract_text() or ""
                total_chars += len(text)
                for w in p.extract_words():
                    bbox = BoundingBox(x0=w.get("x0", 0), top=w.get("top", 0), x1=w.get("x1", 0), bottom=w.get("bottom", 0))
                    tb = TextBlock(text=w.get("text", ""), bbox=bbox, page_number=i)
                    doc.text_blocks.append(tb)
                try:
                    found_tables = p.extract_tables()
                    for t in found_tables or []:
                        # simple normalization
                        headers = t[0] if t and len(t) > 0 else []
                        rows = t[1:] if t and len(t) > 1 else []
                        table_bbox = BoundingBox(x0=0, top=0, x1=p.width, bottom=p.height)
                        to = TableObject(headers=[str(h) for h in headers], rows=[[str(c) for c in r] for r in rows], bbox=table_bbox, page_number=i)
                        doc.tables.append(to)
                except Exception:
                    pass

        avg_chars = total_chars / max(1, doc.pages)
        # confidence scoring heuristic
        score = 0.0
        if avg_chars > 200:
            score += 0.7
        elif avg_chars > 80:
            score += 0.4
        if len(doc.tables) > 0:
            score += 0.1

        processing_time = time.time() - start
        ledger = ExtractionLedgerEntry(
            doc_id=self.path.name,
            strategy_used="fast_text",
            confidence_score=float(min(score, 0.99)),
            cost_estimate=0.01,
            processing_time_s=processing_time,
            notes="pdfplumber extraction",
        )

        return {"document": doc, "confidence": ledger.confidence_score, "ledger": ledger}
