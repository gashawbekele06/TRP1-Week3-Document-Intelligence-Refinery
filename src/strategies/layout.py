from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
from ..models.schemas import ExtractedDocument, ExtractionLedgerEntry
import time


class LayoutExtractor:
    """Layout-aware extractor stub. Intended to integrate MinerU or Docling. For now falls back to pdfplumber-like output.
    Implement a DoclingDocumentAdapter here when available.
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def extract(self) -> Dict[str, Any]:
        start = time.time()
        # Placeholder implementation: in a full implementation, call Docling/MinerU and adapt to ExtractedDocument
        doc = ExtractedDocument(doc_id=self.path.name, pages=0)
        processing_time = time.time() - start
        ledger = ExtractionLedgerEntry(
            doc_id=self.path.name,
            strategy_used="layout",
            confidence_score=0.85,
            cost_estimate=0.2,
            processing_time_s=processing_time,
            notes="layout extractor (placeholder) - integrate Docling/MinerU adapter",
        )
        return {"document": doc, "confidence": ledger.confidence_score, "ledger": ledger}
