from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
from ..models.schemas import ExtractedDocument, ExtractionLedgerEntry, TextBlock, TableObject, BoundingBox
import time


class DoclingDocumentAdapter:
    """Adapter to convert a DoclingDocument into our ExtractedDocument schema.
    If Docling is not available, this adapter is not used.
    """

    @staticmethod
    def adapt(docl_document) -> ExtractedDocument:
        # docl_document is expected to expose pages, blocks, tables, etc.
        doc = ExtractedDocument(doc_id=getattr(docl_document, 'name', 'unknown'), pages=getattr(docl_document, 'pages', 0))
        # best effort mapping
        try:
            for pnum, page in enumerate(docl_document.pages, start=1):
                # text blocks
                for b in getattr(page, 'text_blocks', []) or []:
                    text = getattr(b, 'text', '')
                    bbox = getattr(b, 'bbox', None)
                    if bbox:
                        bb = BoundingBox(x0=bbox.x0, top=bbox.top, x1=bbox.x1, bottom=bbox.bottom)
                    else:
                        bb = BoundingBox(x0=0, top=0, x1=0, bottom=0)
                    tb = TextBlock(text=text, bbox=bb, page_number=pnum)
                    doc.text_blocks.append(tb)
                # tables
                for t in getattr(page, 'tables', []) or []:
                    headers = getattr(t, 'headers', []) or []
                    rows = getattr(t, 'rows', []) or []
                    bbox = getattr(t, 'bbox', None)
                    if bbox:
                        bb = BoundingBox(x0=bbox.x0, top=bbox.top, x1=bbox.x1, bottom=bbox.bottom)
                    else:
                        bb = BoundingBox(x0=0, top=0, x1=0, bottom=0)
                    to = TableObject(headers=[str(h) for h in headers], rows=[[str(c) for c in r] for r in rows], bbox=bb, page_number=pnum)
                    doc.tables.append(to)
        except Exception:
            # best-effort: ignore adapter errors
            pass
        return doc


class LayoutExtractor:
    """Layout-aware extractor using Docling when available; falls back to a lightweight behaviour.
    Tries to instantiate Docling and parse pages; if missing, returns an empty ExtractedDocument.
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def extract(self) -> Dict[str, Any]:
        start = time.time()
        processing_time = 0.0
        ledger = None
        try:
            import docling
            # Attempt to load and parse using Docling API
            dd = docling.load_document(str(self.path))
            doc = DoclingDocumentAdapter.adapt(dd)
            processing_time = time.time() - start
            ledger = ExtractionLedgerEntry(
                doc_id=self.path.name,
                strategy_used="layout_docling",
                confidence_score=0.92,
                cost_estimate=0.25,
                processing_time_s=processing_time,
                notes="layout extractor using Docling",
            )
            return {"document": doc, "confidence": ledger.confidence_score, "ledger": ledger}
        except Exception:
            # fallback: return empty ExtractedDocument but not fail hard
            from pdfplumber import open as pdfopen
            doc = ExtractedDocument(doc_id=self.path.name, pages=0)
            try:
                with pdfopen(self.path) as pdf:
                    doc.pages = len(pdf.pages)
                    # try to extract tables via pdfplumber for fallback
                    for i, p in enumerate(pdf.pages, start=1):
                        try:
                            tbs = p.extract_tables() or []
                            for t in tbs:
                                headers = t[0] if t and len(t) > 0 else []
                                rows = t[1:] if t and len(t) > 1 else []
                                bbox = BoundingBox(x0=0, top=0, x1=p.width, bottom=p.height)
                                to = TableObject(headers=[str(h) for h in headers], rows=[[str(c) for c in r] for r in rows], bbox=bbox, page_number=i)
                                doc.tables.append(to)
                        except Exception:
                            pass
            except Exception:
                pass
            processing_time = time.time() - start
            ledger = ExtractionLedgerEntry(
                doc_id=self.path.name,
                strategy_used="layout_fallback",
                confidence_score=0.6,
                cost_estimate=0.2,
                processing_time_s=processing_time,
                notes="layout extractor fallback (pdfplumber tables)",
            )
            return {"document": doc, "confidence": ledger.confidence_score, "ledger": ledger}
