"""src.pipeline.corpus_runner

Corpus validation runner
=======================

Requirement context
-------------------
Your deliverable must be validated against a heterogeneous corpus of PDFs
(Class A–D). This runner executes the full pipeline over a folder of PDFs:

1) Stage 1: Triage → DocumentProfile (cached under `.refinery/profiles/`)
2) Stage 2: ExtractionRouter → ExtractedDocument (logged under extraction ledger)
3) Stage 3: ChunkingEngine → LDUs (persisted under `.refinery/ldus/`)
4) Stage 4: PageIndexBuilder → PageIndex tree (persisted under `.refinery/pageindex/`)
5) Phase 4: FactTable extraction → SQLite facts table (persisted under `.refinery/facts/`)

Outputs
-------
- `.refinery/extracted/<doc_id>.json`          ExtractedDocument JSON
- `.refinery/ldus/<doc_id>.jsonl`             One LDU per line
- `.refinery/pageindex/<doc_id>.json`         PageIndex JSON
- `.refinery/reports/corpus_report.json`      Summary report

This module is intentionally offline-first: it works without Docling/Gemini.
If OPENROUTER_API_KEY is set, the vision strategy can be used for scanned PDFs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexBuilder
from src.agents.triage import TriageAgent
from src.agents.chunker import ChunkingEngine
from src.data import FactTable


_REFINERY_DIR = Path(".refinery")
_EXTRACTED_DIR = _REFINERY_DIR / "extracted"
_LDU_DIR = _REFINERY_DIR / "ldus"
_REPORT_DIR = _REFINERY_DIR / "reports"


def _ensure_dirs() -> None:
    for d in (_EXTRACTED_DIR, _LDU_DIR, _REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _guess_class(profile) -> str:
    """Heuristic label for reporting only (A/B/C/D)."""
    try:
        if profile.origin_type == "native_digital" and profile.layout_complexity in {"single_column", "multi_column"}:
            return "A"
        if profile.origin_type == "scanned_image":
            return "B"
        if profile.layout_complexity in {"table_heavy", "figure_heavy"}:
            return "D"
        return "C"
    except Exception:
        return "unknown"


@dataclass
class CorpusRunResult:
    file: str
    doc_id: str
    class_guess: str
    origin_type: str
    layout_complexity: str
    initial_cost_route: str
    final_strategy: str
    confidence: float
    pages_processed: int
    page_count: int
    success: bool
    error_message: str | None
    elapsed_sec: float


def process_pdf(
    pdf_path: Path,
    *,
    use_llm_for_pageindex: bool = False,
    ledger_path: Path | None = None,
) -> CorpusRunResult:
    """Run the pipeline for a single PDF and persist artifacts."""
    _ensure_dirs()

    start = time.time()

    triage = TriageAgent(refinery_dir=_REFINERY_DIR / "profiles")
    profile = triage.triage(pdf_path)

    router = ExtractionRouter(ledger_path=ledger_path)
    extracted = router.extract_with_escalation(pdf_path, profile)

    # Persist ExtractedDocument JSON for inspection
    extracted_path = _EXTRACTED_DIR / f"{profile.doc_id}.json"
    extracted_path.write_text(extracted.model_dump_json(indent=2), encoding="utf-8")

    # Stage 3: chunk
    chunker = ChunkingEngine()
    ldus = chunker.chunk(extracted)

    # Persist LDUs (JSONL)
    ldu_out = _LDU_DIR / f"{profile.doc_id}.jsonl"
    with ldu_out.open("w", encoding="utf-8") as f:
        for ldu in ldus:
            f.write(ldu.model_dump_json())
            f.write("\n")

    # Stage 4: page index
    builder = PageIndexBuilder(use_llm=use_llm_for_pageindex)
    # keep default pageindex dir under .refinery/pageindex
    builder.build(extracted, ldus)

    # Phase 4: FactTable
    FactTable().extract_and_store_from_ldus(ldus, doc_id=profile.doc_id, document_name=profile.filename)

    elapsed = time.time() - start

    return CorpusRunResult(
        file=str(pdf_path),
        doc_id=profile.doc_id,
        class_guess=_guess_class(profile),
        origin_type=profile.origin_type,
        layout_complexity=profile.layout_complexity,
        initial_cost_route=profile.estimated_extraction_cost,
        final_strategy=extracted.strategy_used,
        confidence=float(extracted.confidence),
        pages_processed=int(getattr(extracted, "pages_processed", 0) or 0),
        page_count=int(getattr(extracted, "page_count", 0) or 0),
        success=bool(extracted.success),
        error_message=getattr(extracted, "error_message", None),
        elapsed_sec=round(elapsed, 3),
    )


def run_corpus(
    input_dir: Path,
    *,
    pattern: str = "*.pdf",
    limit: int | None = None,
    use_llm_for_pageindex: bool = False,
) -> dict[str, Any]:
    """Run pipeline over all PDFs in a folder and write a report."""
    _ensure_dirs()

    pdfs = sorted(input_dir.rglob(pattern))
    if limit is not None:
        pdfs = pdfs[: max(0, limit)]

    results: list[dict[str, Any]] = []

    ok = 0
    for p in pdfs:
        try:
            r = process_pdf(p, use_llm_for_pageindex=use_llm_for_pageindex)
            d = r.__dict__
            results.append(d)
            ok += 1 if d.get("success") else 0
        except Exception as e:
            results.append(
                {
                    "file": str(p),
                    "doc_id": "",
                    "class_guess": "unknown",
                    "success": False,
                    "error_message": f"{type(e).__name__}: {e}",
                }
            )

    summary = {
        "input_dir": str(input_dir),
        "pattern": pattern,
        "total": len(pdfs),
        "success_count": ok,
        "success_rate": round(ok / max(1, len(pdfs)), 3),
        "results": results,
    }

    out = _REPORT_DIR / "corpus_report.json"
    _save_json(out, summary)
    return summary
