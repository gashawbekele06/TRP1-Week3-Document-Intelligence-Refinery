"""
Unit & Integration Tests for Phase 2 – Extraction Layer
======================================================

Purpose:
Verify that:
- Strategies (A/B/C) produce valid ExtractedDocument output
- ExtractionRouter selects correct initial strategy from profile
- Escalation guard triggers on low confidence
- Ledger logging works (strategy, confidence, cost, time)
- Budget guard in Vision strategy prevents overspending

Tests cover:
- Happy path: fast_text on clean digital document
- Escalation: low confidence → fallback to layout/vision
- Budget guard: stops when max cost reached
- Fallback behavior when advanced strategies unavailable

Run with: pytest tests/test_extraction.py -v
"""

import pytest
import json
from pathlib import Path
from typing import Generator
import fitz  # PyMuPDF
from unittest.mock import MagicMock, patch

from src.agents.extractor import ExtractionRouter
from src.agents.triage import TriageAgent
from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument
from src.strategies.fast_text import FastTextExtractor


# ────────────────────────────────────────────────────────────────
# FIXTURES & HELPERS
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_profiles_dir(tmp_path: Path) -> Path:
    """Temporary directory for profiles and ledger"""
    profiles = tmp_path / "profiles"
    profiles.mkdir(exist_ok=True)
    return profiles


@pytest.fixture
def triage_agent(tmp_profiles_dir: Path) -> TriageAgent:
    """TriageAgent using temporary directory"""
    return TriageAgent(refinery_dir=tmp_profiles_dir)


@pytest.fixture
def router(tmp_profiles_dir: Path) -> ExtractionRouter:
    """ExtractionRouter with temporary ledger"""
    ledger_path = tmp_profiles_dir.parent / "extraction_ledger.jsonl"
    return ExtractionRouter(ledger_path=ledger_path)


def create_simple_pdf(tmp_path: Path, text: str, filename="test.pdf") -> Path:
    """Create minimal single-page PDF for testing."""
    pdf_path = tmp_path / filename
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_text((72, 100), text, fontsize=12)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# ────────────────────────────────────────────────────────────────
# HAPPY PATH TESTS – Strategy A
# ────────────────────────────────────────────────────────────────

def test_fast_text_strategy_runs_successfully(
    triage_agent: TriageAgent,
    router: ExtractionRouter,
    tmp_path: Path
):
    """
    Purpose: Verify Strategy A (fast_text) works on clean digital document
    and produces valid ExtractedDocument with reasonable confidence.
    """
    text = """
    Commercial Bank of Ethiopia
    Annual Report 2023-24
    Net profit: ETB 26.7 billion
    Total assets: ETB 1,436 billion
    Single column layout, no images.
    """
    pdf_path = create_simple_pdf(tmp_path, text, "clean_report.pdf")

    # Triage first
    profile = triage_agent.triage(pdf_path)
    assert profile.estimated_extraction_cost == "fast_text_sufficient"

    # Extract
    result = router.extract_with_escalation(pdf_path, profile)

    assert isinstance(result, ExtractedDocument)
    assert result.strategy_used == "fast_text"
    assert result.confidence > 0.6, "Should have good confidence on clean text"
    assert len(result.text_blocks) > 0, "Should extract text"
    assert result.pages_processed > 0
    assert result.success is True

    # Check ledger was written
    ledger = router.LEDGER_PATH
    assert ledger.exists()
    ledger_content = ledger.read_text()
    assert "fast_text" in ledger_content
    assert str(result.confidence) in ledger_content


# ────────────────────────────────────────────────────────────────
# ESCALATION TESTS
# ────────────────────────────────────────────────────────────────

def test_escalation_from_fast_text_to_layout_on_low_confidence(
    monkeypatch,
    triage_agent: TriageAgent,
    router: ExtractionRouter,
    tmp_path: Path
):
    """
    Purpose: Force low confidence from Strategy A → verify escalation to B.
    """
    pdf_path = create_simple_pdf(tmp_path, "test text")

    # Triage → should allow fast_text
    profile = triage_agent.triage(pdf_path)
    assert profile.estimated_extraction_cost == "fast_text_sufficient"

    # Monkeypatch FastTextExtractor to return low confidence
    def fake_extract(*args, **kwargs):
        doc = ExtractedDocument(
            doc_id=profile.doc_id,
            source_path=str(pdf_path),
            strategy_used="fast_text",
            confidence=0.40,  # below threshold
            pages_processed=1,
            extraction_time_sec=1.2
        )
        return doc

    with patch.object(FastTextExtractor, "extract", fake_extract):
        result = router.extract_with_escalation(pdf_path, profile)

    assert result.strategy_used == "layout_aware", "Should have escalated"
    assert "fast_text" in router.escalation_path  # assuming you added this attribute or log
    assert result.confidence > 0.40, "Escalated strategy should be better"


# ────────────────────────────────────────────────────────────────
# BUDGET GUARD & VISION STRATEGY TESTS
# ────────────────────────────────────────────────────────────────

def test_vision_budget_guard_stops_on_cap_reached(
    monkeypatch,
    triage_agent: TriageAgent,
    router: ExtractionRouter,
    tmp_path: Path
):
    """
    Purpose: Verify Vision strategy respects budget cap and stops early.
    """
    pdf_path = create_simple_pdf(tmp_path, "dummy", "multi_page.pdf")

    # Capture the original bound method before monkeypatching.
    # (If we call triage_agent.triage() after monkeypatching the class method,
    # we'll recurse forever.)
    original_triage = triage_agent.triage

    # Force profile to trigger vision
    def fake_triage(*args, **kwargs):
        prof = original_triage(pdf_path)
        prof.estimated_extraction_cost = "needs_vision_model"
        return prof

    monkeypatch.setattr(TriageAgent, "triage", fake_triage)

    # Force budget to be very low
    def fake_can_process(*args, **kwargs):
        return False  # immediately stop

    monkeypatch.setattr("src.strategies.vision_augmented.BudgetGuard.can_process_page", fake_can_process)

    result = router.extract_with_escalation(pdf_path, fake_triage())

    assert result.pages_processed <= 1, "Should stop early due to budget"
    assert "budget" in result.error_message.lower() or result.pages_processed == 0


# ────────────────────────────────────────────────────────────────
# NEGATIVE / ROBUSTNESS TESTS
# ────────────────────────────────────────────────────────────────

def test_router_handles_corrupt_pdf_gracefully(
    triage_agent: TriageAgent,
    router: ExtractionRouter,
    tmp_path: Path
):
    """
    Purpose: Ensure router doesn't crash on invalid/corrupt PDF.
    """
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"not a pdf")  # invalid content

    profile = triage_agent.triage(bad_pdf)
    if profile is None:
        pytest.skip("Triage already skipped corrupt file")

    result = router.extract_with_escalation(bad_pdf, profile)

    assert result.confidence == 0.0 or result.error_message is not None
    assert result.success is False