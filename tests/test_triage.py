"""
Unit Tests for Triage Agent (Phase 1)
====================================

Purpose:
Verify that the TriageAgent correctly classifies documents according to the five required dimensions:
- origin_type
- layout_complexity
- language_code & language_confidence
- domain_hint
- estimated_extraction_cost

Tests cover:
- Happy path: clean native-digital single-column document
- Edge case: forced scanned image (via monkeypatch)
- Negative cases: empty/corrupt PDF, no text
- Caching behavior

Run with: pytest test_triage.py -v
"""

import fitz  # PyMuPDF
import pytest
from pathlib import Path
from typing import Generator

from src.agents.triage import TriageAgent
from src.models.document_profile import DocumentProfile


# ────────────────────────────────────────────────────────────────
# FIXTURES & HELPERS
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def triage(tmp_path: Path) -> Generator[TriageAgent, None, None]:
    """Create TriageAgent with temporary profile directory."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    agent = TriageAgent(refinery_dir=profiles_dir)
    yield agent


def create_minimal_pdf(tmp_path: Path, text: str, filename: str = "test.pdf") -> Path:
    """
    Create a small single-page PDF with given text using PyMuPDF.
    Used for controlled test inputs.
    """
    out_path = tmp_path / filename
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # US Letter
    page.insert_textbox((72, 72, 540, 720), text, fontsize=12)
    doc.save(out_path)
    doc.close()
    return out_path


def create_two_column_pdf(tmp_path: Path, left_text: str, right_text: str, filename: str = "two_col.pdf") -> Path:
    """Create a single-page PDF with two text columns.

    This is used to validate that the multi-column heuristic can detect
    real 2-column layouts without relying on external corpus PDFs.
    """
    out_path = tmp_path / filename
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    # Insert line-by-line to avoid textbox overflow (which can result in very
    # little or no extractable text, depending on fit).
    left_lines = left_text.splitlines()
    right_lines = right_text.splitlines()
    n = max(len(left_lines), len(right_lines))
    y0 = 80
    dy = 11
    for i in range(min(n, 55)):
        y = y0 + i * dy
        if i < len(left_lines):
            page.insert_text((60, y), left_lines[i], fontsize=10)
        if i < len(right_lines):
            page.insert_text((330, y), right_lines[i], fontsize=10)

    doc.save(out_path)
    doc.close()
    return out_path


# ────────────────────────────────────────────────────────────────
# HAPPY PATH TESTS
# ────────────────────────────────────────────────────────────────

def test_triage_native_digital_single_column(triage: TriageAgent, tmp_path: Path):
    """
    Purpose: Verify correct classification for clean digital text PDF.
    Expected: native_digital, single_column, financial domain, fast_text_sufficient
    """
    text = """
    Annual Financial Report 2023-24
    Revenue: 137 billion Birr
    Profit before tax: 26.7 billion Birr
    Balance sheet shows total assets of 1,436 billion Birr.
    This report is prepared in single-column format with no images.
    """
    pdf_path = create_minimal_pdf(tmp_path, text, "financial_report.pdf")

    profile: DocumentProfile = triage.triage(pdf_path)

    assert profile.origin_type == "native_digital", "Should detect selectable text"
    assert profile.layout_complexity == "single_column", "Single-column layout"
    assert profile.domain_hint == "financial", "Keyword-based domain detection"
    assert profile.estimated_extraction_cost == "fast_text_sufficient", "Low-cost path"
    assert profile.language_code in ("en", "und"), "English or undetermined"
    assert profile.language_confidence > 0.7, "Reasonable language confidence"

    # Check caching
    cached_file = triage.refinery_dir / f"{profile.doc_id}.json"
    assert cached_file.exists(), "Profile should be cached"
    assert cached_file.read_text().strip() != "", "Cached file not empty"


# ────────────────────────────────────────────────────────────────
# EDGE / NEGATIVE CASE TESTS
# ────────────────────────────────────────────────────────────────

def test_triage_scanned_image_forces_vision(monkeypatch, triage: TriageAgent, tmp_path: Path):
    """
    Purpose: Verify that low density + high image ratio forces 'scanned_image'
    and 'needs_vision_model' — simulates scanned document.
    """
    pdf_path = create_minimal_pdf(tmp_path, "filler text", "scanned_sim.pdf")

    # Monkeypatch signals to emulate scanned page
    monkeypatch.setattr("src.agents.triage._compute_char_density", lambda _: 0.02)
    monkeypatch.setattr("src.agents.triage._compute_image_ratio", lambda _: 0.92)
    monkeypatch.setattr("src.agents.triage._has_font_metadata", lambda _: False)

    profile = triage.triage(pdf_path)

    assert profile.origin_type == "scanned_image", "Should detect image-heavy / low-text"
    assert profile.estimated_extraction_cost == "needs_vision_model", "Should escalate to vision"
    assert profile.char_density_mean < 0.1, "Low density preserved in profile"
    assert profile.image_ratio_mean > 0.8, "High image ratio preserved"


def test_triage_empty_pdf_returns_none_or_error(triage: TriageAgent, tmp_path: Path):
    """
    Purpose: Ensure graceful handling of empty/corrupt PDFs.
    Expected: either raises meaningful error or returns None / low-confidence profile
    """
    empty_pdf = tmp_path / "empty.pdf"
    # PyMuPDF does not allow saving zero-page PDFs; create a zero-byte file to
    # simulate a corrupt/empty PDF instead.
    empty_pdf.write_bytes(b"")

    profile = triage.triage(empty_pdf)

    assert profile is not None, "Should not crash on empty PDF"
    assert profile.page_count == 0, "Page count should be 0"
    assert profile.confidence < 0.1, "Very low confidence for empty document"


def test_triage_no_text_forces_mixed_or_scanned(triage: TriageAgent, tmp_path: Path):
    """
    Purpose: Test document with images but no text layer (simulates scanned).
    """
    pdf_path = tmp_path / "image_only.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Insert dummy image (small rectangle)
    page.draw_rect(fitz.Rect(100, 100, 200, 200), color=(1, 0, 0))
    doc.save(pdf_path)
    doc.close()

    # Force signals
    with pytest.MonkeyPatch.context() as m:
        m.setattr("src.agents.triage._compute_char_density", lambda _: 0.01)
        m.setattr("src.agents.triage._compute_image_ratio", lambda _: 0.70)
        profile = triage.triage(pdf_path)

    assert profile.origin_type in ("scanned_image", "mixed")
    assert profile.estimated_extraction_cost in ("needs_vision_model", "needs_layout_model")


def test_triage_image_heavy_but_fonts_is_native_digital(monkeypatch, triage: TriageAgent, tmp_path: Path):
    """Annual-report style PDFs can be image-heavy but still native-digital.

    Regression test:
    - High image ratio alone should not force origin_type to "mixed"/"scanned_image"
      when font metadata and reasonable text density are present.
    """
    pdf_path = create_minimal_pdf(tmp_path, "Annual report text", "annual_like.pdf")

    monkeypatch.setattr("src.agents.triage._compute_char_density", lambda _: 3.0)
    monkeypatch.setattr("src.agents.triage._compute_image_ratio", lambda _: 0.75)
    monkeypatch.setattr("src.agents.triage._has_font_metadata", lambda _: True)

    profile = triage.triage(pdf_path)
    assert profile.origin_type == "native_digital"
    assert profile.estimated_extraction_cost in ("fast_text_sufficient", "needs_layout_model")


def test_triage_detects_multi_column_layout(triage: TriageAgent, tmp_path: Path):
    """Two-column text should be detected as multi_column and route to layout model."""
    left = "\n".join(["Left column text"] * 120)
    right = "\n".join(["Right column text"] * 120)
    pdf_path = create_two_column_pdf(tmp_path, left, right, "two_column.pdf")

    profile = triage.triage(pdf_path)

    assert profile.origin_type == "native_digital"
    assert profile.layout_complexity == "multi_column"
    assert profile.estimated_extraction_cost == "needs_layout_model"


def test_triage_scanned_even_with_fonts(monkeypatch, triage: TriageAgent, tmp_path: Path):
    """OCR'd scans may include font metadata but should still be scanned_image."""
    pdf_path = create_minimal_pdf(tmp_path, "scan text", "ocr_scan.pdf")

    monkeypatch.setattr("src.agents.triage._compute_char_density", lambda _: 0.02)
    monkeypatch.setattr("src.agents.triage._compute_image_ratio", lambda _: 0.90)
    monkeypatch.setattr("src.agents.triage._has_font_metadata", lambda _: True)

    profile = triage.triage(pdf_path)
    assert profile.origin_type == "scanned_image"
    assert profile.estimated_extraction_cost == "needs_vision_model"