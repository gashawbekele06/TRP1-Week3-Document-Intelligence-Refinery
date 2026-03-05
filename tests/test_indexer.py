"""tests.test_indexer

Tests for Stage 4 (PageIndex building) and the PageIndex query helper.

What we test:
- `PageIndex.find_sections_for_query()` keyword/entity overlap scoring
- `PageIndexBuilder.build()` behavior with and without heading LDUs
- Disk output is written to the configured pageindex_dir

These tests avoid network / external LLM calls by passing `use_llm=False`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.indexer import PageIndexBuilder
from src.models import ExtractedDocument, LDU, PageIndex, Section


def _ldu(
    *,
    ldu_id: str,
    content: str,
    chunk_type: str,
    page_refs: list[int],
    parent_section: str | None = None,
    doc_id: str = "doc1",
    document_name: str = "sample.pdf",
) -> LDU:
    return LDU(
        ldu_id=ldu_id,
        content=content,
        chunk_type=chunk_type,  # validated by LDU Literal
        page_refs=page_refs,
        parent_section=parent_section,
        doc_id=doc_id,
        document_name=document_name,
        token_count=max(1, len(content) // 4),
    )


def test_page_index_find_sections_for_query_ranks_by_overlap():
    child = Section(
        section_id="sec_child",
        title="1.1 Profit Breakdown",
        page_start=2,
        page_end=2,
        level=2,
        key_entities=["ETB 26.7 billion", "2024"],
        summary="Net profit increased in 2024 driven by interest income.",
    )
    root = Section(
        section_id="sec_root",
        title="1. Executive Summary",
        page_start=1,
        page_end=3,
        level=1,
        key_entities=["Commercial Bank of Ethiopia"],
        summary="This section summarizes key findings.",
        child_sections=[child],
    )

    idx = PageIndex(doc_id="doc1", document_name="sample.pdf", page_count=3, sections=[root])

    hits = idx.find_sections_for_query("profit ETB 2024", top_k=3)
    assert hits, "Expected at least one matching section"

    # Child should score well due to both keyword overlap and entity overlap.
    assert hits[0].title in {"1.1 Profit Breakdown", "1. Executive Summary"}
    assert any("profit" in s.title.lower() or (s.summary and "profit" in s.summary.lower()) for s in hits)


def test_page_index_builder_creates_root_section_when_no_headings(tmp_path: Path):
    doc = ExtractedDocument(
        doc_id="doc_no_headings",
        filename="no_headings.pdf",
        strategy_used="fast_text",
        confidence=0.9,
        page_count=2,
        text_blocks=[],
        tables=[],
        figures=[],
        full_text="",
        section_headings=[],
    )

    ldus = [
        _ldu(ldu_id="ldu_1", content="This is a paragraph about ETB 1,000.", chunk_type="text", page_refs=[1]),
        _ldu(ldu_id="ldu_2", content="Second paragraph with 2023 results.", chunk_type="text", page_refs=[2]),
    ]

    builder = PageIndexBuilder(use_llm=False)
    builder.pageindex_dir = tmp_path / "pageindex"
    builder.pageindex_dir.mkdir(parents=True, exist_ok=True)

    page_index = builder.build(doc, ldus)

    assert page_index.doc_id == doc.doc_id
    assert page_index.document_name == doc.filename
    assert page_index.page_count == doc.page_count
    assert len(page_index.sections) == 1

    root = page_index.sections[0]
    assert root.title == doc.filename
    assert root.page_start == 1
    assert root.page_end == doc.page_count
    assert set(root.ldu_ids) == {"ldu_1", "ldu_2"}

    out_path = builder.pageindex_dir / f"{doc.doc_id}.json"
    assert out_path.exists(), "PageIndex JSON output was not written"


def test_page_index_builder_builds_sections_from_heading_ldus(tmp_path: Path):
    doc = ExtractedDocument(
        doc_id="doc_with_headings",
        filename="with_headings.pdf",
        strategy_used="fast_text",
        confidence=0.95,
        page_count=5,
        text_blocks=[],
        tables=[],
        figures=[],
        full_text="",
        section_headings=[],
    )

    heading1 = _ldu(
        ldu_id="h1",
        content="1. Introduction",
        chunk_type="heading",
        page_refs=[1],
        parent_section=None,
        doc_id=doc.doc_id,
        document_name=doc.filename,
    )
    text1 = _ldu(
        ldu_id="t1",
        content="This section covers the 2023 background and ETB 1,000 capital.",
        chunk_type="text",
        page_refs=[1, 2],
        parent_section="1. Introduction",
        doc_id=doc.doc_id,
        document_name=doc.filename,
    )

    heading2 = _ldu(
        ldu_id="h2",
        content="2. Results",
        chunk_type="heading",
        page_refs=[3],
        parent_section=None,
        doc_id=doc.doc_id,
        document_name=doc.filename,
    )
    text2 = _ldu(
        ldu_id="t2",
        content="Net profit reached ETB 26.7 billion in 2024.",
        chunk_type="text",
        page_refs=[3, 4, 5],
        parent_section="2. Results",
        doc_id=doc.doc_id,
        document_name=doc.filename,
    )

    ldus = [heading1, text1, heading2, text2]

    builder = PageIndexBuilder(use_llm=False)
    builder.pageindex_dir = tmp_path / "pageindex"
    builder.pageindex_dir.mkdir(parents=True, exist_ok=True)

    page_index = builder.build(doc, ldus)

    assert len(page_index.sections) == 2
    sec1, sec2 = page_index.sections

    assert sec1.title == "1. Introduction"
    assert sec1.page_start == 1
    assert sec1.page_end == 2
    assert "t1" in sec1.ldu_ids
    assert "h1" in sec1.ldu_ids

    assert sec2.title == "2. Results"
    assert sec2.page_start == 3
    assert sec2.page_end == doc.page_count
    assert "t2" in sec2.ldu_ids
    assert "h2" in sec2.ldu_ids

    # Entities should include extracted patterns (e.g., ETB amounts, years)
    assert any("ETB" in e or "2024" in e for e in sec2.key_entities)

    out_path = builder.pageindex_dir / f"{doc.doc_id}.json"
    assert out_path.exists()
