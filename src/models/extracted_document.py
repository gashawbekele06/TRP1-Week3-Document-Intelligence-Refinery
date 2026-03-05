# src/models/extracted_document.py
"""
Extracted Document Model – Normalized Output Format
==================================================

Purpose in the pipeline:
This file defines the **single unified schema** that **every extraction strategy** (A: fast_text, B: layout_aware, C: vision_augmented) must produce.

It is the critical bridge between:
- Stage 2 (Structure Extraction Layer)
- Stage 3 (Semantic Chunking Engine)
- Stage 4 (PageIndex Builder)

Every field carries enough provenance (page + bbox) for audit, provenance chain, and correct chunking.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class BBox(BaseModel):
    """
    Spatial bounding box in PDF points (bottom-left origin = 0,0).

    Purpose: Standard way to record exact location of any extracted element.
    Used for provenance, audit trails, and highlighting in original PDF.
    """

    x0: float = Field(..., description="Left edge")
    y0: float = Field(..., description="Bottom edge")
    x1: float = Field(..., description="Right edge")
    y1: float = Field(..., description="Top edge")
    page: int = Field(..., ge=1, description="1-based page number")

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height


class TextBlock(BaseModel):
    """
    A single contiguous text block extracted from the document.

    Purpose: Preserve text content + spatial/style metadata for accurate chunking and reading order.
    """

    text: str = Field(..., description="The extracted text content")
    bbox: Optional[BBox] = None
    page: int = Field(default=-1, ge=1)
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    is_header: bool = Field(default=False, description="Heuristically detected as heading")
    reading_order: int = Field(default=0, description="Approximate global reading order position")


class TableCell(BaseModel):
    """Detailed representation of one table cell (supports merged cells)."""

    value: str
    row: int
    col: int
    is_header: bool = False
    colspan: int = 1
    rowspan: int = 1


class ExtractedTable(BaseModel):
    """
    Structured table with spatial and semantic information.

    Purpose: Allow accurate table reconstruction, SQL extraction, and correct chunking (never split tables).
    """

    table_id: str
    bbox: Optional[BBox] = None
    page: int = Field(default=-1, ge=1)
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    cells: List[TableCell] = Field(default_factory=list)  # detailed cell-level info
    caption: Optional[str] = None
    reading_order: int = 0

    def to_markdown(self) -> str:
        """
        Render table as clean Markdown (useful for RAG embedding or display).

        Returns:
            str: Markdown table string
        """
        if not self.headers and not self.rows:
            return ""

        header_row = "| " + " | ".join(str(h) for h in self.headers) + " |"
        separator = "| " + " | ".join(["---"] * len(self.headers)) + " |"
        body_rows = [
            "| " + " | ".join(str(cell) for cell in row) + " |"
            for row in self.rows
        ]

        return "\n".join([header_row, separator] + body_rows)


class ExtractedFigure(BaseModel):
    """
    Extracted image/figure with caption and location.

    Purpose: Preserve figures + captions together (critical rule for semantic chunking).
    """

    figure_id: str
    bbox: Optional[BBox] = None
    page: int = Field(default=-1, ge=1)
    caption: Optional[str] = None
    image_bytes: Optional[bytes] = Field(
        default=None,
        description="Optional raw image bytes (only store if needed for downstream use)"
    )
    reading_order: int = 0


class ExtractedDocument(BaseModel):
    """
    Final normalized output of any extraction strategy.

    Purpose:
    - Every strategy (fast, layout, vision) must return this exact schema
    - Enables consistent chunking, indexing, provenance, and RAG
    - Carries cost, time, confidence for ledger and escalation decisions
    """

    model_config = ConfigDict(
        extra="forbid",               # prevent accidental extra fields
        validate_assignment=True,
        json_encoders={bytes: lambda v: "<binary image data>"},  # avoid dumping huge base64
    )

    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    strategy_used: Literal["fast_text", "layout_aware", "vision_augmented"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall reliability score")
    page_count: int = Field(..., ge=0)

    # Metadata fields commonly written/expected by the router and strategies
    source_path: Optional[str] = Field(
        default=None,
        description="Original file path (string)"
    )
    pages_processed: int = Field(
        default=0,
        ge=0,
        description="Number of pages actually processed by the strategy"
    )
    extraction_time_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Elapsed seconds spent extracting (alias-friendly name)"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Optional error message when extraction failed"
    )

    # Core extracted content
    text_blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[ExtractedTable] = Field(default_factory=list)
    figures: List[ExtractedFigure] = Field(default_factory=list)

    # Convenience fields for downstream
    full_text: str = Field(
        default="",
        description="Concatenated text in approximate reading order (for quick embedding)"
    )
    section_headings: List[TextBlock] = Field(
        default_factory=list,
        description="Detected top-level section titles (for PageIndex tree)"
    )

    # Execution metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    processing_time_sec: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time when this extraction was produced"
    )

    def is_successful(self) -> bool:
        """Used by escalation guard to decide if escalation is needed"""
        return self.confidence >= 0.65

    @property
    def success(self) -> bool:
        """Convenience property used by router; True when confidence meets threshold."""
        return self.is_successful()

    def summary(self) -> Dict[str, Any]:
        """Short summary for logging / debugging"""
        return {
            "doc_id": self.doc_id,
            "strategy": self.strategy_used,
            "confidence": round(self.confidence, 3),
            "pages": self.page_count,
            "text_blocks": len(self.text_blocks),
            "tables": len(self.tables),
            "figures": len(self.figures),
            "cost_usd": round(self.estimated_cost_usd, 4),
            "time_sec": round(self.processing_time_sec, 2),
        }