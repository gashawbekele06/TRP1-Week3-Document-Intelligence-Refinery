# src/models/document_profile.py
"""
Pydantic model for DocumentProfile - output of Stage 1 (Triage Agent).
Contains all five required classification dimensions + audit signals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentProfile(BaseModel):
    """
    Classification result from the Triage Agent.
    Governs which extraction strategy downstream stages will use.
    Saved as JSON in .refinery/profiles/{doc_id}.json
    """

    model_config = ConfigDict(
        extra="forbid",           # reject any fields not explicitly defined
        populate_by_name=True,    # allow field aliases if needed later
        json_schema_extra={
            "title": "DocumentProfile",
            "description": "Full document classification for intelligent extraction routing"
        }
    )

    # ── Identification & Metadata ─────────────────────────────────────────────
    doc_id: str = Field(
        ...,
        description="Deterministic short ID (e.g. truncated SHA-256 of path + size)"
    )
    filename: str = Field(..., description="Original filename of the document")
    file_path: str = Field(..., description="Absolute path to the source file")
    page_count: int = Field(..., ge=0, description="Total number of pages")

    # ── Classification Dimensions (exact names from challenge spec) ───────────
    origin_type: Literal["native_digital", "scanned_image", "mixed", "form_fillable"] = Field(
        ..., description="Whether the PDF has selectable text vs image-based content"
    )
    layout_complexity: Literal[
        "single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"
    ] = Field(
        ..., description="Structural layout type (columns, tables, figures, etc.)"
    )
    language_code: str = Field(
        default="und",
        description="Detected language (ISO 639-1 code: 'en', 'am', 'und', etc.)"
    )
    language_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of language detection (0.0–1.0)"
    )
    domain_hint: Literal["financial", "legal", "technical", "medical", "general"] = Field(
        ..., description="Inferred domain for prompt/strategy selection"
    )

    # ── Extraction Strategy Recommendation ─────────────────────────────────────
    estimated_extraction_cost: Literal[
        "fast_text_sufficient",
        "needs_layout_model",
        "needs_vision_model"
    ] = Field(
        ..., description="Recommended tier for the Structure Extraction Layer"
    )

    # ── Raw Signals (audit, debugging, future tuning) ──────────────────────────
    char_density_mean: float = Field(
        default=0.0,
        ge=0.0,
        description="Average characters per 1000 pt² across sampled pages"
    )
    image_ratio_mean: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average fraction of page area covered by images"
    )
    table_count_total: int = Field(
        default=0,
        ge=0,
        description="Total number of detected tables (across sampled pages)"
    )
    has_font_metadata: bool = Field(
        default=False,
        description="PDF contains embedded font/encoding information"
    )
    is_form_fillable: bool = Field(
        default=False,
        description="PDF contains interactive fillable form fields (AcroForm/XFA)"
    )

    # ── Audit & Traceability ───────────────────────────────────────────────────
    triage_version: str = Field(
        default="1.0",
        description="Version of the triage logic used"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this profile was generated"
    )
    # Overall confidence score for the triage decisions (0.0 - 1.0)
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence for the triage classification"
    )