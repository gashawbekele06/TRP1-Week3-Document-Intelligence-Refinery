from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.models.logical_document_unit import BBoxRef


class ProvenanceCitation(BaseModel):
    """A single source citation for one piece of evidence.

    Required by Phase 4 / Stage 5:
    Every answer must cite *where* it came from, including:
    - document name
    - page number
    - bounding box (when available)
    - content hash (stable identifier for the cited content)
    """

    model_config = ConfigDict()

    document_name: str = Field(..., description="Source document filename")
    page_number: int = Field(..., ge=1, description="1-based page number")
    bbox: Optional[BBoxRef] = Field(
        default=None,
        description="Bounding box in PDF points (if available)",
    )
    content_hash: str = Field(..., description="Hash of the cited content")

    # Optional-but-useful fields for richer audits/debugging.
    doc_id: str = ""
    file_path: str = ""
    ldu_id: str = ""
    snippet: str = ""
    strategy_used: str = ""
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ProvenanceChain(BaseModel):
    """
    Full audit trail for an answer — a list of source citations.
    Every answer from the Query Agent must carry a ProvenanceChain.
    """

    model_config = ConfigDict()

    citations: List[ProvenanceCitation] = Field(
        default_factory=list,
        description="Ordered evidence list supporting the answer",
    )
    answer: str = Field(default="", description="Final answer text")
    verified: Optional[bool] = Field(
        default=None,
        description="Audit mode result: True/False when a claim is verified/refuted",
    )
    audit_note: Optional[str] = Field(
        default=None,
        description="Explanation when verified is False/None",
    )

    def summary(self) -> str:
        """Human-readable provenance summary."""
        if not self.citations:
            return "⚠️  No source citations available."
        lines = [f"📄 {c.document_name}, page {c.page_number}" for c in self.citations]
        return "\n".join(lines)

    def add_citation(self, citation: ProvenanceCitation) -> None:
        """Append a citation to the chain."""
        self.citations.append(citation)
