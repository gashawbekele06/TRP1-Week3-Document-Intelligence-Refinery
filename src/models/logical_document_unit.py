from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, Field
import hashlib


class BBoxRef(BaseModel):
    """
    Bounding Box Reference — Spatial location of an extracted element in PDF coordinates.
    
    Purpose:
    - Stores exact position (x0,y0,x1,y1) and page number
    - Used for provenance, audit trails, highlighting, and visual reconstruction
    - Bottom-left origin (standard PDF coordinate system)
    """

    x0: float = Field(..., description="Left edge coordinate (x0)")
    y0: float = Field(..., description="Bottom edge coordinate (y0)")
    x1: float = Field(..., description="Right edge coordinate (x1)")
    y1: float = Field(..., description="Top edge coordinate (y1)")
    page: int = Field(..., description="1-based page number where this element appears")


class LDU(BaseModel):
    """
    Logical Document Unit (LDU) — the fundamental RAG-ready chunk.

    Purpose in the pipeline:
    - Represents one semantically coherent, self-contained unit after chunking
    - Preserves structural context (section headers, table/figure integrity, lists)
    - Never splits logical elements (tables, figures+caption, numbered lists, etc.)
    - Carries full provenance for traceability, audit, and correct retrieval
    - Used directly for embedding, vector search, and PageIndex navigation

    This is the output of Phase 3 (Semantic Chunking Engine) and input to RAG / vector store.
    """

    model_config = ConfigDict(
        extra="forbid",  # Prevent accidental extra fields
        validate_assignment=True,
    )

    ldu_id: str = Field(
        ...,
        description="Unique identifier for this LDU (uuid or deterministic hash)"
    )
    content: str = Field(
        ...,
        description="Main readable content of the chunk (plain text or markdown for tables)"
    )
    chunk_type: Literal["text", "table", "figure", "list", "heading", "equation"] = Field(
        ...,
        description="Type of chunk — used for specialized handling in RAG / indexing"
    )

    # ── Provenance (critical for audit and reconstruction) ──────────────────────────────
    page_refs: List[int] = Field(
        default_factory=list,
        description="All pages this LDU spans (sorted)"
    )
    bounding_box: Optional[BBoxRef] = Field(
        None,
        description="Spatial bounding box of the main content (if available)"
    )
    parent_section: Optional[str] = Field(
        None,
        description="Heading text of the parent section this chunk belongs to"
    )
    doc_id: str = Field(
        default="",
        description="ID of the parent document (from Phase 1/2)"
    )
    document_name: str = Field(
        default="",
        description="Original filename of the source document"
    )

    # ── Size & Identity (for deduplication and token limits) ───────────────────────────
    token_count: int = Field(
        default=0,
        ge=0,
        description="Approximate token count (used to enforce max_tokens rule)"
    )
    content_hash: str = Field(
        default="",
        description="SHA-256 hash of normalized content (for provenance verification)"
    )

    # ── Relationships (for cross-references and context) ───────────────────────────────
    cross_references: List[str] = Field(
        default_factory=list,
        description="LDU IDs of related chunks (e.g. 'see Table 3' → table LDU ID)"
    )

    # ── Flexible metadata (for future extensions) ──────────────────────────────────────
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata (e.g. caption for figures, table_id, font info)"
    )

    @classmethod
    def compute_hash(cls, content: str) -> str:
        """
        Deterministic SHA-256 hash of normalized content.

        Purpose:
        - Enables provenance verification even if document pages shift
        - Used for deduplication and integrity checks
        """
        normalized = " ".join(content.split())  # normalize whitespace
        return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        """
        Automatically compute content_hash if not provided.
        Called after model initialization by Pydantic.
        """
        if not self.content_hash and self.content:
            self.content_hash = self.compute_hash(self.content)