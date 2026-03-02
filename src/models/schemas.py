from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x0: float
    top: float
    x1: float
    bottom: float


class DocumentProfile(BaseModel):
    doc_id: str
    origin_type: str  # native_digital | scanned_image | mixed | form_fillable
    layout_complexity: str  # single_column | multi_column | table_heavy | figure_heavy | mixed
    language: Optional[str] = None
    language_confidence: Optional[float] = None
    domain_hint: Optional[str] = None  # financial | legal | technical | medical | general
    estimated_extraction_cost: Optional[str] = None  # fast_text_sufficient | needs_layout_model | needs_vision_model
    signals: Dict[str, Any] = Field(default_factory=dict)


class TextBlock(BaseModel):
    text: str
    bbox: BoundingBox
    page_number: int


class TableObject(BaseModel):
    headers: List[str]
    rows: List[List[str]]
    bbox: BoundingBox
    page_number: int


class ExtractedDocument(BaseModel):
    doc_id: str
    pages: int
    text_blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[TableObject] = Field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None


class LDU(BaseModel):
    content: str
    chunk_type: str
    page_refs: List[int]
    bbox: Optional[BoundingBox]
    parent_section: Optional[str]
    token_count: int
    content_hash: str


class ExtractionLedgerEntry(BaseModel):
    doc_id: str
    strategy_used: str
    confidence_score: float
    cost_estimate: float
    processing_time_s: float
    notes: Optional[str]


class ProvenanceItem(BaseModel):
    document_name: str
    page_number: int
    bbox: BoundingBox
    content_hash: str


class PageIndexNode(BaseModel):
    title: str
    page_start: int
    page_end: int
    child_sections: List["PageIndexNode"] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    data_types_present: List[str] = Field(default_factory=list)


PageIndexNode.update_forward_refs()


class PageIndex(BaseModel):
    doc_id: str
    root: PageIndexNode

