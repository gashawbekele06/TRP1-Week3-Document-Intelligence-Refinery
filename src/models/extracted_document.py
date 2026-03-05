# src/models/extracted_document.py
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from pathlib import Path

class TextBlock(BaseModel):
    text: str
    page: int
    bbox: Optional[tuple[float, float, float, float]] = None  # x0,y0,x1,y1

class Table(BaseModel):
    page: int
    bbox: Optional[tuple[float, float, float, float]]
    headers: List[str]
    rows: List[List[Any]]

class ExtractedDocument(BaseModel):
    doc_id: str
    source_path: str
    strategy_used: str
    confidence: float                # 0.0–1.0
    text_blocks: List[TextBlock] = []
    tables: List[Table] = []
    pages_processed: int
    extraction_time_sec: float
    error_message: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.confidence >= 0.65 and self.error_message is None