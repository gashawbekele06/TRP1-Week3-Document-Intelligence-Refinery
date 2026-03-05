# DOMAIN_NOTES.md - Phase 0

Extraction Decision Tree:
- If char density > 0.5 and tables > 0 → Strategy B (layout-aware)
- If almost no text → Strategy C (vision)

Failure modes observed on CBE Annual Report 2023-24.pdf:
- Multi-column pages → raw text is jumbled
- Big tables → broken into flat strings (Structure Collapse)
- Tables split across pages → bad for RAG

Pipeline Diagram:
```mermaid
graph TD
    A[PDF Input] --> B[Triage Agent]
    B --> C[Extraction Router + Escalation]
    C --> D[Chunking Engine]
    D --> E[PageIndex + Vector Store]
    E --> F[Query Agent]


You have completed Phase 0 ✅

### PHASE 1: The Triage Agent & Document Profiling (Full Working Code)

**Create** `src/models/__init__.py`
```python
from pydantic import BaseModel
from typing import Literal, List, Optional

class DocumentProfile(BaseModel):
    doc_id: str
    origin_type: Literal["native_digital", "scanned_image", "mixed"]
    layout_complexity: Literal["single_column", "multi_column", "table_heavy", "mixed"]
    domain_hint: str = "financial"
    estimated_extraction_cost: Literal["fast_text_sufficient", "needs_layout_model", "needs_vision_model"]