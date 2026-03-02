from __future__ import annotations
from typing import List, Optional
from hashlib import sha256
from pathlib import Path
import json
from ..models.schemas import ExtractedDocument as ED, LDU as LDUModel, BoundingBox


def _approx_token_count(text: str) -> int:
    return max(1, len(text.split()))


class ChunkValidator:
    """Enforces chunking rules before emission."""

    def validate(self, ldu: LDUModel) -> bool:
        # Rule examples: table cells must include header (we assume chunk_type marks table)
        if ldu.chunk_type == "table_cell":
            if not ldu.parent_section:
                return False
        return True


class ChunkingEngine:
    def __init__(self, rules: dict | None = None):
        self.rules = rules or {}
        self.validator = ChunkValidator()

    def _content_hash(self, content: str, page_refs: List[int], bbox: Optional[BoundingBox]) -> str:
        b = bbox.json() if bbox else ""
        m = sha256()
        m.update(content.encode("utf-8"))
        m.update(str(page_refs).encode("utf-8"))
        m.update(b.encode("utf-8"))
        return m.hexdigest()

    def chunk_document(self, doc: ED) -> List[LDUModel]:
        ldus: List[LDUModel] = []
        # Strategy: group text_blocks into paragraph LDUs by page and nearest bbox
        for tb in doc.text_blocks:
            content = tb.text.strip()
            if not content:
                continue
            # detect header-like lines
            parent_section = None
            if content.isupper() or content.strip().split()[0].endswith('.') or content.strip().split()[0].isdigit():
                parent_section = content[:120]

            bbox = tb.bbox
            page_refs = [tb.page_number]
            token_count = _approx_token_count(content)
            ch = self._content_hash(content, page_refs, bbox)
            l = LDUModel(
                content=content,
                chunk_type="paragraph",
                page_refs=page_refs,
                bbox=bbox,
                parent_section=parent_section,
                token_count=token_count,
                content_hash=ch,
            )
            if self.validator.validate(l):
                ldus.append(l)

        # Tables: represent entire table as one LDU preserving headers+rows
        for t in doc.tables:
            content = " | ".join([", ".join(r) for r in ([t.headers] + t.rows)])
            token_count = _approx_token_count(content)
            ch = self._content_hash(content, [t.page_number], t.bbox)
            l = LDUModel(
                content=content,
                chunk_type="table",
                page_refs=[t.page_number],
                bbox=t.bbox,
                parent_section=None,
                token_count=token_count,
                content_hash=ch,
            )
            if self.validator.validate(l):
                ldus.append(l)

        return ldus

    def emit_ldus_jsonl(self, ldus: List[LDUModel], outpath: Path):
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w", encoding="utf-8") as f:
            for l in ldus:
                f.write(l.model_dump_json() + "\n")


if __name__ == "__main__":
    print("Chunker module")
