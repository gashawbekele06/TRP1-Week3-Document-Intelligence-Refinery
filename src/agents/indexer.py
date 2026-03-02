from __future__ import annotations
from typing import List
from pathlib import Path
import json
from .chunker import ChunkingEngine
from ..models.schemas import ExtractedDocument, PageIndex, PageIndexNode, LDU


def _cheap_summary(text: str, max_words: int = 50) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


class PageIndexBuilder:
    def __init__(self):
        self.chunker = ChunkingEngine()

    def build(self, doc: ExtractedDocument, ldus: List[LDU]) -> PageIndex:
        # Simple one-level index: root with child sections inferred from LDU.parent_section
        root = PageIndexNode(title=doc.doc_id, page_start=1, page_end=doc.pages)
        section_map: dict[str, PageIndexNode] = {}

        for l in ldus:
            sec = l.parent_section or "__body__"
            if sec not in section_map:
                node = PageIndexNode(title=sec[:120], page_start=min(l.page_refs), page_end=max(l.page_refs), child_sections=[], key_entities=[], summary=None, data_types_present=[])
                section_map[sec] = node
            else:
                node = section_map[sec]
                node.page_start = min(node.page_start, min(l.page_refs))
                node.page_end = max(node.page_end, max(l.page_refs))

            # collect simple key entities heuristically
            words = [w for w in l.content.split() if w.istitle() and len(w) > 3]
            for w in words[:5]:
                if w not in node.key_entities:
                    node.key_entities.append(w)

            if l.chunk_type == "table" and "table" not in node.data_types_present:
                node.data_types_present.append("tables")

            if not node.summary:
                node.summary = _cheap_summary(l.content, max_words=40)

        # attach children under root
        root.child_sections = list(section_map.values())
        pageindex = PageIndex(doc_id=doc.doc_id, root=root)

        return pageindex

    def persist(self, pageindex: PageIndex, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{pageindex.doc_id}_pageindex.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(pageindex.model_dump_json(indent=2))

    def ingest_ldus(self, ldus: List[LDU], collection_name: str = None):
        """Attempt to ingest LDUs into ChromaDB if available; otherwise persist to JSONL as a fallback.
        Embeddings are generated via a deterministic hash -> float vector fallback when no embedding service is available.
        """
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.Client(Settings())
            name = collection_name or "ldus"
            try:
                col = client.get_collection(name)
            except Exception:
                col = client.create_collection(name)

            ids = []
            metadatas = []
            documents = []
            vectors = []
            for l in ldus:
                ids.append(l.content_hash)
                metadatas.append({"page_refs": l.page_refs, "chunk_type": l.chunk_type})
                documents.append(l.content)
                # deterministic fake embedding: use sha256 digest split into floats
                import hashlib
                h = hashlib.sha256(l.content.encode('utf-8')).digest()
                vec = [b / 255.0 for b in h[:64]]
                vectors.append(vec)

            col.add(ids=ids, metadatas=metadatas, documents=documents, embeddings=vectors)
            return {"status": "ok", "collection": name, "count": len(ids)}
        except Exception:
            # fallback: persist to .refinery/ldus
            out = Path('.refinery/ldus')
            out.mkdir(parents=True, exist_ok=True)
            path = out / "fallback_ldus.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                for l in ldus:
                    f.write(l.model_dump_json() + "\n")
            return {"status": "fallback", "path": str(path), "count": len(ldus)}
