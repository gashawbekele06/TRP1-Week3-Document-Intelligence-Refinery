"""src.agents.query_agent

Stage 5: Query Interface Agent
==============================

This module implements the *front-end* of the refinery.

Requirements (from the rubric)
------------------------------
1. Provide three tools for retrieval:
    - ``pageindex_navigate``: navigate the PageIndex tree ("smart ToC")
    - ``semantic_search``: retrieve relevant chunks (vector retrieval in prod)
    - ``structured_query``: SQL over a FactTable for precise numerical queries
2. Every answer must include provenance:
    - document name
    - page number
    - bounding box (when available)
    - content hash
3. Support Audit Mode:
    - given a claim, verify it with citations or flag as not found/unverifiable

Design notes
------------
- The repo doesn't ship LangGraph/Chroma by default. To keep the agent usable
  and test-friendly, this module uses local, deterministic fallbacks from
  ``src.data`` (keyword search + SQLite).
- If a Gemini API key is present, the agent can optionally synthesize a final
  response with an LLM. Otherwise it returns a deterministic, citation-heavy
  answer.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, TypedDict

from src.data import AuditMode, FactTable, VectorStore
from src.models import PageIndex, ProvenanceChain

_PAGEINDEX_DIR = Path(".refinery") / "pageindex"


class QueryState(TypedDict):
    query: str
    doc_id: Optional[str]
    answer: str
    provenance: Optional[ProvenanceChain]
    tool_calls: list[str]


# ─── Tool Implementations ────────────────────────────────────────────────────

def pageindex_navigate(query: str, doc_id: Optional[str] = None) -> dict:
    """
    Tool 1: Navigate the PageIndex tree to find relevant sections.
    Returns top-3 sections + their LDU IDs for focused retrieval.
    """
    results = []

    index_files = list(_PAGEINDEX_DIR.glob("*.json"))
    if doc_id:
        index_files = [f for f in index_files if f.stem == doc_id]

    for idx_file in index_files:
        try:
            page_index = PageIndex.model_validate_json(idx_file.read_text())
            matching = page_index.find_sections_for_query(query, top_k=3)
            for section in matching:
                results.append(
                    {
                        "document": page_index.document_name,
                        "doc_id": page_index.doc_id,
                        "section_title": section.title,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                        "summary": section.summary,
                        "key_entities": section.key_entities,
                        "ldu_ids": section.ldu_ids,
                    }
                )
        except Exception:
            continue

    return {
        "tool": "pageindex_navigate",
        "query": query,
        "results": results[:3],
        "message": f"Found {len(results)} sections matching '{query}'",
    }


def semantic_search(query: str, k: int = 5, doc_id: Optional[str] = None) -> dict:
    """Tool 2: Semantic search over LDUs.

    In this repository we provide a deterministic fallback implementation that
    searches locally indexed LDUs (when available) under ``.refinery/ldus``.

    Returns passages and a :class:`~src.models.provenance.ProvenanceChain`.
    """
    store = VectorStore()

    hits = store.search(query, k=k, doc_id=doc_id)
    provenance = store.build_provenance(hits)

    passages: list[dict] = []
    for hit in hits:
        ldu = hit.ldu
        page = (ldu.page_refs[0] if ldu.page_refs else 1) or 1
        passages.append(
            {
                "content": (ldu.content or "")[:500],
                "document": ldu.document_name,
                "page": page,
                "section": ldu.parent_section or "",
                "chunk_type": ldu.chunk_type,
                "score": round(hit.score, 3),
                "content_hash": ldu.content_hash,
                "bbox": ldu.bounding_box.model_dump() if ldu.bounding_box else None,
                "ldu_id": ldu.ldu_id,
            }
        )

    return {
        "tool": "semantic_search",
        "query": query,
        "passages": passages,
        "provenance": provenance.model_dump(),
    }


def structured_query(sql: str) -> dict:
    """
    Tool 3: SQL query over the SQLite FactTable.
    Returns rows with document provenance.
    """
    ft = FactTable()
    try:
        # Safety: only allow SELECT statements
        if not sql.strip().upper().startswith("SELECT"):
            return {"tool": "structured_query", "error": "Only SELECT queries allowed.", "rows": []}

        rows = ft.query(sql)
        return {
            "tool": "structured_query",
            "sql": sql,
            "rows": rows,
            "count": len(rows),
        }
    except Exception as e:
        return {"tool": "structured_query", "sql": sql, "error": str(e), "rows": []}


# ─── LangGraph Query Agent ────────────────────────────────────────────────────

class QueryAgent:
    """
    LangGraph-based query agent that combines all 3 tools.
    Falls back to simple tool orchestration if LangGraph not available.
    """

    def __init__(self, use_llm: bool = True):
        # Accept either GEMINI_API_KEY or GOOGLE_API_KEY as the project currently
        # uses both names in different places.
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.use_llm = bool(use_llm and gemini_key)
        self.audit = AuditMode()
        self.facts = FactTable()

    def query(self, question: str, doc_id: Optional[str] = None) -> dict:
        """
        Answer a natural language question using multi-tool orchestration.
        Returns answer + ProvenanceChain.
        """
        if self.use_llm:
            return self._llm_query(question, doc_id)
        return self._deterministic_query(question, doc_id)

    @staticmethod
    def _format_citations(chain: ProvenanceChain, *, max_items: int = 5) -> str:
        """Format a short human-readable citations block.

        The structured `provenance` object is the primary audit artifact, but
        including a compact citations block in the answer helps ensure the
        requirement is met even for simple CLI/front-end callers.
        """
        if not chain.citations:
            return ""

        lines: list[str] = ["\nCitations:"]
        for c in chain.citations[:max_items]:
            bbox = ""
            if c.bbox is not None:
                b = c.bbox
                bbox = f" | bbox=({b.x0:.1f},{b.y0:.1f},{b.x1:.1f},{b.y1:.1f})"
            lines.append(
                f"- {c.document_name} p.{c.page_number}{bbox} | {c.content_hash}"
            )
        return "\n".join(lines)

    def _llm_query(self, question: str, doc_id: Optional[str] = None) -> dict:
        """LLM-orchestrated query using Gemini Flash."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Step 1: Navigate PageIndex for section context
            nav_result = pageindex_navigate(question, doc_id)
            section_context = ""
            if nav_result["results"]:
                top_sec = nav_result["results"][0]
                section_context = (
                    f"Most relevant section: {top_sec['section_title']} "
                    f"(pages {top_sec['page_start']}–{top_sec['page_end']})\n"
                    f"Summary: {top_sec.get('summary', '')}"
                )

            # Step 2: Semantic search for supporting passages
            search_result = semantic_search(question, k=5, doc_id=doc_id)
            passages_text = ""
            for p in search_result.get("passages", []):
                passages_text += (
                    f"\n[{p['document']}, p.{p['page']}]: {p['content'][:300]}\n"
                )

            # Step 3: Check FactTable for relevant facts
            # Extract potential number-related keywords
            keywords = re.findall(r"\b[A-Za-z][\w\s]{4,30}\b", question)
            fact_context = ""
            for kw in keywords[:2]:
                facts = self.facts.search_facts(kw[:20], doc_id)
                if facts:
                    fact_context += f"\nFact: {facts[0]['label']} = {facts[0]['value']} {facts[0].get('unit','')}\n"

            # Step 4: Synthesize answer
            prompt = f"""You are a document intelligence assistant. Answer the following question based ONLY on the provided document excerpts. Include specific page numbers in your answer.

Question: {question}

{section_context}

Document passages:
{passages_text}

{fact_context}

Answer concisely, cite specific page numbers, and note if information is not found in the documents."""

            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 500},
            )
            answer = response.text.strip()

            # Build provenance
            provenance_data = search_result.get("provenance", {})
            provenance = ProvenanceChain.model_validate(provenance_data) if provenance_data else ProvenanceChain()
            provenance.answer = answer

            citations_block = self._format_citations(provenance)
            if citations_block:
                answer = answer + "\n" + citations_block
                provenance.answer = answer

            return {
                "question": question,
                "answer": answer,
                "provenance": provenance.model_dump(),
                "pageindex_result": nav_result,
                "passages_used": len(search_result.get("passages", [])),
            }
        except Exception as e:
            return self._deterministic_query(question, doc_id)

    def _deterministic_query(self, question: str, doc_id: Optional[str] = None) -> dict:
        """Rule-based fallback when LLM is unavailable."""
        nav = pageindex_navigate(question, doc_id)
        search = semantic_search(question, k=5, doc_id=doc_id)

        # Optional: if the user asks a numeric question, include fact matches.
        fact_lines: list[str] = []
        for kw in re.findall(r"[A-Za-z]{3,}", question)[:3]:
            rows = self.facts.search_facts(kw, doc_id=doc_id, limit=3)
            for r in rows[:1]:
                fact_lines.append(
                    f"FactTable: {r.get('label','')} = {r.get('value','')} {r.get('unit','')} (p.{r.get('page_number','?')})"
                )

        answer_parts = []
        if nav["results"]:
            top = nav["results"][0]
            answer_parts.append(
                f"In section '{top['section_title']}' (pages {top['page_start']}–{top['page_end']}): "
                f"{top.get('summary', 'See original document')}"
            )

        for p in search.get("passages", [])[:2]:
            if p["similarity"] > 0.4:
                answer_parts.append(
                    f"From {p['document']}, page {p['page']}: {p['content'][:200]}"
                )

        if fact_lines:
            answer_parts.append("\n".join(fact_lines))

        answer = "\n\n".join(answer_parts) or "No relevant information found in the indexed content."

        provenance_data = search.get("provenance", {})
        provenance = ProvenanceChain.model_validate(provenance_data) if provenance_data else ProvenanceChain()
        citations_block = self._format_citations(provenance)
        if citations_block:
            answer = answer + "\n" + citations_block
        provenance.answer = answer

        return {
            "question": question,
            "answer": answer,
            "provenance": provenance.model_dump(),
            "pageindex_result": nav,
        }

    def verify_claim(self, claim: str, doc_id: Optional[str] = None) -> dict:
        """Audit mode: verify or refute a claim."""
        chain = self.audit.verify(claim, doc_id)
        return {
            "claim": claim,
            "verified": chain.verified,
            "audit_note": chain.audit_note,
            "citations": [c.model_dump() for c in chain.citations],
        }

    def index_fact_table(self, ldus, *, doc_id: str, document_name: str) -> int:
        """Extract and store facts from LDUs.

        This is a convenience hook for the pipeline to populate the FactTable.
        """
        try:
            return self.facts.extract_and_store_from_ldus(
                list(ldus),
                doc_id=doc_id,
                document_name=document_name,
            )
        except Exception:
            return 0
