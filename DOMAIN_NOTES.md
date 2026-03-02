# Domain Notes — Document Science Primer (Phase 0)

This file will capture observations from running `pdfplumber` and Docling across the corpus.

Planned contents:

- Extraction strategy decision tree (fast_text -> layout -> vision escalation)
- Failure modes observed (structure collapse, context poverty, provenance blindspots)
- Empirical thresholds used for triage and extraction (documented in `rubric/extraction_rules.yaml`)
- Pipeline diagram (Mermaid) — to be added after initial experiments

Notes:
- Fast text (pdfplumber) is high-speed and low-cost but fails on scanned PDFs and complex multi-column layouts.
- Layout-aware extractors (Docling/MinerU) are necessary for table fidelity and reading-order reconstruction.
- Vision-augmented extraction (VLMs) is the fallback for scanned and handwriting-heavy pages, but requires budget caps and careful prompting.

Next steps: run `src/agents/triage.py` over `data/` documents and collect `.refinery/profiles` for 12 documents (Phase 0 experiments).

Phase 3 experiments (chunking & pageindex):

- Implemented a ChunkingEngine that converts `ExtractedDocument` text blocks and tables into LDUs with content_hash and token_count.
- ChunkValidator enforces basic rules (table chunking preserved). More validation rules will be added as we iterate.
- PageIndexBuilder produces a simple section-level index using `LDU.parent_section` heuristics and a cheap summarizer (first ~50 words). Outputs saved to `.refinery/pageindex/`.
- Vector ingestion: a Chroma ingestion shim will be added; currently LDUs are persisted to `.refinery/ldus/` as JSONL.

Next actions for Phase 3:

- Integrate Docling/MinerU output into `ExtractedDocument` to improve chunking inputs.
- Replace cheap summarizer with a fast local LLM or API-backed summarizer with budget guard.
- Implement vector ingestion to Chroma/FAISS and confirm retrieval precision with and without PageIndex traversal.
