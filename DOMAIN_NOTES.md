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

Extraction strategy decision tree
---------------------------------

```text
Start -> Triage(DocumentProfile)
	if origin_type == native_digital and layout == single_column and avg_chars_per_page > fast_text.min_chars_per_page:
		-> Strategy A: FastTextExtractor (pdfplumber)
			if confidence >= 0.5 -> accept
			else -> Strategy B: LayoutExtractor (Docling/MinerU)
	elif origin_type in (mixed, scanned_image) or layout in (multi_column, table_heavy):
		-> Strategy B: LayoutExtractor
			if confidence < 0.5 -> Strategy C: VisionExtractor (Docling OCR / VLM)
	else:
		-> Strategy C: VisionExtractor
```

Key thresholds (recorded in `rubric/extraction_rules.yaml`)
- fast_text.min_chars_per_page: 80 (pages with fewer chars likely scanned)
- fast_text.high_confidence_chars_per_page: 200 (high confidence)
- fast_text.max_image_area_ratio: 0.5 (if images dominate, escalate)
- layout.table_detection_pages_ratio: 0.1 (if tables present on >10% pages consider table_heavy)
- vision.budget_cap_per_document: 5.0 (USD-equivalent token budget)

Failure modes observed
----------------------
- Structure Collapse: pdfplumber frequently flattens multi-column layouts and produces paragraph bisections. Docling preserved reading order better on digital PDFs.
- Context Poverty: naive token chunking produced chunks that split tables; chunker rules were extended to preserve table rows and headers.
- Provenance Blindness: embedding-only search cannot localize a number to a page; we added content_hash + bbox metadata to every LDU.

Pipeline diagram (Mermaid)
-------------------------

```mermaid
flowchart TD
	A[Input PDFs/Images] --> B[Triage Agent]
	B --> C{Strategy Router}
	C -->|FastText| D[FastTextExtractor]
	C -->|Layout| E[LayoutExtractor (Docling)]
	C -->|Vision| F[VisionExtractor (Docling OCR / VLM)]
	D --> G[ExtractedDocument]
	E --> G
	F --> G
	G --> H[ChunkingEngine -> LDUs]
	H --> I[PageIndex Builder]
	H --> J[Vector Store / Chroma]
	I --> K[Query Agent]
	J --> K
	K --> L[Answers + ProvenanceChain]
```

Cost analysis (high-level estimates per document)
-----------------------------------------------
- Strategy A (FastText): low-cost, CPU-only, ~ $0.01 per doc (local compute)
- Strategy B (Layout/Docling): medium-cost, may use model-backed layout components, ~ $0.20 per doc (depends on models used)
- Strategy C (Vision/VLM): high-cost, VLM calls and image tokens, budget cap configurable (suggest default $1.0–5.0 per doc)

Phase 3 & 4 summary
-------------------
- ChunkingEngine produced LDUs with content_hash, bbox, token_count and enforced rules: preserve table cells, attach captions to figures, keep numbered lists together.
- PageIndexBuilder creates a navigable tree of sections with cheap LLM-free summaries (placeholder) and key_entities. These are stored in `.refinery/pageindex/`.
- QueryAgent provides three tools: `pageindex_navigate`, `semantic_search` (vector fallback), and `structured_query` (SQLite fact table). Every answer returns a `ProvenanceChain` with document name, page_number, bbox, and content_hash.

Artifacts generated in this run
-----------------------------
- `.refinery/profiles/` — DocumentProfile JSON for each processed document
- `.refinery/extraction_ledger.jsonl` — ledger of extraction runs (strategy, confidence, cost estimate)
- `.refinery/ldus/` — LDU JSONL files for each document
- `.refinery/pageindex/` — PageIndex JSON for each document
- `.refinery/facttable.db` — SQLite DB containing extracted numeric facts
- `.refinery/qa_examples.jsonl` and `.refinery/audit.jsonl` — 12 sample Q&A and audit runs produced by `scripts/ingest_12.py`

Next actions (recommended)
-------------------------
- Add a lightweight LLM (local or API) for PageIndex summaries (with a budget guard).
- Improve fact normalization (currency, thousands separators, dates) and deduplication.
- Integrate a VLM OCR service for hard scanned pages where Docling/rapidocr underperforms.

Phase 3 experiments (chunking & pageindex):

- Implemented a ChunkingEngine that converts `ExtractedDocument` text blocks and tables into LDUs with content_hash and token_count.
- ChunkValidator enforces basic rules (table chunking preserved). More validation rules will be added as we iterate.
- PageIndexBuilder produces a simple section-level index using `LDU.parent_section` heuristics and a cheap summarizer (first ~50 words). Outputs saved to `.refinery/pageindex/`.
- Vector ingestion: a Chroma ingestion shim will be added; currently LDUs are persisted to `.refinery/ldus/` as JSONL.

Next actions for Phase 3:

- Integrate Docling/MinerU output into `ExtractedDocument` to improve chunking inputs.
- Replace cheap summarizer with a fast local LLM or API-backed summarizer with budget guard.
- Implement vector ingestion to Chroma/FAISS and confirm retrieval precision with and without PageIndex traversal.
