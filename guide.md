# Document Intelligence Refinery — Complete Step-by-step Guide

This guide shows how to build the Document Intelligence Refinery from scratch (Phases 0–4), where to put code, what each component does, how it works, and how to verify it. Follow steps in order.

Prerequisites
- Linux/macOS
- Python 3.11+ (3.13 recommended)
- Git
- Optional: Docker

Quick setup (create repo, clone, venv)
1. Create a GitHub repo named `trp1-week3-document-intelligence-refinery`.
2. Clone and create folders:

```bash
git clone git@github.com:<you>/trp1-week3-document-intelligence-refinery.git
cd trp1-week3-document-intelligence-refinery
mkdir -p src/agents src/models src/strategies src/data src/utils tests rubric scripts .refinery data
```

3. Create and activate venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

4. Install minimal dependencies (use pyproject or requirements in your repo):

```bash
pip install pdfplumber pillow pydantic pyyaml langdetect docling chromadb langchain typer python-dotenv tqdm
```

Where to start (recommended order)
1. Core models: `src/models/schemas.py` — define all Pydantic models. Start here because all agents use these types.
2. Triage agent: `src/agents/triage.py` — profile documents. This is fast and gives insight into the corpus.
3. FastText extractor: `src/strategies/fasttext.py` — low-cost extractor for native PDFs.
4. Layout adapter: `src/strategies/layout.py` — integrate Docling/MinerU to get structured extraction.
5. Vision extractor: `src/strategies/vision.py` — OCR/VLM fallback for scanned docs.
6. Extraction router: `src/agents/extractor.py` — ties extractors together with escalation guard.
7. Chunker & indexer: `src/agents/chunker.py` and `src/agents/indexer.py` — build LDUs and PageIndex.
8. Fact table & query agent: `src/agents/facttable.py` and `src/agents/query_agent.py` — build structured facts, semantic search and audit.
9. Runner script: `scripts/ingest_12.py` — orchestrates an end-to-end run for N documents.

Detailed component guide (what, why, how, how to check)

1) src/models/schemas.py
- Purpose: canonical schemas used across the pipeline (DocumentProfile, ExtractedDocument, TextBlock, TableObject, BoundingBox, LDU, PageIndex, ProvenanceChain, ExtractionLedgerEntry).
- How it works: Pydantic models with serialization helpers (model_dump_json). All agents create/consume these models.
- How to check:
  - Run a quick import:
    ```bash
    python -c "import src.models.schemas as s; print([k for k in dir(s) if k.endswith('Model') or k in ['DocumentProfile','ExtractedDocument'] ])"
    ```

2) src/agents/triage.py
- Purpose: classify each input document and produce `DocumentProfile` that controls routing.
- How it works: opens PDF with pdfplumber, computes avg chars/page, image area ratio, naive column detection, table counts, language via langdetect, and keyword domain hints.
- How to run & check:
  ```bash
  python -c "from src.agents.triage import classify_document; p=classify_document('data/<DOC>.pdf'); print(p.model_dump_json(indent=2))"
  ls .refinery/profiles
  cat .refinery/profiles/<DOC>.pdf.json
  ```
  Expectation: `origin_type`, `layout_complexity`, and `estimated_extraction_cost` should be populated.

3) src/strategies/fasttext.py
- Purpose: fast, low-cost extraction for digital PDFs using pdfplumber.
- How it works: extracts paragraph-level text blocks, estimates bounding boxes by grouping words, extracts tables via pdfplumber; computes a multi-signal confidence score.
- How to run & check:
  ```bash
  python - <<'PY'
from src.strategies.fasttext import FastTextExtractor
res=FastTextExtractor('data/<DOC>.pdf').extract()
print(res['confidence'])
print(len(res['document'].text_blocks), len(res['document'].tables))
PY
  ```
  Expectation: `confidence` near 0.0–1.0, non-zero text_blocks for digital PDFs.

4) src/strategies/layout.py
- Purpose: layout-aware extraction using Docling (or MinerU) to recover blocks, tables, figures, reading order.
- How it works: uses Docling's API to load and parse a document and adapts Docling's document into `ExtractedDocument`.
- How to run & check:
  ```bash
  python - <<'PY'
from src.strategies.layout import LayoutExtractor
res=LayoutExtractor('data/<DOC>.pdf').extract()
print(res['confidence'])
print(res['document'].pages, len(res['document'].text_blocks), len(res['document'].tables))
PY
  ```
  Expectation: For table-heavy or multi-column docs, `layout` yields more structured tables/blocks than FastText.

5) src/strategies/vision.py
- Purpose: handle scanned/image-first PDFs via OCR or VLM.
- How it works: prefers Docling OCR; fallback to per-page pytesseract via pdfplumber image rendering.
- How to run & check:
  ```bash
  python - <<'PY'
from src.strategies.vision import VisionExtractor
res=VisionExtractor('data/<SCANNED_DOC>.pdf').extract()
print(res['confidence'], len(res['document'].text_blocks))
PY
  ```
  Expectation: scanned docs produce text_blocks when OCR is available.

6) src/agents/extractor.py
- Purpose: ExtractionRouter - selects extractor based on `DocumentProfile` and enforces escalation guard.
- How it works: reads profile, runs initial strategy, checks `confidence`, escalates when needed, writes ledger entry to `.refinery/extraction_ledger.jsonl`.
- How to run & check:
  ```bash
  python - <<'PY'
from src.agents.extractor import ExtractionRouter
r=ExtractionRouter('data/<DOC>.pdf')
res=r.run()
print(res['ledger'].model_dump_json())
PY
  tail -n 5 .refinery/extraction_ledger.jsonl
  ```

7) src/agents/chunker.py
- Purpose: turn `ExtractedDocument` into LDUs (Logical Document Units) that preserve structure and provenance.
- How it works: groups text blocks into paragraphs, treats each table as an LDU, enforces chunking rules (no split header/cell), computes `content_hash`.
- How to run & check:
  ```bash
  python - <<'PY'
from src.agents.chunker import ChunkingEngine
from src.strategies.fasttext import FastTextExtractor
doc_res=FastTextExtractor('data/<DOC>.pdf').extract()['document']
chunker=ChunkingEngine(); ldus=chunker.chunk_document(doc_res)
print(len(ldus)); chunker.emit_ldus_jsonl(ldus, Path('.refinery/ldus')/f"{doc_res.doc_id}_ldus.jsonl")
PY
  ls .refinery/ldus
  head -n 1 .refinery/ldus/<DOC>_ldus.jsonl
  ```

8) src/agents/indexer.py
- Purpose: build PageIndex (section tree) that speeds targeted retrieval for LLMs.
- How it works: groups LDUs by parent_section, extracts key_entities heuristically, creates summaries (cheap or LLM-backed), persists JSON to `.refinery/pageindex`.
- How to run & check:
  ```bash
  python - <<'PY'
from src.agents.indexer import PageIndexBuilder
from pathlib import Path
builder=PageIndexBuilder()
doc_res=... # as above
ldus=...    # produce via chunker
pi=builder.build(doc_res, ldus)
builder.persist(pi, Path('.refinery/pageindex'))
print('.refinery/pageindex/', list(Path('.refinery/pageindex').glob('*')))
PY
  ```

9) src/agents/facttable.py
- Purpose: extract numeric/fact key-values into a SQLite fact table to support structured queries and audit.
- How it works: regex numeric extraction from LDUs, inserts rows into `.refinery/facttable.db`.
- How to run & check:
  ```bash
  python - <<'PY'
from src.agents.facttable import extract_facts_from_ldus
print(extract_facts_from_ldus(ldus))
PY
  sqlite3 .refinery/facttable.db "select count(*) from facts;"
  sqlite3 .refinery/facttable.db "select doc_id,key,value,page from facts limit 10;"
  ```

10) src/agents/query_agent.py
- Purpose: orchestrate PageIndex navigation, semantic_search (vector), and structured_query; return answers with ProvenanceChain.
- How it works: given a query and optional doc_id, it first traverses PageIndex for relevance, then semantic_search over LDUs (Chroma/FAISS fallback), and assembles citations.
- How to run & check:
  ```bash
  python - <<'PY'
from src.agents.query_agent import answer_query, audit_claim
print(answer_query('What are the capital expenditure projections for Q3?', doc_id='<DOC>.pdf'))
print(audit_claim('revenue $', doc_id='<DOC>.pdf'))
PY
  ```
  Expectation: `answer_query` returns `answer` and `provenance.citations` with `document_name`, `page_number`, `bbox`, `content_hash`.

11) scripts/ingest_12.py
- Purpose: convenience runner to triage→extract→chunk→index→ingest→facts for N documents.
- How to run:
  ```bash
  python -c "from scripts.ingest_12 import process_files; process_files(12)"
  ```

Validation & end-to-end checks
- After running `ingest_12` you should see the `.refinery` artifacts:
  - `ls .refinery/profiles` — should contain one JSON per processed doc
  - `tail -n 5 .refinery/extraction_ledger.jsonl` — check strategies/confidences
  - `ls .refinery/ldus` / `head -n1` one of the JSONL LDUs
  - `ls .refinery/pageindex` / `cat` a pageindex JSON
  - `sqlite3 .refinery/facttable.db "select count(*) from facts;"`

How to test and debug
- Unit tests: add pytest tests under `tests/` and run `pytest -q`.
- If `pdfplumber` yields no text for a PDF, it is likely scanned — run the VisionExtractor directly or install `pytesseract`.
- Check `.refinery/extraction_ledger.jsonl` lines for confidence and notes.
- For Docling issues, ensure Docling is installed in venv and available (`pip install docling`) and review Docling logs.

Common troubleshooting commands
- Re-run triage on one file and inspect profile:
  ```bash
  python -c "from src.agents.triage import classify_document; print(classify_document('data/<DOC>.pdf').model_dump_json(indent=2))"
  ```
- Re-run extraction router and print ledger:
  ```bash
  python -c "from src.agents.extractor import ExtractionRouter; r=ExtractionRouter('data/<DOC>.pdf'); print(r.run()['ledger'].model_dump_json())"
  ```
- Re-run chunker/indexer for a doc and inspect LDU/pageindex files (see earlier commands).

Deliverables checklist (interim)
- `DOMAIN_NOTES.md` with decision tree and thresholds
- `rubric/extraction_rules.yaml` with thresholds
- `src/models/` Pydantic schemas
- `src/agents/triage.py` and tests
- `src/strategies/*` extractors
- `src/agents/extractor.py` and ledger
- `.refinery/profiles` and `.refinery/extraction_ledger.jsonl` for 12 docs
- README and guide (this file)

Deliverables checklist (final)
- All of interim plus: `.refinery/pageindex/`, `.refinery/ldus/`, `.refinery/facttable.db`, 12 Q&A with provenance, Dockerfile (optional), video demo.

Next steps (pick one)
- I can generate a one-page demo cue card, produce a Dockerfile + docker-compose, or assemble the final PDF report combining `DOMAIN_NOTES.md` + architecture diagrams.

If you want me to produce any of those, tell me which and I'll create it next.
