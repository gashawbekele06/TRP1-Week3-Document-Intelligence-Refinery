# Document Intelligence Refinery — Step-by-step guide

This guide walks you through building the Document Intelligence Refinery from scratch (Phases 0–4). Follow each step and run the commands in order.

Prerequisites
- Linux or macOS (Linux used here)
- Python 3.11+ (project uses 3.13 in pyproject but 3.11+ is fine)
- Git
- Optional: Docker (recommended for reproducible runs)

1. Create a GitHub repository
- On GitHub create a new repo (private or public) named `trp1-week3-document-intelligence-refinery`.
- Add a descriptive README and .gitignore for Python.

2. Clone the repository locally

```bash
git clone git@github.com:<your-org>/trp1-week3-document-intelligence-refinery.git
cd trp1-week3-document-intelligence-refinery
```

3. Create the folder structure

```text
LICENSE
README.md
pyproject.toml
data/                # place corpus PDFs here
src/
  agents/
  models/
  strategies/
  data/
  utils/
tests/
rubric/
.refinery/           # artifacts created during runs
scripts/
```

Create them with:

```bash
mkdir -p src/agents src/models src/strategies src/data src/utils tests rubric scripts .refinery
```

4. Create & activate a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

5. Add dependencies
- Use `pyproject.toml` or `requirements.txt`. Minimal list:

```toml
[project]
dependencies = [
  "pdfplumber", "pillow", "pydantic>=2", "pyyaml", "langdetect",
  "docling", "chromadb", "langchain", "typer", "python-dotenv",
  "tqdm"
]
```

Install them:

```bash
pip install pdfplumber pillow pydantic pyyaml langdetect docling chromadb langchain typer python-dotenv tqdm
```

6. Implement core Pydantic models
- File: `src/models/schemas.py`
- Define: `DocumentProfile`, `ExtractedDocument`, `TextBlock`, `TableObject`, `BoundingBox`, `LDU`, `PageIndex`, `ProvenanceItem`, `ProvenanceChain`, `ExtractionLedgerEntry`.

7. Phase 0 — Domain onboarding
- Read docs for MinerU, Docling, PageIndex, Chunkr, Marker. Take notes in `DOMAIN_NOTES.md`.
- Run quick experiments with `pdfplumber` and `docling` on `data/` files to measure:
  - chars per page
  - image area ratio
  - table detection rate
- Save a short pipeline diagram (Mermaid) in `DOMAIN_NOTES.md`.

Commands:

```bash
python - <<'PY'
import pdfplumber,glob
for f in glob.glob('data/*.pdf')[:5]:
  with pdfplumber.open(f) as pdf:
    print(f, len(pdf.pages), sum(len(p.extract_text() or '') for p in pdf.pages)//max(1,len(pdf.pages)))
PY
```

Deliverables (Phase 0): `DOMAIN_NOTES.md`, updated `rubric/extraction_rules.yaml`.

8. Phase 1 — Triage Agent (DocumentProfile)
- Implement `src/agents/triage.py`:
  - use `pdfplumber` to compute char density and image area ratio
  - implement `_detect_columns` heuristic
  - implement keyword-based `domain_hint`
  - save profile to `.refinery/profiles/{doc_id}.json`
- Unit tests: `tests/test_triage.py` (assert classification fields and thresholds)

9. Phase 2 — Multi-Strategy Extraction Engine
- Implement three extractors in `src/strategies/`:
  - `fasttext.py`: pdfplumber paragraph extraction, multi-signal confidence scoring
  - `layout.py`: integrate Docling/MinerU adapter to produce `ExtractedDocument`
  - `vision.py`: use Docling OCR when available; fallback to `pytesseract` via `pdfplumber` images
- Implement `src/agents/extractor.py` (ExtractionRouter) with escalation guard:
  - choose strategy from `DocumentProfile`
  - if `fast_text` confidence < threshold, escalate to `layout`; if `layout` low, escalate to `vision`
  - append entries to `.refinery/extraction_ledger.jsonl`

10. Phase 3 — Semantic Chunking & PageIndex
- Implement `src/agents/chunker.py`:
  - Convert `ExtractedDocument` into LDUs (Logical Document Units)
  - Enforce chunking rules: no split table cells, captions attached, lists kept together
  - Produce `content_hash` (sha256 of content+page+bbox)
  - Emit LDUs to `.refinery/ldus/{doc_id}_ldus.jsonl`
- Implement `src/agents/indexer.py`:
  - Build `PageIndex` tree nodes (title, page_start/end, key_entities, summary)
  - Use cheap summary (first 50 words) or an LLM with budget guard for summaries
  - Persist to `.refinery/pageindex/{doc_id}_pageindex.json`
- Ingest LDUs to vector store (Chroma or FAISS). Provide deterministic fallback embeddings for dev.

11. Phase 4 — Query Agent & Provenance
- Implement `src/agents/facttable.py`:
  - Simple numeric/key-value extraction into SQLite (`.refinery/facttable.db`)
  - Query API `query_facts(sql)`
- Implement `src/agents/query_agent.py`:
  - Tools: `pageindex_navigate(topic, doc_id)`, `semantic_search(query, doc_id)`, `structured_query(sql)`
  - `answer_query` orchestrator: use PageIndex to restrict, then semantic search, then return `ProvenanceChain` with citations
  - `audit_claim(claim, doc_id)` verifying claims against `facttable` and LDUs

12. Runner and artifacts
- Add `scripts/ingest_12.py` to run triage→extract→chunk→index→ingest→facts for N docs.
- Run:
  ```bash
  python -c "from scripts.ingest_12 import process_files; process_files(12)"
  ```
- Expected artifacts:
  - `.refinery/profiles/`
  - `.refinery/extraction_ledger.jsonl`
  - `.refinery/ldus/`
  - `.refinery/pageindex/`
  - `.refinery/facttable.db`
  - `.refinery/qa_examples.jsonl` and `.refinery/audit.jsonl`

13. Tests & QA
- Add unit tests for key components: triage, fasttext confidence, chunker rules, indexer build.
- Run tests:
  ```bash
  pip install pytest
  pytest -q
  ```

14. Demo & Deliverables
- Prepare a 5-minute video following the Demo Protocol. Use `guide.md` and artifacts.
- Produce the interim GitHub deliverable: include `DOMAIN_NOTES.md`, `rubric/extraction_rules.yaml`, `README.md`, `src/models/` and `src/agents/` code, `.refinery/profiles` and `extraction_ledger.jsonl` for 12 docs.
- Final deliverable: everything above plus `.refinery/pageindex/`, `.refinery/ldus/`, `.refinery/facttable.db`, Dockerfile (optional), and 12 Q&A examples with provenance.

15. Optional improvements (stretch)
- Replace cheap summaries with an LLM (budget-guarded) for PageIndex nodes.
- Integrate a VLM-based OCR (OpenRouter / OpenAI Vision) for difficult scanned pages.
- Implement fact normalization pipeline (currency, date parsing, units) and deduplication in `facttable`.
- Add a web UI or LangChain agent wrapper for interactive queries.

Support files you can copy from this repo as examples
- `src/agents/triage.py`, `src/strategies/fasttext.py`, `src/strategies/layout.py`, `src/strategies/vision.py`, `src/agents/extractor.py`, `src/agents/chunker.py`, `src/agents/indexer.py`, `src/agents/query_agent.py`, `scripts/ingest_12.py`, `DOMAIN_NOTES.md`, `rubric/extraction_rules.yaml`.

If you want, I can also:
- generate a one-page cue card for the demo,
- produce a Dockerfile + `docker-compose` for reproducible runs, or
- assemble the single PDF report for submission.

Good luck — tell me which optional add-on you'd like next and I'll prepare it.
