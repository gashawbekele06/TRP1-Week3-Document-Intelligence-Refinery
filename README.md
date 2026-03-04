# Document Intelligence Refinery — TRP1 Week 3

This repository implements a multi-stage document intelligence pipeline: triage → multi-strategy extraction → semantic chunking → PageIndex → query agent with provenance.

Requirements
- Python 3.13
- venv (see pyproject.toml for dependencies)

Quick setup

1. Create and activate the venv (if not already):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r <(python -c "import tomllib,sys;print('\n'.join([d for d in __import__('tomllib').loads(open('pyproject.toml','rb').read())['project']['dependencies']]))")
```

2. Run the ingestion (example for first 12 documents):

```bash
source .venv/bin/activate
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from scripts.ingest_12 import process_files
process_files(12)
PY
```

Artifacts
- `.refinery/profiles/` — DocumentProfile JSONs
- `.refinery/extraction_ledger.jsonl`
- `.refinery/ldus/` — LDU JSONL per document
- `.refinery/pageindex/` — PageIndex JSON per document
- `.refinery/facttable.db` — SQLite facts
- `.refinery/qa_examples.jsonl` — sample Q&A
- `.refinery/audit.jsonl` — audit outputs

Running tests

```bash
# with venv active
pip install pytest
pytest -q
```

Notes
- Docling is used when available for layout-aware parsing. The `LayoutExtractor` falls back to pdfplumber tables when Docling is absent.
- `VisionExtractor` uses Docling OCR when available; otherwise it falls back to pytesseract (if installed).
- The repository includes stubs for production concerns: budget_guard, ledger logging, and deterministic fallback embeddings for local experimentation.
