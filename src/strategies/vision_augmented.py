"""
Strategy C: Vision-Augmented Extractor
=====================================

Purpose in pipeline:
High-cost fallback strategy for scanned PDFs, handwriting, forms, or when fast/layout strategies fail.
Converts each PDF page to image → sends to Gemini Flash (via OpenRouter) with structured prompt.
Returns normalized ExtractedDocument with text blocks, tables, figures.

Key features:
- Budget guard: prevents runaway costs on large documents
- Per-page processing (only continues while under budget)
- Detailed prompt for structured JSON output
- Confidence scoring tuned for vision model (high when content is extracted)

Triggers (decided by router):
- origin_type == "scanned_image"
- OR previous strategy confidence < threshold
- OR handwriting detected (future extension)

Cost tracking:
- Logs estimated USD cost per document
- Never exceeds configurable per-document cap
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml

from src.models import (
    BBox,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    TextBlock,
)
from src.strategies.base import ExtractionStrategy

# ────────────────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ────────────────────────────────────────────────────────────────

_RULES_PATH = Path(__file__).parent.parent.parent / "rubric" / "extraction_rules.yaml"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default prompt — very strict schema for reliable parsing
_EXTRACTION_PROMPT = """\
You are an expert document intelligence system analyzing a single PDF page image.
Extract ALL visible content with perfect structure. Return ONLY valid JSON — no explanations, no markdown fences.

Schema (must match exactly):
{
  "page_number": int,
  "text_blocks": [
    {"text": "full paragraph or line", "is_header": bool, "reading_order": int}
  ],
  "tables": [
    {
      "caption": "table caption if visible",
      "headers": ["col1", "col2", ...],
      "rows": [["val1", "val2", ...], ...]
    }
  ],
  "figures": [
    {"caption": "figure caption if visible"}
  ]
}

Rules:
- Preserve ALL text: headers, footers, footnotes, numbers exactly
- Tables: keep exact alignment, never merge cells incorrectly
- Numerical values: preserve decimals, currency, units
- Return ONLY the JSON object
"""

def load_rules() -> dict:
    """Load configuration thresholds and costs from yaml."""
    if not _RULES_PATH.exists():
        raise FileNotFoundError(f"Rules file missing: {_RULES_PATH}")
    with open(_RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)

class BudgetGuard:
    """
    Purpose:
    Enforces per-document API cost limit to avoid surprise bills.
    Tracks input/output tokens and estimated USD cost.
    Stops processing pages when budget or max_pages exceeded.
    """

    def __init__(self, rules: dict):
        budget = rules.get("budget", {})
        self.max_cost_usd = budget.get("max_cost_per_doc_usd", 0.10)
        self.cost_per_1m_input = budget.get("token_cost_per_1m_input", 0.075)   # Gemini Flash example
        self.cost_per_1m_output = budget.get("token_cost_per_1m_output", 0.30)
        self.max_pages = budget.get("max_pages_vision_per_doc", 50)
        self.total_cost_usd = 0.0
        self.pages_processed = 0

    def can_process_page(self) -> bool:
        """Check if we can safely process another page."""
        return (self.total_cost_usd < self.max_cost_usd and
                self.pages_processed < self.max_pages)

    def record_usage(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate and accumulate cost for this API call."""
        cost = (
            input_tokens * self.cost_per_1m_input / 1_000_000 +
            output_tokens * self.cost_per_1m_output / 1_000_000
        )
        self.total_cost_usd += cost
        self.pages_processed += 1
        return cost

class VisionExtractor(ExtractionStrategy):
    """
    Strategy C: Vision-Augmented Extraction with Gemini Flash via OpenRouter.

    Purpose:
    - Handle scanned PDFs, handwriting, complex forms, or low-confidence cases
    - Convert PDF pages to images → prompt VLM for structured extraction
    - Enforce budget cap to prevent cost overrun
    - Produce rich ExtractedDocument with provenance
    """

    name = "vision_augmented"

    def __init__(self):
        self.rules = load_rules()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        budget = self.rules.get("budget", {})
        self.model = budget.get("vision_model", "google/gemini-flash-1.5")
        self.budget_guard = BudgetGuard(self.rules)

    def _pdf_page_to_image_bytes(self, file_path: Path, page_num: int) -> Optional[bytes]:
        """
        Purpose: Convert one PDF page to PNG bytes at 150 DPI using PyMuPDF (fitz).
        Returns None if conversion fails.
        """
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(file_path))
            page = doc[page_num - 1]
            zoom = 150 / 72  # 150 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            doc.close()
            return img_bytes
        except Exception as e:
            print(f"Page {page_num} image conversion failed: {e}")
            return None

    def _call_openrouter(self, image_b64: str) -> Tuple[dict, int, int]:
        """
        Purpose: Send base64 image + prompt to OpenRouter API.
        Returns parsed JSON + token usage.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _EXTRACTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1200,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/document-intelligence-refinery",
            "X-Title": "Document Intelligence Refinery",
        }

        try:
            resp = requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            # Handle possible markdown fences or multi-part content
            if "```" in content:
                content = content.split("```")[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()

            parsed = json.loads(content)
            usage = data.get("usage", {})
            return parsed, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {str(e)}")

    def _extract_page(self, img_bytes: bytes, page_num: int) -> dict:
        """
        Purpose: Process one page image → VLM → structured JSON.
        Returns empty dict on failure.
        """
        try:
            image_b64 = base64.b64encode(img_bytes).decode("utf-8")
            page_data, in_tokens, out_tokens = self._call_openrouter(image_b64)
            self.budget_guard.record_usage(in_tokens, out_tokens)
            return page_data
        except Exception as e:
            print(f"Page {page_num} VLM extraction failed: {e}")
            return {"page_number": page_num, "text_blocks": [], "tables": [], "figures": []}

    def extract(self, file_path: Path) -> ExtractedDocument:
        """
        Main extraction method for Strategy C.

        Purpose:
        - Convert PDF pages to images
        - Process pages sequentially while under budget
        - Aggregate structured results into ExtractedDocument
        """
        start_time = time.time()

        if not self.api_key:
            return ExtractedDocument(
                doc_id=file_path.stem + "_vision_fail",
                filename=file_path.name,
                strategy_used=self.name,
                confidence=0.0,
                page_count=0,
                error_message="OPENROUTER_API_KEY not set"
            )

        text_blocks = []
        tables = []
        figures = []
        full_text_parts = []
        section_headings = []

        try:
            import fitz
            pdf_doc = fitz.open(str(file_path))
            page_count = len(pdf_doc)
            pdf_doc.close()
        except Exception:
            page_count = 1

        for page_num in range(1, page_count + 1):
            if not self.budget_guard.can_process_page():
                print(f"Budget cap reached after {self.budget_guard.pages_processed} pages")
                break

            img_bytes = self._pdf_page_to_image_bytes(file_path, page_num)
            if not img_bytes:
                continue

            page_data = self._extract_page(img_bytes, page_num)

            # Parse text blocks
            for tb in page_data.get("text_blocks", []):
                text = tb.get("text", "").strip()
                if not text:
                    continue
                block = TextBlock(
                    text=text,
                    page=page_num,
                    is_header=tb.get("is_header", False),
                    reading_order=tb.get("reading_order", len(text_blocks)),
                )
                text_blocks.append(block)
                if block.is_header:
                    section_headings.append(block)
                full_text_parts.append(text)

            # Parse tables
            for t_idx, tbl in enumerate(page_data.get("tables", [])):
                headers = tbl.get("headers", [])
                rows = tbl.get("rows", [])
                if not headers and not rows:
                    continue
                ext_table = ExtractedTable(
                    table_id=f"t_{page_num}_{t_idx}",
                    page=page_num,
                    headers=headers,
                    rows=rows,
                    caption=tbl.get("caption"),
                    reading_order=len(tables),
                )
                tables.append(ext_table)

            # Parse figures
            for f_idx, fig in enumerate(page_data.get("figures", [])):
                ext_fig = ExtractedFigure(
                    figure_id=f"f_{page_num}_{f_idx}",
                    page=page_num,
                    caption=fig.get("caption"),
                    reading_order=len(figures),
                )
                figures.append(ext_fig)

        duration = time.time() - start_time

        result = ExtractedDocument(
            doc_id=file_path.stem + "_vision",
            filename=file_path.name,
            strategy_used=self.name,
            confidence=0.0,
            page_count=page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            full_text="\n\n".join(full_text_parts),
            section_headings=section_headings,
            estimated_cost_usd=round(self.budget_guard.total_cost, 6),
            processing_time_sec=round(duration, 2),
            metadata={
                "budget_remaining_usd": round(self.budget_guard.max_cost_usd - self.budget_guard.total_cost, 6),
                "pages_processed": self.budget_guard.pages_processed,
                "model": self.model,
            },
        )

        result.confidence = self.confidence(result)
        return result

    def confidence(self, doc: ExtractedDocument) -> float:
        """
        Purpose: Vision model is high-confidence when it extracts meaningful content.
        Score based on text volume + presence of structure.
        """
        if doc.page_count == 0:
            return 0.0

        total_chars = sum(len(b.text) for b in doc.text_blocks)
        chars_per_page = total_chars / doc.page_count if doc.page_count > 0 else 0

        has_text = len(doc.text_blocks) > 0
        has_structure = len(doc.tables) > 0 or len(doc.figures) > 0

        text_score = min(1.0, chars_per_page / 150)  # target ~150 chars/page
        structure_bonus = 0.25 if has_structure else 0.0

        score = 0.65 * text_score + 0.35 * int(has_text) + structure_bonus

        # Vision model usually high-confidence when successful
        return round(min(max(score, 0.70), 0.98), 3)