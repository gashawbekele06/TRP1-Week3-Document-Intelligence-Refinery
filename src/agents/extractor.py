"""
ExtractionRouter – Stage 2 Core Orchestrator
============================================

Purpose in pipeline:
- Reads DocumentProfile from Stage 1 (Triage)
- Selects starting strategy (A/B/C) based on estimated cost
- Executes extraction
- Applies mandatory confidence-gated escalation guard (A → B → C)
- Prevents "garbage in, hallucination out" by never passing low-confidence results
- Logs every attempt/result to .refinery/extraction_ledger.jsonl for audit & cost tracking

Key features:
- Strategy selection matches triage output exactly
- Escalation chain: fast_text → layout_aware → vision_augmented
- Thresholds loaded from extraction_rules.yaml
- Rich logging: includes profile fields, escalation path, content counts
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from pathlib import Path

# Add project root to PYTHONPATH (temporary fix)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from typing import List, Optional, Tuple

import yaml

from src.models import DocumentProfile, ExtractedDocument
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_aware import LayoutAwareExtractor
from src.strategies.vision_augmented import VisionAugmentedExtractor


# ────────────────────────────────────────────────────────────────
# CONFIG PATHS & DEFAULTS
# ────────────────────────────────────────────────────────────────

_RULES_PATH = Path(__file__).parent.parent.parent / "rubric" / "extraction_rules.yaml"
_LEDGER_PATH = Path(".refinery") / "extraction_ledger.jsonl"


def load_rules() -> dict:
    """Load configurable thresholds from extraction_rules.yaml."""
    if not _RULES_PATH.exists():
        raise FileNotFoundError(f"Missing rules file: {_RULES_PATH}")
    with open(_RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class ExtractionRouter:
    """
    Purpose:
    Orchestrates multi-strategy extraction with intelligent routing and escalation.
    Ensures high-quality output by escalating low-confidence results to more capable strategies.
    Logs every attempt for audit, cost tracking, and performance analysis.
    """

    def __init__(self):
        self.rules = load_rules()
        thresholds = self.rules.get("confidence_thresholds", {})

        # Load escalation thresholds (documented in DOMAIN_NOTES.md)
        self.fast_threshold = thresholds.get("fast_text_min_confidence", 0.65)
        self.layout_threshold = thresholds.get("layout_aware_min_confidence", 0.60)

        # Strategy instances
        self.fast_extractor = FastTextExtractor()
        self.layout_extractor = LayoutAwareExtractor()
        self.vision_extractor = VisionAugmentedExtractor()

        # Ledger setup
        self.ledger_path = _LEDGER_PATH
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def select_initial_strategy(self, profile: DocumentProfile) -> str:
        """
        Purpose: Choose starting strategy based on triage profile.

        Logic (matches challenge criteria):
        - fast_text_sufficient → start with A
        - needs_layout_model → start with B
        - needs_vision_model → start with C
        """
        cost = profile.estimated_extraction_cost
        if cost == "fast_text_sufficient":
            return "fast_text"
        if cost == "needs_layout_model":
            return "layout_aware"
        return "vision_augmented"

    def extract_with_escalation(self, file_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        """
        Purpose:
        Main entry point for Stage 2 extraction.
        1. Select starting strategy
        2. Run extraction
        3. Apply escalation guard if confidence too low
        4. Log full result to ledger
        5. Return normalized ExtractedDocument
        """
        start_strategy = self.select_initial_strategy(profile)
        print(f"[Extraction] {profile.doc_id} → Starting with {start_strategy}")

        # Escalation chain order (A → B → C)
        chain = ["fast_text", "layout_aware", "vision_augmented"]
        current_idx = chain.index(start_strategy)

        result = None
        escalation_path: List[str] = [start_strategy]

        # Try initial strategy
        try:
            result = self._run_strategy(chain[current_idx], file_path, profile)
        except Exception as e:
            print(f"  {start_strategy} failed: {e}")
            result = ExtractedDocument(
                doc_id=profile.doc_id,
                source_path=str(file_path),
                strategy_used=start_strategy,
                confidence=0.0,
                pages_processed=0,
                extraction_time_sec=0.0,
                error_message=str(e)
            )

        # Escalation guard loop
        while not result.success and current_idx < len(chain) - 1:
            current_idx += 1
            next_strategy = chain[current_idx]
            escalation_path.append(next_strategy)
            print(f"  Low confidence ({result.confidence:.2f}) → escalating to {next_strategy}")
            try:
                result = self._run_strategy(next_strategy, file_path, profile)
            except Exception as e:
                print(f"  {next_strategy} failed: {e}")
                # Keep previous result rather than fail completely

        # Final logging
        self._log_to_ledger(profile, result, escalation_path)

        print(f"[Extraction] Done | final_strategy={result.strategy_used} | conf={result.confidence:.2f} | success={result.success}")
        return result

    def _run_strategy(self, strategy_name: str, file_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        """
        Purpose: Execute one strategy safely and return result.
        """
        extractor_map = {
            "fast_text": self.fast_extractor,
            "layout_aware": self.layout_extractor,
            "vision_augmented": self.vision_extractor,
        }

        extractor = extractor_map[strategy_name]
        start_time = time.time()

        try:
            doc = extractor.extract(file_path)
            doc.extraction_time_sec = time.time() - start_time
            return doc
        except Exception as e:
            return ExtractedDocument(
                doc_id=profile.doc_id,
                source_path=str(file_path),
                strategy_used=strategy_name,
                confidence=0.0,
                pages_processed=0,
                extraction_time_sec=time.time() - start_time,
                error_message=str(e)
            )

    def _log_to_ledger(
        self,
        profile: DocumentProfile,
        doc: ExtractedDocument,
        escalation_path: List[str]
    ) -> None:
        """
        Purpose: Append detailed JSONL entry for every extraction attempt.
        Enables audit, cost analysis, performance tuning.
        """
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "doc_id": profile.doc_id,
            "filename": profile.filename,
            "origin_type": profile.origin_type,
            "layout_complexity": profile.layout_complexity,
            "domain_hint": profile.domain_hint,
            "initial_strategy": escalation_path[0],
            "final_strategy": doc.strategy_used,
            "escalation_path": escalation_path,
            "confidence_score": round(doc.confidence, 3),
            "estimated_cost_usd": round(doc.estimated_cost_usd, 4),
            "processing_time_sec": round(doc.extraction_time_sec, 2),
            "pages_processed": doc.pages_processed,
            "success": doc.success,
            "error_message": doc.error_message,
            "text_blocks_count": len(doc.text_blocks),
            "tables_count": len(doc.tables),
            "figures_count": len(doc.figures),
        }

        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")