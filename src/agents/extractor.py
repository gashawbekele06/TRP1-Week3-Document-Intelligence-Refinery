from __future__ import annotations
from pathlib import Path
import json
import time
from typing import Any
from ..models.schemas import ExtractionLedgerEntry
from ..strategies.fasttext import FastTextExtractor
from ..strategies.layout import LayoutExtractor
from ..strategies.vision import VisionExtractor


REFINERY_DIR = Path(__file__).resolve().parents[2] / ".refinery"
REFINERY_DIR.mkdir(exist_ok=True)


class ExtractionRouter:
    def __init__(self, path: str, profile: dict | None = None):
        self.path = Path(path)
        self.profile = profile or {}
        self.ledger_file = REFINERY_DIR / "extraction_ledger.jsonl"

    def run(self) -> dict[str, Any]:
        # Decide initial strategy
        est = self.profile.get("estimated_extraction_cost") if self.profile else None
        if est == "needs_vision_model":
            strategy = "vision"
        elif est == "needs_layout_model":
            strategy = "layout"
        else:
            strategy = "fast_text"

        result = None
        if strategy == "fast_text":
            fx = FastTextExtractor(str(self.path))
            result = fx.extract()
            confidence = result.get("confidence", 0.0)
            # Escalation guard
            if confidence < 0.5:
                # try layout
                lx = LayoutExtractor(str(self.path))
                result = lx.extract()
        elif strategy == "layout":
            lx = LayoutExtractor(str(self.path))
            result = lx.extract()
            if result.get("confidence", 0.0) < 0.5:
                vx = VisionExtractor(str(self.path))
                result = vx.extract()
        else:
            vx = VisionExtractor(str(self.path))
            result = vx.extract()

        # write ledger entry
        ledger = result.get("ledger")
        if ledger:
            with open(self.ledger_file, "a", encoding="utf-8") as f:
                f.write(ledger.model_dump_json() + "\n")

        return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: extractor.py path/to/doc.pdf")
        raise SystemExit(1)
    r = ExtractionRouter(sys.argv[1])
    out = r.run()
    print(out)
