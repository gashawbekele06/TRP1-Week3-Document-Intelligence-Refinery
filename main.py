"""Project entrypoint.

This repository is organized as a multi-stage PDF processing pipeline.

For corpus validation (Classes A–D), run:

    python main.py --run-corpus --input-dir data

This will execute triage→extraction→chunking→pageindex→fact extraction across
all PDFs in the folder and write a summary report to:

    .refinery/reports/corpus_report.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.corpus_runner import run_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Document Intelligence Refinery")
    parser.add_argument("--run-corpus", action="store_true", help="Process all PDFs in a folder")
    parser.add_argument("--input-dir", type=str, default="data", help="Folder containing PDFs")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of PDFs")
    parser.add_argument(
        "--use-llm-pageindex",
        action="store_true",
        help="Use Gemini for PageIndex summaries when API key is set",
    )
    args = parser.parse_args()

    if args.run_corpus:
        report = run_corpus(
            Path(args.input_dir),
            limit=args.limit,
            use_llm_for_pageindex=args.use_llm_pageindex,
        )
        print(
            f"Corpus run complete: {report['success_count']}/{report['total']} succeeded. "
            f"Report: .refinery/reports/corpus_report.json"
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
