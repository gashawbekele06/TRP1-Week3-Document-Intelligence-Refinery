"""
ExtractionStrategy Abstract Base Class
=====================================

Purpose:
Defines the common interface that all three strategies (A, B, C) must follow.
Ensures polymorphism: the router can treat any strategy the same way.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument


class ExtractionStrategy(ABC):
    """Base class for all extraction strategies"""

    name: str = "abstract"  # must be overridden by subclasses

    @abstractmethod
    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        """
        Main method each strategy must implement.

        Args:
            pdf_path: Path to the input PDF file
            profile: Triage result (tells us what kind of document this is)

        Returns:
            ExtractedDocument: normalized extraction result
        """
        pass

    def compute_confidence(self, doc: ExtractedDocument) -> float:
        """
        Calculate how reliable this extraction result is.
        Used by the router to decide whether to escalate.

        Logic:
        - High character count → good text coverage
        - Tables/figures present → good structure capture
        - Low values → likely scanned or broken layout

        Can be overridden in subclasses for strategy-specific scoring.
        """
        if not doc.text_blocks and not doc.tables and not doc.figures:
            return 0.0

        char_count = sum(len(b.text) for b in doc.text_blocks)
        structure_count = len(doc.tables) + len(doc.figures)
        pages = doc.pages_processed or 1

        density_score = min(1.0, char_count / (pages * 1500 + 1))
        structure_score = min(1.0, structure_count / (pages * 0.8 + 1))

        return min(0.98, 0.42 + 0.34 * density_score + 0.24 * structure_score)