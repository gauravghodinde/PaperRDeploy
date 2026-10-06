"""
see_stage.py
============
"See" Stage of the Read–See–Synthesize pipeline.

This module bridges the classification module with the relative_work_generation
pipeline. Given the text extracted from a target paper's graphical abstract
(or inferred from the paper content), it:

  1. Classifies each information unit as SUPPLEMENTARY or COMPLEMENTARY
  2. Aligns classified evidence with the textual context
  3. Returns structured, usable evidence for the Synthesize stage

The module is designed to be called from pipeline.py after TargetPaperExtractor
has produced supplementary_info / complementary_info strings.
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# ── Locate the classification module ─────────────────────────────────────────
_THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR        = _THIS_DIR                            # …/relative_work_generation/src
_PROJECT_ROOT   = os.path.dirname(_THIS_DIR)          # …/relative_work_generation
_PARENT_ROOT    = os.path.dirname(_PROJECT_ROOT)      # …/final  (local dev only)

# Search order:
#  1. Bundled copy inside the project  (Streamlit Cloud / self-contained deploy)
#  2. Sibling classification/ directory (local monorepo dev)
_BUNDLED_CLASS_DIR = os.path.join(_PROJECT_ROOT, "classification")
_SIBLING_CLASS_DIR = os.path.join(_PARENT_ROOT,  "classification")

_CLASS_DIR = _BUNDLED_CLASS_DIR  # default; used for model/data paths below

for _candidate in [_BUNDLED_CLASS_DIR, _SIBLING_CLASS_DIR]:
    if os.path.isfile(os.path.join(_candidate, "details_classifier.py")):
        _CLASS_DIR = _candidate
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break
else:
    # Neither bundled nor sibling — leave sys.path as-is
    pass

try:
    from details_classifier import DetailsClassifier, DetailItem
    _CLASSIFIER_AVAILABLE = True
except ImportError:
    _CLASSIFIER_AVAILABLE = False
    logger.warning(
        "details_classifier.py not found. "
        "SeeStage will run in pass-through mode (no ML classification)."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceUnit:
    """A single classified information unit extracted from a graphical abstract."""
    text:       str                   # Raw text of the unit
    label:      str                   # "SUPPLEMENTARY" | "COMPLEMENTARY" | "UNKNOWN"
    confidence: float = 0.0           # Classifier confidence (0–1)
    source:     str   = "graphical_abstract"
    domain:     str   = ""
    metadata:   Dict  = field(default_factory=dict)


@dataclass
class SeeStageOutput:
    """Structured output of the See stage."""
    supplementary: List[EvidenceUnit] = field(default_factory=list)
    complementary: List[EvidenceUnit] = field(default_factory=list)
    unclassified:  List[EvidenceUnit] = field(default_factory=list)

    # Formatted plain-text summaries for downstream prompts
    supplementary_text: str = ""
    complementary_text: str = ""

    # Classification stats
    total_units:         int   = 0
    classified_units:    int   = 0
    classification_rate: float = 0.0
    model_used:          str   = ""

    def as_context_strings(self) -> Dict[str, str]:
        """Return {supplementary, complementary} strings for LLM prompts."""
        return {
            "supplementary": self.supplementary_text,
            "complementary": self.complementary_text,
        }


# ═══════════════════════════════════════════════════════════════════════════
# See Stage
# ═══════════════════════════════════════════════════════════════════════════

class SeeStage:
    """
    Stage 2: See.

    Classifies information units extracted from graphical abstracts into
    SUPPLEMENTARY (supportive/validating) and COMPLEMENTARY (additive/novel)
    categories, and structures them for downstream synthesis.

    Usage
    -----
        see = SeeStage()
        see.load_classifier(
            model_path     = "classification/saved_models/best_classifier_model.pkl",
            extractor_path = "classification/saved_models/feature_extractor.pkl"
        )
        output = see.process(
            abstract           = "...",
            supplementary_raw  = "bullet1\nbullet2\n...",
            complementary_raw  = "bullet1\nbullet2\n...",
        )
        print(output.complementary_text)
    """

    # Default saved-model paths (relative to project root)
    _DEFAULT_MODEL_PATH     = os.path.join(_CLASS_DIR, "saved_models", "best_classifier_model.pkl")
    _DEFAULT_EXTRACTOR_PATH = os.path.join(_CLASS_DIR, "saved_models", "feature_extractor.pkl")

    # Fall-back data files used when no saved model exists
    _DATA_DIRS = [
        os.path.join(_CLASS_DIR, "Augumented"),
        os.path.join(_CLASS_DIR, "without_augumentation"),
        _CLASS_DIR,
    ]
    _DATA_FILES = ["Combined.json", "Biomedical.json", "Wireless.json", "nlp_output.json"]

    def __init__(self, auto_load: bool = True):
        """
        Args:
            auto_load: Try to load a saved model automatically on init.
        """
        self.classifier: Optional[DetailsClassifier] = None
        self._ready = False

        if auto_load and _CLASSIFIER_AVAILABLE:
            self._try_auto_load()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _try_auto_load(self):
        """Silently try to load saved model; fall back to training."""
        if (os.path.exists(self._DEFAULT_MODEL_PATH)
                and os.path.exists(self._DEFAULT_EXTRACTOR_PATH)):
            try:
                self.classifier = DetailsClassifier.load(
                    self._DEFAULT_MODEL_PATH,
                    self._DEFAULT_EXTRACTOR_PATH
                )
                self._ready = True
                logger.info("SeeStage: loaded saved classifier.")
                return
            except Exception as e:
                logger.warning(f"SeeStage: auto-load failed ({e}), will train.")

        # Try to train from available JSON data
        self._try_auto_train()

    def _try_auto_train(self):
        """Train on any available JSON data files."""
        json_files = []
        for d in self._DATA_DIRS:
            for fname in self._DATA_FILES:
                p = os.path.join(d, fname)
                if os.path.exists(p) and p not in json_files:
                    json_files.append(p)
                    if len(json_files) >= 3:
                        break
            if len(json_files) >= 3:
                break

        if not json_files:
            logger.warning("SeeStage: no data files found. Running in pass-through mode.")
            return

        logger.info(f"SeeStage: training on {[os.path.basename(f) for f in json_files]} …")
        try:
            self.classifier = DetailsClassifier()
            n = self.classifier.load_data(json_files)
            if n < 10:
                logger.warning("SeeStage: too few samples to train reliably.")
                return
            self.classifier.train(verbose=False)

            # Persist for next run
            os.makedirs(os.path.dirname(self._DEFAULT_MODEL_PATH), exist_ok=True)
            self.classifier.save(self._DEFAULT_MODEL_PATH, self._DEFAULT_EXTRACTOR_PATH)

            self._ready = True
            logger.info("SeeStage: classifier trained and saved.")
        except Exception as e:
            logger.warning(f"SeeStage: training failed ({e}). Pass-through mode.")

    def load_classifier(self, model_path: str, extractor_path: str):
        """Explicitly load a saved classifier from given paths."""
        self.classifier = DetailsClassifier.load(model_path, extractor_path)
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready and self.classifier is not None

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(
        self,
        abstract:           str,
        supplementary_raw:  str = "",
        complementary_raw:  str = "",
        domain:             str = "",
        verify_labels:      bool = True,
    ) -> SeeStageOutput:
        """
        Classify and structure all information units.

        Args:
            abstract          : Abstract of the target paper
            supplementary_raw : Raw supplementary text (newline-separated bullets/sentences)
            complementary_raw : Raw complementary text
            domain            : Domain hint ("biomedical", "nlp", "wireless", "")
            verify_labels     : If True, re-classify LLM-labelled units using ML classifier

        Returns:
            SeeStageOutput with structured evidence
        """
        output = SeeStageOutput()

        # Parse input text into discrete units
        supp_units = self._parse_units(supplementary_raw, presumed_label="SUPPLEMENTARY", domain=domain)
        comp_units = self._parse_units(complementary_raw, presumed_label="COMPLEMENTARY", domain=domain)
        all_units  = supp_units + comp_units
        output.total_units = len(all_units)

        if not all_units:
            logger.info("SeeStage: no information units to classify.")
            output.supplementary_text = supplementary_raw
            output.complementary_text = complementary_raw
            output.model_used = "pass-through (empty input)"
            return output

        # Classify with ML model (if available) or use LLM labels as-is
        if self.is_ready and verify_labels:
            classified = self._classify_units(abstract, all_units)
            output.model_used = self.classifier.model_name
        else:
            # Pass-through: trust the LLM labels
            classified = all_units
            output.model_used = "pass-through (no ML model)"

        # Bucket into supp / comp / unknown
        for unit in classified:
            if unit.label == "SUPPLEMENTARY":
                output.supplementary.append(unit)
            elif unit.label == "COMPLEMENTARY":
                output.complementary.append(unit)
            else:
                output.unclassified.append(unit)

        output.classified_units    = len(output.supplementary) + len(output.complementary)
        output.classification_rate = output.classified_units / max(output.total_units, 1)

        # Build formatted text blocks
        output.supplementary_text = self._format_units(output.supplementary) or supplementary_raw
        output.complementary_text = self._format_units(output.complementary) or complementary_raw

        logger.info(
            f"SeeStage: {output.classified_units}/{output.total_units} units classified "
            f"({len(output.supplementary)} supp, {len(output.complementary)} comp) "
            f"using {output.model_used}"
        )
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_units(self, raw_text: str, presumed_label: str, domain: str) -> List[EvidenceUnit]:
        """Split raw text into individual information units."""
        if not raw_text or not raw_text.strip():
            return []
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        # Remove bullet-point prefixes
        clean = [l.lstrip("-•*–·▪▸►→").strip() for l in lines]
        # Remove heading-like lines (short lines with no period, often headings)
        units = [l for l in clean if len(l) > 15 and not l.endswith(":")]
        return [
            EvidenceUnit(text=u, label=presumed_label, confidence=0.0, domain=domain)
            for u in units
        ]

    def _classify_units(self, abstract: str, units: List[EvidenceUnit]) -> List[EvidenceUnit]:
        """Re-classify each unit with the ML model."""
        updated = []
        for unit in units:
            try:
                result = self.classifier.classify_single_item(
                    abstract=abstract,
                    information=unit.text,
                    justification="",
                    domain=unit.domain,
                )
                unit.label      = result["label"]
                unit.confidence = result["confidence"]
                unit.metadata["probabilities"] = result.get("probabilities", {})
            except Exception as e:
                logger.debug(f"Classification failed for unit: {e}")
                # Keep original (LLM-provided) label
            updated.append(unit)
        return updated

    @staticmethod
    def _format_units(units: List[EvidenceUnit]) -> str:
        """Format evidence units into a readable bullet list."""
        if not units:
            return ""
        lines = []
        for u in units:
            conf_str = f" [conf: {u.confidence:.2f}]" if u.confidence > 0 else ""
            lines.append(f"• {u.text}{conf_str}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Batch classification (for dataset evaluation)
    # ------------------------------------------------------------------

    def classify_dataset(
        self,
        json_paths: List[str],
        augment:    bool = False
    ) -> Dict[str, Any]:
        """
        Train and evaluate classifier on domain JSON files.
        Returns per-model metrics as a dict.
        """
        if not _CLASSIFIER_AVAILABLE:
            return {"error": "details_classifier module not available"}

        clf = DetailsClassifier()
        n = clf.load_data(json_paths)
        if n < 10:
            return {"error": f"Too few samples ({n}) to evaluate"}

        results = clf.train(augment=augment, verbose=True)
        self.classifier = clf
        self._ready = True
        return {
            "n_samples":  n,
            "best_model": clf.model_name,
            "metrics":    results,
            "summary":    clf.summary().to_dict(orient="records"),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Stand-alone demo
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    see = SeeStage(auto_load=True)

    sample_abstract = (
        "We propose a multi-source framework for automated related work generation "
        "that augments textual evidence with information from graphical abstracts. "
        "The framework includes keyword extraction, preference-aware retrieval, "
        "and multi-LLM synthesis."
    )
    sample_supp = (
        "- Uses KeyBERT with MiniLM sentence embeddings for keyword extraction\n"
        "- Retrieves top-50 papers from Semantic Scholar API\n"
        "- Scoring: citation impact (log-normalized) + recency + abstract quality\n"
        "- Hyperparameters: K=15 papers, max 512 tokens, temperature=0.7"
    )
    sample_comp = (
        "- First framework to integrate graphical abstract evidence into related work generation\n"
        "- Introduces supplementary/complementary classification as a novel NLP task\n"
        "- Multi-LLM ensemble (LLaMA, GPT, Gemini, DeepSeek) reduces single-model bias\n"
        "- Agent-assisted human-in-the-loop workflow for traceability"
    )

    output = see.process(
        abstract=sample_abstract,
        supplementary_raw=sample_supp,
        complementary_raw=sample_comp,
        domain="nlp"
    )

    print(f"\nTotal units  : {output.total_units}")
    print(f"Classified   : {output.classified_units} ({output.classification_rate:.0%})")
    print(f"Model used   : {output.model_used}")
    print(f"\nSUPPLEMENTARY ({len(output.supplementary)} units):")
    print(output.supplementary_text or "  (none)")
    print(f"\nCOMPLEMENTARY ({len(output.complementary)} units):")
    print(output.complementary_text or "  (none)")
