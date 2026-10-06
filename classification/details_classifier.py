"""
details_classifier.py
=====================
Standalone Python module for the Supplementary/Complementary Detail Classifier.

This module is extracted from `details_classifier_clean2.ipynb` and provides:
  - FeatureExtractor  : 51-feature hybrid (semantic, lexical, discourse, entity, role)
  - DetailsClassifier : multi-model training, evaluation, persistence, and prediction
  - classify_detail() : one-call convenience function

Usage
-----
    from details_classifier import DetailsClassifier, FeatureExtractor

    # Train on domain JSON data
    clf = DetailsClassifier()
    clf.load_data(["Biomedical.json", "Wireless.json", "nlp_output.json"])
    clf.train()

    # Classify a new information unit
    result = clf.classify_single_item(
        abstract="Rheumatoid arthritis is treated with hydrogels...",
        information="Molecular structure showing polymer cross-linking and drug encapsulation",
        justification="Adds molecular detail not mentioned in abstract"
    )
    print(result)  # {'label': 'COMPLEMENTARY', 'confidence': 0.92, ...}
"""

from __future__ import annotations

import os
import re
import json
import pickle
import string
import warnings
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from collections import Counter

# NLTK
import nltk
nltk.download("punkt",       quiet=True)
nltk.download("punkt_tab",   quiet=True)
nltk.download("stopwords",   quiet=True)
nltk.download("wordnet",     quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Optional heavy libraries (graceful degradation) ──────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Embedding features disabled.")

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except (ImportError, OSError):
    _SPACY_AVAILABLE = False
    logger.warning("spaCy / en_core_web_sm not available. NER features disabled.")

try:
    from scipy.spatial.distance import cosine as scipy_cosine
    from scipy.stats import entropy as scipy_entropy
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Data Model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DetailItem:
    """A single annotated information unit from a research paper."""
    abstract:      str
    information:   str
    label:         int        # 0 = SUPPLEMENTARY, 1 = COMPLEMENTARY
    justification: str = ""
    domain:        str = ""
    paper_id:      str = ""

    LABEL_MAP = {0: "SUPPLEMENTARY", 1: "COMPLEMENTARY"}

    @property
    def label_name(self) -> str:
        return self.LABEL_MAP[self.label]


# ═══════════════════════════════════════════════════════════════════════════
# Text Preprocessor
# ═══════════════════════════════════════════════════════════════════════════

class TextProcessor:
    """Basic NLP preprocessing utilities."""

    def __init__(self):
        self.stop_words  = set(stopwords.words("english"))
        self.lemmatizer  = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        """Lowercase, remove punctuation/digits."""
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(self.clean(text))

    def tokens_no_stop(self, text: str) -> List[str]:
        return [t for t in self.tokenize(text) if t not in self.stop_words]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def keywords(self, text: str) -> List[str]:
        return self.lemmatize_tokens(self.tokens_no_stop(text))


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extractor  (51 features, as described in the paper)
# ═══════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Extract the 51-dimensional hybrid feature vector described in the paper.

    Feature groups:
      Semantic  (10) : TF-IDF cosine, embedding cosine/L2, novelty, JS divergence
      Lexical   (14) : keyword overlap, new-word count/ratio, rare words, new nouns…
      Discourse  (5) : contrast, addition, novelty, elaboration, causal markers
      Entity     (8) : NER entities, new datasets/methods/metrics/variables
      Roles      (6) : sentence-role indicators (background, method, result…)
      Structure  (8) : length, type-token ratio, clause count, quotes, tech terms
    """

    # Discourse marker sets
    _CONTRAST_MARKERS    = {"however", "although", "but", "yet", "whereas",
                            "nevertheless", "on the other hand", "despite",
                            "in contrast", "conversely"}
    _ADDITION_MARKERS    = {"furthermore", "moreover", "additionally", "also",
                            "in addition", "besides", "as well", "similarly"}
    _NOVELTY_MARKERS     = {"novel", "new", "propose", "introduce", "present",
                            "first", "unique", "innovative", "unprecedented",
                            "original", "pioneer"}
    _ELABORATION_MARKERS = {"specifically", "in particular", "that is",
                            "for example", "namely", "such as", "i.e.", "e.g."}
    _CAUSAL_MARKERS      = {"therefore", "thus", "hence", "consequently",
                            "as a result", "because", "since", "due to"}

    _TECH_TERMS          = {"algorithm", "model", "neural", "dataset", "baseline",
                            "experiment", "benchmark", "evaluation", "performance",
                            "accuracy", "precision", "recall", "f1", "loss",
                            "gradient", "layer", "training", "inference"}

    def __init__(self):
        self.processor = TextProcessor()
        self.tfidf     = TfidfVectorizer(max_features=5000, stop_words="english")
        self._tfidf_fitted = False

        self.sbert_model = None
        if _SBERT_AVAILABLE:
            try:
                self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("SBERT loaded (all-MiniLM-L6-v2)")
            except Exception as e:
                logger.warning(f"SBERT load failed: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_tfidf(self, corpus: List[str]):
        """Fit TF-IDF on a corpus of texts."""
        self.tfidf.fit(corpus)
        self._tfidf_fitted = True

    def extract(self, item: DetailItem) -> np.ndarray:
        """Return a 1-D float array of 51 features."""
        feats = {}
        feats.update(self._semantic_features(item))
        feats.update(self._lexical_features(item))
        feats.update(self._discourse_features(item))
        feats.update(self._entity_features(item))
        feats.update(self._role_features(item))
        feats.update(self._structure_features(item))
        return np.array(list(feats.values()), dtype=np.float32)

    @property
    def feature_names(self) -> List[str]:
        dummy = DetailItem(abstract="a", information="b", label=0)
        feats = {}
        feats.update(self._semantic_features(dummy))
        feats.update(self._lexical_features(dummy))
        feats.update(self._discourse_features(dummy))
        feats.update(self._entity_features(dummy))
        feats.update(self._role_features(dummy))
        feats.update(self._structure_features(dummy))
        return list(feats.keys())

    # ------------------------------------------------------------------
    # Feature groups (each returns an ordered dict)
    # ------------------------------------------------------------------

    def _semantic_features(self, item: DetailItem) -> dict:
        abs_kw  = " ".join(self.processor.keywords(item.abstract))
        info_kw = " ".join(self.processor.keywords(item.information))
        just_kw = " ".join(self.processor.keywords(item.justification)) if item.justification else ""

        # TF-IDF cosine
        try:
            if self._tfidf_fitted:
                vec_a = self.tfidf.transform([abs_kw]).toarray()[0]
                vec_i = self.tfidf.transform([info_kw]).toarray()[0]
                tfidf_cos = float(self._cosine(vec_a, vec_i))
            else:
                tfidf_cos = 0.0
        except Exception:
            tfidf_cos = 0.0

        # SBERT embeddings
        emb_cos = emb_l2 = novelty = 0.0
        if self.sbert_model:
            try:
                ea = self.sbert_model.encode(item.abstract)
                ei = self.sbert_model.encode(item.information)
                emb_cos  = float(1 - scipy_cosine(ea, ei)) if _SCIPY_AVAILABLE else float(np.dot(ea, ei) / (np.linalg.norm(ea) * np.linalg.norm(ei) + 1e-9))
                emb_l2   = float(np.linalg.norm(ea - ei))
                novelty  = float(1 - emb_cos)
            except Exception:
                pass

        # Jensen–Shannon divergence (token distributions)
        js_div = self._js_divergence(abs_kw, info_kw)

        # Abstract–info similarity (justification based)
        abs_info_sim = tfidf_cos

        # Abstract–justification similarity
        try:
            if self._tfidf_fitted and just_kw:
                vec_j  = self.tfidf.transform([just_kw]).toarray()[0]
                abs_just_sim = float(self._cosine(vec_a, vec_j))
            else:
                abs_just_sim = 0.0
        except Exception:
            abs_just_sim = 0.0

        return {
            "tfidf_cosine":          tfidf_cos,
            "emb_cosine":            emb_cos,
            "emb_l2":                emb_l2,
            "semantic_novelty":      novelty,
            "js_divergence":         js_div,
            "abstract_info_sim":     abs_info_sim,
            "abstract_just_sim":     abs_just_sim,
            "info_length_norm":      min(len(item.information) / 500, 1.0),
            "abs_length_norm":       min(len(item.abstract) / 1000, 1.0),
            "just_length_norm":      min(len(item.justification) / 300, 1.0) if item.justification else 0.0,
        }

    def _lexical_features(self, item: DetailItem) -> dict:
        abs_tokens  = set(self.processor.keywords(item.abstract))
        info_tokens = set(self.processor.keywords(item.information))
        just_tokens = set(self.processor.keywords(item.justification)) if item.justification else set()

        overlap         = abs_tokens & info_tokens
        new_words       = info_tokens - abs_tokens
        rare_words      = {w for w in new_words if len(w) > 8}

        kw_overlap_cnt  = len(overlap)
        kw_overlap_ratio= len(overlap) / max(len(abs_tokens), 1)
        new_word_cnt    = len(new_words)
        new_word_ratio  = len(new_words) / max(len(info_tokens), 1)
        rare_word_cnt   = len(rare_words)
        rare_word_ratio = len(rare_words) / max(len(info_tokens), 1)

        # New nouns (POS tag)
        try:
            pos_tags = nltk.pos_tag(word_tokenize(item.information.lower()))
            info_nouns = {w for w, p in pos_tags if p.startswith("NN")}
            abs_nouns  = {w for w, p in nltk.pos_tag(word_tokenize(item.abstract.lower())) if p.startswith("NN")}
            new_nouns  = len(info_nouns - abs_nouns)
        except Exception:
            new_nouns = 0

        # Justification-specific flags
        just_adds_insight = int(any(w in just_tokens for w in
                                    {"add", "extend", "complement", "novel", "new", "beyond", "additional"}))
        just_supports     = int(any(w in just_tokens for w in
                                    {"support", "reinforce", "confirm", "validate", "elaborate", "detail"}))

        # Topic shift score (rough proxy: 1 - overlap ratio)
        topic_shift = 1.0 - kw_overlap_ratio

        return {
            "kw_overlap_cnt":    float(kw_overlap_cnt),
            "kw_overlap_ratio":  kw_overlap_ratio,
            "new_word_cnt":      float(new_word_cnt),
            "new_word_ratio":    new_word_ratio,
            "rare_word_cnt":     float(rare_word_cnt),
            "rare_word_ratio":   rare_word_ratio,
            "new_nouns":         float(new_nouns),
            "just_adds_insight": float(just_adds_insight),
            "just_supports":     float(just_supports),
            "topic_shift":       topic_shift,
            "info_vocab_size":   float(len(info_tokens)),
            "abs_vocab_size":    float(len(abs_tokens)),
            "vocab_expansion":   float(len(info_tokens - abs_tokens)),
            "shared_ratio":      kw_overlap_ratio,
        }

    def _discourse_features(self, item: DetailItem) -> dict:
        text = (item.information + " " + item.justification).lower()

        contrast    = int(any(m in text for m in self._CONTRAST_MARKERS))
        addition    = int(any(m in text for m in self._ADDITION_MARKERS))
        novelty     = int(any(m in text for m in self._NOVELTY_MARKERS))
        elaboration = int(any(m in text for m in self._ELABORATION_MARKERS))
        causal      = int(any(m in text for m in self._CAUSAL_MARKERS))

        return {
            "has_contrast_marker":    float(contrast),
            "has_addition_marker":    float(addition),
            "has_novelty_marker":     float(novelty),
            "has_elaboration_marker": float(elaboration),
            "has_causal_marker":      float(causal),
        }

    def _entity_features(self, item: DetailItem) -> dict:
        if not _SPACY_AVAILABLE:
            return {k: 0.0 for k in [
                "ner_entity_cnt", "new_entity_cnt",
                "new_dataset_cnt", "new_method_cnt",
                "new_metric_cnt", "new_variable_cnt",
                "entity_overlap", "entity_expansion_ratio"
            ]}
        try:
            doc_abs  = _NLP(item.abstract[:500])
            doc_info = _NLP(item.information[:500])

            abs_ents  = {e.text.lower() for e in doc_abs.ents}
            info_ents = {e.text.lower() for e in doc_info.ents}
            new_ents  = info_ents - abs_ents

            # Heuristic sub-categories
            dataset_words = {"dataset", "corpus", "benchmark", "collection", "database"}
            method_words  = {"model", "method", "algorithm", "approach", "framework", "network"}
            metric_words  = {"accuracy", "precision", "recall", "f1", "bleu", "rouge", "score", "metric"}

            tokens_info = set(item.information.lower().split())
            new_dataset = int(bool(tokens_info & dataset_words - set(item.abstract.lower().split())))
            new_method  = int(bool(tokens_info & method_words  - set(item.abstract.lower().split())))
            new_metric  = int(bool(tokens_info & metric_words  - set(item.abstract.lower().split())))
            new_var     = len(new_ents)

            return {
                "ner_entity_cnt":        float(len(info_ents)),
                "new_entity_cnt":        float(len(new_ents)),
                "new_dataset_cnt":       float(new_dataset),
                "new_method_cnt":        float(new_method),
                "new_metric_cnt":        float(new_metric),
                "new_variable_cnt":      float(new_var),
                "entity_overlap":        float(len(abs_ents & info_ents)),
                "entity_expansion_ratio": len(new_ents) / max(len(info_ents), 1),
            }
        except Exception:
            return {k: 0.0 for k in [
                "ner_entity_cnt", "new_entity_cnt",
                "new_dataset_cnt", "new_method_cnt",
                "new_metric_cnt", "new_variable_cnt",
                "entity_overlap", "entity_expansion_ratio"
            ]}

    def _role_features(self, item: DetailItem) -> dict:
        text = item.information.lower()
        return {
            "role_background": float(any(w in text for w in
                ["background", "prior", "previous", "existing", "traditional"])),
            "role_method":     float(any(w in text for w in
                ["proposed", "develop", "design", "implement", "architecture", "train"])),
            "role_result":     float(any(w in text for w in
                ["result", "achieve", "outperform", "improve", "performance", "accuracy"])),
            "role_discussion": float(any(w in text for w in
                ["show", "indicate", "suggest", "demonstrate", "observe", "find"])),
            "role_elaboration":float(any(w in text for w in
                ["detail", "specifically", "particular", "namely", "example"])),
            "role_contrast":   float(any(w in text for w in
                ["however", "unlike", "contrast", "differ", "instead"])),
        }

    def _structure_features(self, item: DetailItem) -> dict:
        words = item.information.split()
        sents = sent_tokenize(item.information) if item.information else [""]

        avg_word_len  = np.mean([len(w) for w in words]) if words else 0.0
        vocab         = set(w.lower() for w in words)
        ttr           = len(vocab) / max(len(words), 1)
        clause_cnt    = item.information.count(",") + item.information.count(";")
        has_quotes    = int('"' in item.information or "'" in item.information)
        tech_term_cnt = sum(1 for w in words if w.lower() in self._TECH_TERMS)
        sent_cnt      = len(sents)
        avg_sent_len  = np.mean([len(s.split()) for s in sents]) if sents else 0.0

        return {
            "word_count":      float(len(words)),
            "avg_word_len":    float(avg_word_len),
            "type_token_ratio":ttr,
            "clause_count":    float(clause_cnt),
            "has_quotes":      float(has_quotes),
            "tech_term_count": float(tech_term_cnt),
            "sentence_count":  float(sent_cnt),
            "avg_sentence_len":float(avg_sent_len),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _js_divergence(text_a: str, text_b: str) -> float:
        if not _SCIPY_AVAILABLE:
            return 0.0
        try:
            tokens_a = word_tokenize(text_a)
            tokens_b = word_tokenize(text_b)
            vocab = list(set(tokens_a + tokens_b))
            ca = Counter(tokens_a)
            cb = Counter(tokens_b)
            pa = np.array([ca.get(v, 0) for v in vocab], dtype=float) + 1e-9
            pb = np.array([cb.get(v, 0) for v in vocab], dtype=float) + 1e-9
            pa /= pa.sum(); pb /= pb.sum()
            m = 0.5 * (pa + pb)
            return float(0.5 * scipy_entropy(pa, m) + 0.5 * scipy_entropy(pb, m))
        except Exception:
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════════════

class DetailsClassifier:
    """
    Full train/evaluate/predict pipeline for Supp/Comp classification.

    Steps:
      1. load_data()       – parse domain JSON files into DetailItem objects
      2. prepare_features()– extract 51-d feature matrix
      3. train()           – fit & evaluate 6 model types
      4. predict() / classify_single_item() – inference
      5. save() / load()   – model persistence
    """

    MODELS = {
        "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)":           lambda: SVC(kernel="rbf", probability=True, random_state=42),
        "MLP":                 lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "Ridge":               lambda: RidgeClassifier(),
    }

    def __init__(self, model_name: str = "Gradient Boosting"):
        self.model_name      = model_name
        self.feature_extractor = FeatureExtractor()
        self.scaler            = StandardScaler()
        self.classifier        = None
        self.is_trained        = False
        self.items: List[DetailItem] = []
        self.results: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, json_paths: List[str]) -> int:
        """Load and parse one or more domain JSON files."""
        self.items = []
        for path in json_paths:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.items.extend(self._parse_json(data, domain=os.path.basename(path)))
        logger.info(f"Loaded {len(self.items)} detail items from {len(json_paths)} file(s).")
        return len(self.items)

    def _parse_json(self, data, domain: str) -> List[DetailItem]:
        items = []
        papers = data if isinstance(data, list) else data.get("papers", [])
        for paper in papers:
            abstract = paper.get("abstract", "")
            pid      = paper.get("title", paper.get("doi", ""))[:40]

            for detail in paper.get("supplementary_details", []):
                info = detail if isinstance(detail, str) else detail.get("detail", str(detail))
                items.append(DetailItem(
                    abstract=abstract, information=info, label=0,
                    justification=detail.get("justification", "") if isinstance(detail, dict) else "",
                    domain=domain, paper_id=pid
                ))

            for detail in paper.get("complementary_details", []):
                info = detail if isinstance(detail, str) else detail.get("detail", str(detail))
                items.append(DetailItem(
                    abstract=abstract, information=info, label=1,
                    justification=detail.get("justification", "") if isinstance(detail, dict) else "",
                    domain=domain, paper_id=pid
                ))
        return items

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build feature matrix X and label vector y."""
        if not self.items:
            raise ValueError("No data loaded. Call load_data() first.")

        corpus = [" ".join(self.feature_extractor.processor.keywords(
            it.abstract + " " + it.information)) for it in self.items]
        self.feature_extractor.fit_tfidf(corpus)

        X = np.vstack([self.feature_extractor.extract(it) for it in self.items])
        y = np.array([it.label for it in self.items])

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        test_size:  float = 0.2,
        val_size:   float = 0.2,
        augment:    bool  = False,
        verbose:    bool  = True
    ) -> Dict[str, dict]:
        """
        Train all model types and return evaluation metrics.

        Args:
            test_size : fraction of data held out for final test
            val_size  : fraction of train used for validation
            augment   : whether to oversample minority class (basic duplication)
            verbose   : print progress

        Returns:
            dict of {model_name: {accuracy, precision, recall, f1, auc}}
        """
        X, y = self.prepare_features()

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        if augment:
            X_trainval, y_trainval = self._augment(X_trainval, y_trainval)

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size, stratify=y_trainval, random_state=42
        )

        # Scale
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)
        X_test_s  = self.scaler.transform(X_test)

        self.results = {}
        best_f1  = -1.0
        best_clf = None

        for name, factory in self.MODELS.items():
            if verbose:
                logger.info(f"Training {name}...")
            try:
                clf = factory()
                clf.fit(X_train_s, y_train)
                preds = clf.predict(X_test_s)

                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(X_test_s)[:, 1]
                    auc = roc_auc_score(y_test, proba)
                elif hasattr(clf, "decision_function"):
                    scores = clf.decision_function(X_test_s)
                    auc = roc_auc_score(y_test, scores)
                else:
                    auc = 0.0

                metrics = {
                    "accuracy":  accuracy_score(y_test, preds),
                    "precision": precision_score(y_test, preds, zero_division=0),
                    "recall":    recall_score(y_test, preds, zero_division=0),
                    "f1":        f1_score(y_test, preds, zero_division=0),
                    "auc":       auc,
                }
                self.results[name] = metrics

                if metrics["f1"] > best_f1:
                    best_f1  = metrics["f1"]
                    best_clf = clf
                    self.model_name = name

                if verbose:
                    logger.info(
                        f"  {name}: acc={metrics['accuracy']:.4f} "
                        f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}"
                    )
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")

        self.classifier = best_clf
        self.is_trained = True
        logger.info(f"\nBest model: {self.model_name} (F1={best_f1:.4f})")
        return self.results

    @staticmethod
    def _augment(X: np.ndarray, y: np.ndarray):
        """Simple majority-class duplication to balance minority class."""
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        majority_cnt   = counts.max()

        X_parts, y_parts = [X], [y]
        for c, cnt in zip(classes, counts):
            if c != majority_class:
                deficit = majority_cnt - cnt
                idx = np.where(y == c)[0]
                extra_idx = np.random.choice(idx, size=deficit, replace=True)
                X_parts.append(X[extra_idx])
                y_parts.append(y[extra_idx])

        return np.vstack(X_parts), np.concatenate(y_parts)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, items: List[DetailItem]) -> List[Dict[str, Any]]:
        """Predict labels for a list of DetailItem objects."""
        self._check_trained()
        results = []
        for item in items:
            corpus_text = " ".join(self.feature_extractor.processor.keywords(
                item.abstract + " " + item.information
            ))
            feats = self.feature_extractor.extract(item).reshape(1, -1)
            feats = np.nan_to_num(feats)
            feats_s = self.scaler.transform(feats)
            pred  = self.classifier.predict(feats_s)[0]
            proba = (self.classifier.predict_proba(feats_s)[0]
                     if hasattr(self.classifier, "predict_proba") else None)
            confidence = float(proba[pred]) if proba is not None else 1.0
            results.append({
                "label":      DetailItem.LABEL_MAP[int(pred)],
                "label_int":  int(pred),
                "confidence": round(confidence, 4),
                "probabilities": {
                    "SUPPLEMENTARY": round(float(proba[0]), 4) if proba is not None else None,
                    "COMPLEMENTARY": round(float(proba[1]), 4) if proba is not None else None,
                }
            })
        return results

    def classify_single_item(
        self,
        abstract:      str,
        information:   str,
        justification: str = "",
        domain:        str = ""
    ) -> Dict[str, Any]:
        """
        Classify a single (abstract, information) pair.

        Returns
        -------
        dict with keys: label, label_int, confidence, probabilities
        """
        item = DetailItem(
            abstract=abstract,
            information=information,
            justification=justification,
            label=0,       # placeholder
            domain=domain
        )
        return self.predict([item])[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_path: str = "saved_model.pkl", extractor_path: str = "feature_extractor.pkl"):
        """Save classifier and feature extractor to disk."""
        self._check_trained()
        with open(model_path, "wb") as f:
            pickle.dump({
                "classifier":  self.classifier,
                "scaler":      self.scaler,
                "model_name":  self.model_name,
                "results":     self.results,
            }, f)
        with open(extractor_path, "wb") as f:
            pickle.dump(self.feature_extractor, f)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Feature extractor saved to {extractor_path}")

    @classmethod
    def load(cls, model_path: str = "saved_model.pkl", extractor_path: str = "feature_extractor.pkl"):
        """Load a previously saved classifier."""
        instance = cls()
        with open(model_path, "rb") as f:
            state = pickle.load(f)
        instance.classifier  = state["classifier"]
        instance.scaler      = state["scaler"]
        instance.model_name  = state["model_name"]
        instance.results     = state.get("results", {})
        instance.is_trained  = True

        with open(extractor_path, "rb") as f:
            instance.feature_extractor = pickle.load(f)
        logger.info(f"Model loaded from {model_path} ({instance.model_name})")
        return instance

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame comparing all trained models."""
        if not self.results:
            raise ValueError("No results. Call train() first.")
        rows = []
        for name, m in self.results.items():
            rows.append({
                "Model":     name,
                "Accuracy":  round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall":    round(m["recall"],    4),
                "F1":        round(m["f1"],        4),
                "AUC":       round(m["auc"],       4),
            })
        df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_trained(self):
        if not self.is_trained or self.classifier is None:
            raise RuntimeError(
                "Classifier is not trained. "
                "Call train() or load() before predicting."
            )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience one-call function
# ═══════════════════════════════════════════════════════════════════════════

def classify_detail(
    abstract:      str,
    information:   str,
    justification: str = "",
    model_path:    str = None,
    extractor_path:str = None,
) -> Dict[str, Any]:
    """
    Classify a single information unit without instantiating the full class.

    If saved model paths are provided, loads from disk.
    Otherwise, raises RuntimeError asking you to train first.
    """
    if model_path and os.path.exists(model_path):
        clf = DetailsClassifier.load(model_path, extractor_path or "feature_extractor.pkl")
    else:
        raise RuntimeError(
            "No saved model found. Train a model first:\n"
            "  clf = DetailsClassifier()\n"
            "  clf.load_data(['Biomedical.json', ...])\n"
            "  clf.train()\n"
            "  clf.save()"
        )
    return clf.classify_single_item(abstract, information, justification)


# ═══════════════════════════════════════════════════════════════════════════
# CLI Demo
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  DetailsClassifier — Quick Demo")
    print("=" * 70)

    # Find JSON data files relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        script_dir,
        os.path.join(script_dir, "Augumented"),
        os.path.join(script_dir, "without_augumentation"),
    ]

    json_files = []
    for d in candidate_dirs:
        for fname in ["Biomedical.json", "Wireless.json", "nlp_output.json", "Combined.json"]:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                json_files.append(p)

    if not json_files:
        print("[WARN] No JSON data files found. Place Biomedical.json / Wireless.json / nlp_output.json next to this script.")
        sys.exit(0)

    print(f"\n[1/3] Loading data from: {[os.path.basename(f) for f in json_files[:3]]}")
    clf = DetailsClassifier()
    n = clf.load_data(json_files[:3])  # use first 3 files
    print(f"       Loaded {n} detail items.")

    print("\n[2/3] Training models...")
    results = clf.train(verbose=True)

    print("\n[3/3] Model comparison:")
    print(clf.summary().to_string(index=False))

    # Save
    save_dir = os.path.join(script_dir, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    clf.save(
        model_path=os.path.join(save_dir, "best_classifier_model.pkl"),
        extractor_path=os.path.join(save_dir, "feature_extractor.pkl"),
    )

    # Quick inference demo
    print("\n" + "=" * 70)
    print("  Inference Demo")
    print("=" * 70)
    result = clf.classify_single_item(
        abstract="Rheumatoid arthritis (RA) is an autoimmune disorder treated with hydrogels.",
        information="Molecular structure showing polymer cross-linking and drug encapsulation mechanism.",
        justification="Adds molecular detail not mentioned in the abstract."
    )
    print(f"  Label      : {result['label']}")
    print(f"  Confidence : {result['confidence']}")
    print(f"  Proba      : {result['probabilities']}")
