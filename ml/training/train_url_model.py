"""
PhishGuard – URL Model Training Pipeline (Three-Class Architecture)

Loads a processed CSV (e.g. from dataset_pipeline.py), maps samples to
three distinct threat categories:
    0 = legitimate
    1 = phishing
    2 = malware_distribution

Partitions data strictly using registered-domain grouping to guarantee zero
domain leakage between training and testing. Trains a RandomForestClassifier
on the 20 canonical URL features, evaluates comprehensive performance
(accuracy, macro/weighted precision/recall/F1, 3-class confusion matrix,
and per-class classification report), and saves all artefacts:
- Model (url_model.joblib)
- Feature names (feature_names.json)
- Class mappings (class_mapping.json)
- Evaluation & Provenance metrics (url_model_eval.json)

No data is invented or fabricated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features.url_features import FEATURE_NAMES
from ml.training.dataset_pipeline import extract_registered_domain, split_by_registered_domain

logger = logging.getLogger(__name__)

# ── Class Mapping (3-Class Architecture) ─────────────────────────────
CLASS_LEGITIMATE = 0
CLASS_PHISHING = 1
CLASS_MALWARE = 2

CLASS_MAP = {
    CLASS_LEGITIMATE: "legitimate",
    CLASS_PHISHING: "phishing",
    CLASS_MALWARE: "malware_distribution",
}

LABEL_NAME_TO_INT = {
    "legitimate": CLASS_LEGITIMATE,
    "phishing": CLASS_PHISHING,
    "malware_distribution": CLASS_MALWARE,
}

# ── Defaults ────────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "url_model.joblib")
_DEFAULT_FEATURES_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "feature_names.json")
_DEFAULT_CLASS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "class_mapping.json")
_DEFAULT_EVAL_PATH = os.path.join(PROJECT_ROOT, "ml", "evaluation", "url_model_eval.json")

MIN_SAMPLES = 20          # minimum rows required to train
MIN_CLASS_SAMPLES = 5     # minimum rows per class
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ── Label Conversion ────────────────────────────────────────────────

def convert_to_three_class_label(threat_category: Any | None, label: Any | None = None) -> int:
    """Convert threat_category or legacy binary label to canonical 3-class integer.

    0 = legitimate
    1 = phishing
    2 = malware_distribution
    """
    if threat_category is not None and str(threat_category).strip() != "":
        tc = str(threat_category).strip().lower()
        if tc in ("legitimate", "0"):
            return CLASS_LEGITIMATE
        if tc in ("phishing", "1"):
            return CLASS_PHISHING
        if tc in ("malware_distribution", "malware", "2"):
            return CLASS_MALWARE

    if label is not None and str(label).strip() != "":
        lbl = str(label).strip().lower()
        if lbl in ("0", "legitimate"):
            return CLASS_LEGITIMATE
        if lbl in ("1", "phishing"):
            return CLASS_PHISHING
        if lbl in ("2", "malware_distribution", "malware"):
            return CLASS_MALWARE

    raise ValueError(f"Cannot map to 3-class label: threat_category='{threat_category}', label='{label}'")


# ── SHA-256 Provenance Helper ───────────────────────────────────────

def _compute_sha256(filepath: str) -> str:
    """Compute SHA-256 digest of a dataset file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ── Public API ──────────────────────────────────────────────────────

def train_model(
    dataset_path: str,
    model_path: str | None = None,
    features_path: str | None = None,
    eval_path: str | None = None,
    class_mapping_path: str | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """End-to-end 3-class training pipeline with registered-domain grouping.

    Parameters
    ----------
    dataset_path : str
        Path to processed CSV (must contain 20 feature columns + label or threat_category).
    model_path, features_path, eval_path, class_mapping_path : str | None
        Override default artefact save paths.
    test_size : float
        Fraction of domain groups held out for evaluation (default 0.2).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        ``{"metrics": {...}, "provenance": {...}, "model_path": ..., ...}``
    """
    model_path = os.path.abspath(model_path or _DEFAULT_MODEL_PATH)
    features_path = os.path.abspath(features_path or _DEFAULT_FEATURES_PATH)
    eval_path = os.path.abspath(eval_path or _DEFAULT_EVAL_PATH)
    class_mapping_path = os.path.abspath(class_mapping_path or _DEFAULT_CLASS_MAPPING_PATH)

    # ── 1. Load & validate dataset ──────────────────────────────
    records = _load_dataset(dataset_path)
    total_samples = len(records)
    class_counts_before = dict(Counter(r["label"] for r in records))
    logger.info("Loaded %d records. Class distribution before split: %s", total_samples, class_counts_before)

    # ── 2. Registered-domain-grouped train/test split ───────────
    # Guarantees zero domain overlap between train and test folds
    train_records, test_records = split_by_registered_domain(
        records,
        test_size=test_size,
        random_state=random_state,
        stratify_by_category=True,
    )

    train_domains = {r["registered_domain"] for r in train_records}
    test_domains = {r["registered_domain"] for r in test_records}
    domain_overlap = train_domains & test_domains

    if domain_overlap:
        raise RuntimeError(f"Domain leakage detected in split! Overlapping domains: {domain_overlap}")

    class_counts_train = dict(Counter(r["label"] for r in train_records))
    class_counts_test = dict(Counter(r["label"] for r in test_records))

    logger.info(
        "Domain-grouped split: %d train (%d domains), %d test (%d domains), overlap: %d",
        len(train_records),
        len(train_domains),
        len(test_records),
        len(test_domains),
        len(domain_overlap),
    )
    logger.info("Class distribution in train: %s", class_counts_train)
    logger.info("Class distribution in test:  %s", class_counts_test)

    # ── 3. Extract Feature Arrays ───────────────────────────────
    X_train = np.array([r["features"] for r in train_records], dtype=np.float64)
    y_train = np.array([r["label"] for r in train_records], dtype=np.int32)
    X_test = np.array([r["features"] for r in test_records], dtype=np.float64)
    y_test = np.array([r["label"] for r in test_records], dtype=np.int32)

    # ── 4. Train Random Forest (Fit strictly on training data) ──
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    logger.info("RandomForestClassifier successfully fitted on %d training samples.", len(X_train))

    # ── 5. Evaluate on Held-Out Test Set ────────────────────────
    y_pred = clf.predict(X_test)
    classes_present = sorted(list(set(y_train) | set(y_test)))
    metrics = _compute_metrics(y_test, y_pred, classes=classes_present)
    logger.info("Evaluation metrics: accuracy=%.4f, f1_macro=%.4f, f1_weighted=%.4f",
                metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"])

    # ── 6. Compile Provenance & Save Artefacts ──────────────────
    dataset_sha256 = _compute_sha256(dataset_path)
    provenance = {
        "dataset_path": os.path.abspath(dataset_path),
        "dataset_sha256": dataset_sha256,
        "total_samples": total_samples,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "unique_domains_total": len(train_domains | test_domains),
        "unique_domains_train": len(train_domains),
        "unique_domains_test": len(test_domains),
        "domain_leakage_count": 0,
        "domain_leakage_rate": "0.00%",
        "class_counts_before_split": {
            CLASS_MAP.get(k, str(k)): v for k, v in class_counts_before.items()
        },
        "class_counts_train": {
            CLASS_MAP.get(k, str(k)): v for k, v in class_counts_train.items()
        },
        "class_counts_test": {
            CLASS_MAP.get(k, str(k)): v for k, v in class_counts_test.items()
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "feature_count": len(FEATURE_NAMES),
    }

    metrics["dataset_provenance"] = provenance

    _save_model(clf, model_path)
    _save_feature_names(features_path)
    _save_class_mapping(class_mapping_path)
    _save_eval(metrics, eval_path)

    return {
        "metrics": metrics,
        "provenance": provenance,
        "model_path": model_path,
        "features_path": features_path,
        "class_mapping_path": class_mapping_path,
        "eval_path": eval_path,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "class_counts_before_split": class_counts_before,
        "class_counts_train": class_counts_train,
        "class_counts_test": class_counts_test,
        "domain_leakage": 0,
    }


# ── Dataset Loading & Validation ────────────────────────────────────

def _load_dataset(path: str) -> list[dict[str, Any]]:
    """Load CSV, validate columns and class counts, returning structured records."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset is empty: {path}")

        header_set = {c.strip() for c in reader.fieldnames}
        missing_features = [fn for fn in FEATURE_NAMES if fn not in header_set]
        if missing_features:
            raise ValueError(f"Dataset is missing feature columns: {missing_features}")

        has_label = "label" in header_set or "binary_label" in header_set
        has_cat = "threat_category" in header_set
        if not (has_label or has_cat):
            raise ValueError("Dataset is missing the 'label' column")

        rows = list(reader)

    if len(rows) < MIN_SAMPLES:
        raise ValueError(
            f"Dataset too small: {len(rows)} rows (minimum {MIN_SAMPLES}). "
            "Please provide a larger dataset."
        )

    records: list[dict[str, Any]] = []
    class_counts: dict[int, int] = defaultdict(int)

    for r in rows:
        url = r.get("url", "").strip()
        reg_dom = r.get("registered_domain") or extract_registered_domain(url) or "unknown_domain"
        cat_val = r.get("threat_category")
        lbl_val = r.get("label") or r.get("binary_label")
        y_int = convert_to_three_class_label(cat_val, lbl_val)
        class_counts[y_int] += 1

        feats = [float(r[fn]) for fn in FEATURE_NAMES]
        records.append({
            "url": url,
            "registered_domain": reg_dom,
            "threat_category": CLASS_MAP.get(y_int, str(y_int)),
            "label": y_int,
            "features": feats,
        })

    unique_classes = set(class_counts.keys())
    if len(unique_classes) < 2:
        raise ValueError(
            f"Dataset must contain at least 2 distinct classes (both classes required). Found: {unique_classes}"
        )

    for cls, count in class_counts.items():
        if count < MIN_CLASS_SAMPLES:
            raise ValueError(
                f"Class {cls} has only {count} samples (minimum {MIN_CLASS_SAMPLES})"
            )

    return records


# ── Metrics Computation ─────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[int] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro/weighted precision/recall/F1, CM and per-class report."""
    if classes is None:
        classes = sorted(list(set(y_true.tolist()) | set(y_pred.tolist())))

    accuracy = float(accuracy_score(y_true, y_pred))

    # Macro-averaged metrics
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Weighted-averaged metrics
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    # Per-class classification report
    target_names = [CLASS_MAP.get(c, str(c)) for c in classes]
    clf_report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Top-level backward compatibility values
    if len(classes) == 2:
        top_precision = float(precision_score(y_true, y_pred, zero_division=0))
        top_recall = float(recall_score(y_true, y_pred, zero_division=0))
        top_f1 = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        top_precision = precision_weighted
        top_recall = recall_weighted
        top_f1 = f1_weighted

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(top_precision, 4),
        "recall": round(top_recall, 4),
        "f1_score": round(top_f1, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "macro_avg": {
            "precision": round(precision_macro, 4),
            "recall": round(recall_macro, 4),
            "f1_score": round(f1_macro, 4),
        },
        "weighted_avg": {
            "precision": round(precision_weighted, 4),
            "recall": round(recall_weighted, 4),
            "f1_score": round(f1_weighted, 4),
        },
        "confusion_matrix": cm,
        "classification_report": clf_report,
    }


# ── Artefact Persistence ────────────────────────────────────────────

def _save_model(clf: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(clf, path)
    logger.info("Model saved: %s", path)


def _save_feature_names(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    logger.info("Feature names saved: %s", path)


def _save_class_mapping(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mapping = {
        "class_to_name": {str(k): v for k, v in CLASS_MAP.items()},
        "name_to_class": LABEL_NAME_TO_INT,
        "num_classes": len(CLASS_MAP),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Class mapping saved: %s", path)


def _save_eval(metrics: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation & provenance saved: %s", path)


# ── CLI Entry Point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Train PhishGuard URL 3-Class model.")
    parser.add_argument("dataset", help="Path to processed CSV dataset")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--eval-path", default=None)
    parser.add_argument("--class-mapping-path", default=None)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args()

    result = train_model(
        args.dataset,
        model_path=args.model_path,
        features_path=args.features_path,
        eval_path=args.eval_path,
        class_mapping_path=args.class_mapping_path,
        test_size=args.test_size,
    )
    print(json.dumps(result["metrics"], indent=2))
