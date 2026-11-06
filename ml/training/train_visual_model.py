"""
PhishGuard – Visual Classifier Training Pipeline (Phase 2, Steps 2.3 & 2.4)

Partitions processed visual features (datasets/processed/visual_features.csv)
using registered-domain grouping to guarantee zero domain leakage between
training and testing. Trains a dedicated visual RandomForestClassifier on the
30 canonical visual features, evaluates performance, and saves all visual
artefacts independently from the URL detection model.

Ground-Truth & Provenance Policy:
- Phishing (1): Confirmed PhishTank/Zenodo phishing samples (phishtank_verified).
- Legitimate Candidates (0): Tranco popularity candidates (unverified_candidate).
- Critical Limitation: Tranco candidates are unverified candidates and not
  independently verified benign ground truth. All evaluation metrics reflect
  performance against this candidate baseline and must not be described as
  fully validated production performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
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
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features.visual_features import VISUAL_FEATURE_NAMES
from ml.training.dataset_pipeline import split_by_registered_domain

logger = logging.getLogger("phishguard.train_visual_model")

# ── Class Mapping ────────────────────────────────────────────────────
CLASS_UNVERIFIED_CANDIDATE = 0
CLASS_PHISHING = 1

CLASS_MAP: dict[int, str] = {
    CLASS_UNVERIFIED_CANDIDATE: "unverified_candidate",
    CLASS_PHISHING: "phishing",
}

LABEL_NAME_TO_INT: dict[str, int] = {
    "unverified_candidate": CLASS_UNVERIFIED_CANDIDATE,
    "legitimate": CLASS_UNVERIFIED_CANDIDATE,  # map legacy/raw label string
    "0": CLASS_UNVERIFIED_CANDIDATE,
    "phishing": CLASS_PHISHING,
    "1": CLASS_PHISHING,
}

# ── File Defaults ────────────────────────────────────────────────────
DEFAULT_INPUT_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "visual_features.csv")
DEFAULT_TRAIN_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "visual_train.csv")
DEFAULT_TEST_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "visual_test.csv")

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "visual_model.joblib")
DEFAULT_FEATURES_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "visual_feature_names.json")
DEFAULT_CLASS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "visual_class_mapping.json")
DEFAULT_EVAL_PATH = os.path.join(PROJECT_ROOT, "ml", "evaluation", "visual_model_eval.json")
DEFAULT_DOCS_EVAL_PATH = os.path.join(PROJECT_ROOT, "docs", "visual_model_evaluation.md")

URL_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "url_model.joblib")

METADATA_COLUMNS: list[str] = [
    "csv_line_num",
    "original_url",
    "normalized_url",
    "registered_domain",
    "original_label",
    "threat_category",
    "label_provenance",
    "screenshot_path",
    "screenshot_sha256",
]

CSV_FIELDNAMES: list[str] = METADATA_COLUMNS + VISUAL_FEATURE_NAMES

DATASET_LIMITATION_NOTICE = (
    "CRITICAL LIMITATION: Legitimate class consists of Tranco candidates "
    "('unverified_candidate') rather than manually verified benign ground truth. "
    "The visual classifier cannot be claimed as fully validated or deployed as a "
    "standalone production oracle without human/hybrid corroboration."
)


# ── Step 2.3: Dataset Splitting ──────────────────────────────────────

def split_visual_dataset(
    input_csv_path: str | os.PathLike = DEFAULT_INPUT_CSV,
    train_csv_path: str | os.PathLike = DEFAULT_TRAIN_CSV,
    test_csv_path: str | os.PathLike = DEFAULT_TEST_CSV,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Partition visual feature records into domain-separated train and test sets.

    Parameters
    ----------
    input_csv_path : str | os.PathLike
        Path to processed visual_features.csv.
    train_csv_path : str | os.PathLike
        Destination path for visual_train.csv.
    test_csv_path : str | os.PathLike
        Destination path for visual_test.csv.
    test_size : float
        Fraction of domains allocated to the test set (default: 0.2).
    random_state : int
        Seed for reproducible domain partitioning.

    Returns
    -------
    tuple[list[dict], list[dict], dict]
        (train_records, test_records, split_summary)
    """
    in_p = Path(input_csv_path).resolve()
    if not in_p.is_file():
        raise FileNotFoundError(f"Visual features dataset not found: {in_p}")

    records: list[dict[str, Any]] = []
    with open(in_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    logger.info("Loaded %d records from %s for domain-grouped splitting.", len(records), in_p)

    train_recs, test_recs = split_by_registered_domain(
        records=records,
        test_size=test_size,
        random_state=random_state,
        stratify_by_category=True,
    )

    # Verification: assert 0 domain overlap
    train_domains = {r["registered_domain"] for r in train_recs}
    test_domains = {r["registered_domain"] for r in test_recs}
    domain_overlap = train_domains & test_domains
    if domain_overlap:
        raise RuntimeError(
            f"Domain leakage detected! Overlapping registered domains: {domain_overlap}"
        )

    # Verification: assert 0 screenshot hash overlap
    train_hashes = {r["screenshot_sha256"] for r in train_recs}
    test_hashes = {r["screenshot_sha256"] for r in test_recs}
    hash_overlap = train_hashes & test_hashes
    if hash_overlap:
        raise RuntimeError(
            f"Screenshot hash leakage detected! Overlapping SHA-256 hashes: {hash_overlap}"
        )

    # Write visual_train.csv
    train_p = Path(train_csv_path).resolve()
    train_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_train = str(train_p) + f".tmp.{os.getpid()}"
    with open(tmp_train, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(train_recs)
    os.replace(tmp_train, str(train_p))

    # Write visual_test.csv
    test_p = Path(test_csv_path).resolve()
    test_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_test = str(test_p) + f".tmp.{os.getpid()}"
    with open(tmp_test, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(test_recs)
    os.replace(tmp_test, str(test_p))

    summary = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_csv_path": str(in_p).replace("\\", "/"),
        "train_csv_path": str(train_p).replace("\\", "/"),
        "test_csv_path": str(test_p).replace("\\", "/"),
        "total_samples": len(records),
        "train_samples": len(train_recs),
        "test_samples": len(test_recs),
        "train_class_distribution": dict(Counter(r["threat_category"] for r in train_recs)),
        "test_class_distribution": dict(Counter(r["threat_category"] for r in test_recs)),
        "train_unique_domains": len(train_domains),
        "test_unique_domains": len(test_domains),
        "domain_overlap_count": len(domain_overlap),
        "hash_overlap_count": len(hash_overlap),
        "random_state": random_state,
        "test_size": test_size,
    }

    logger.info(
        "Split complete: Train=%d (%s), Test=%d (%s). Domain overlap=0.",
        len(train_recs), summary["train_class_distribution"],
        len(test_recs), summary["test_class_distribution"],
    )
    return train_recs, test_recs, summary


# ── Step 2.4: Train Visual Classifier ────────────────────────────────

def extract_feature_matrix(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract strictly the 30 canonical visual features and binary targets from records.

    Guarantees that no URLs, registered domains, screenshot hashes, or metadata
    enter the feature matrix.
    """
    X_rows: list[list[float]] = []
    y_vals: list[int] = []

    for r in records:
        row_feats: list[float] = []
        for feat_name in VISUAL_FEATURE_NAMES:
            val = float(r[feat_name])
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"Invalid feature value in {feat_name}: {val}")
            row_feats.append(val)
        X_rows.append(row_feats)

        cat = r.get("threat_category", "").strip().lower()
        if cat not in LABEL_NAME_TO_INT:
            orig = r.get("original_label", "").strip().lower()
            cat = orig if orig in LABEL_NAME_TO_INT else "unverified_candidate"

        y_vals.append(LABEL_NAME_TO_INT[cat])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_vals, dtype=np.int64)

    # Verification: assert exact dimensionality
    if X.shape[1] != len(VISUAL_FEATURE_NAMES):
        raise ValueError(
            f"Feature matrix dimension mismatch: expected {len(VISUAL_FEATURE_NAMES)}, got {X.shape[1]}"
        )

    return X, y


def train_visual_classifier(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    model_path: str | os.PathLike = DEFAULT_MODEL_PATH,
    features_path: str | os.PathLike = DEFAULT_FEATURES_PATH,
    class_mapping_path: str | os.PathLike = DEFAULT_CLASS_MAPPING_PATH,
    eval_path: str | os.PathLike = DEFAULT_EVAL_PATH,
    docs_eval_path: str | os.PathLike = DEFAULT_DOCS_EVAL_PATH,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Train, cross-validate, evaluate, and save the separate visual classifier.

    Parameters
    ----------
    train_records : list[dict[str, Any]]
        Training set records.
    test_records : list[dict[str, Any]]
        Held-out testing set records.
    model_path : str | os.PathLike
        Destination path for visual_model.joblib.
    features_path : str | os.PathLike
        Destination path for visual_feature_names.json.
    class_mapping_path : str | os.PathLike
        Destination path for visual_class_mapping.json.
    eval_path : str | os.PathLike
        Destination path for visual_model_eval.json.
    docs_eval_path : str | os.PathLike
        Destination path for docs/visual_model_evaluation.md.
    n_estimators : int
        Number of trees in RandomForest.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple[RandomForestClassifier, dict[str, Any]]
        (trained_model, evaluation_report)
    """
    # ── Safety Check: Protect existing URL model ────────────────────
    target_abs = str(Path(model_path).resolve())
    url_abs = str(Path(URL_MODEL_PATH).resolve())
    if target_abs == url_abs:
        raise ValueError(
            f"SAFETY VIOLATION: Refusing to overwrite existing URL model at {URL_MODEL_PATH}!"
        )

    # ── 1. Prepare Feature Matrices (No Metadata) ───────────────────
    X_train, y_train = extract_feature_matrix(train_records)
    X_test, y_test = extract_feature_matrix(test_records)

    logger.info("X_train shape: %s, X_test shape: %s", X_train.shape, X_test.shape)

    # ── 2. Cross-Validation on Training Set (5-Fold Stratified) ─────
    logger.info("Running 5-fold cross-validation on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    cv_f1_scores = cross_val_score(cv_clf, X_train, y_train, cv=cv, scoring="f1")
    cv_acc_scores = cross_val_score(cv_clf, X_train, y_train, cv=cv, scoring="accuracy")

    # ── 3. Fit Model on Full Training Set ───────────────────────────
    logger.info("Fitting visual RandomForestClassifier on training data...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── 4. Evaluate on Held-Out Test Set ────────────────────────────
    logger.info("Evaluating visual classifier on held-out test data...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    roc_auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_rep_dict = classification_report(
        y_test,
        y_pred,
        target_names=["unverified_candidate", "phishing"],
        output_dict=True,
    )
    clf_rep_text = classification_report(
        y_test,
        y_pred,
        target_names=["unverified_candidate", "phishing"],
    )

    # Feature importances
    importances = clf.feature_importances_
    feat_imp = sorted(
        [{"feature": name, "importance": round(float(imp), 5)} for name, imp in zip(VISUAL_FEATURE_NAMES, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # ── 5. Assemble Evaluation Report ───────────────────────────────
    evaluation_report = {
        "metadata": {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": "RandomForestClassifier",
            "feature_domain": "visual_screenshots",
            "features_count": len(VISUAL_FEATURE_NAMES),
            "random_state": random_state,
            "n_estimators": n_estimators,
            "limitation_notice": DATASET_LIMITATION_NOTICE,
        },
        "dataset_split": {
            "train_samples_count": int(len(train_records)),
            "test_samples_count": int(len(test_records)),
            "train_class_counts": {k: int(v) for k, v in Counter(r["threat_category"] for r in train_records).items()},
            "test_class_counts": {k: int(v) for k, v in Counter(r["threat_category"] for r in test_records).items()},
            "domain_leakage_count": 0,
        },
        "cross_validation_train_5fold": {
            "f1_scores": [round(float(s), 4) for s in cv_f1_scores],
            "f1_mean": round(float(cv_f1_scores.mean()), 4),
            "f1_std": round(float(cv_f1_scores.std()), 4),
            "accuracy_scores": [round(float(s), 4) for s in cv_acc_scores],
            "accuracy_mean": round(float(cv_acc_scores.mean()), 4),
            "accuracy_std": round(float(cv_acc_scores.std()), 4),
        },
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": {
                "true_negative_candidate": cm[0][0],
                "false_positive_candidate_as_phish": cm[0][1],
                "false_negative_phish_as_candidate": cm[1][0],
                "true_positive_phish": cm[1][1],
            },
            "classification_report": clf_rep_dict,
        },
        "top_10_feature_importances": feat_imp[:10],
    }

    # ── 6. Save Artefacts ───────────────────────────────────────────
    # A. Model
    mod_p = Path(model_path).resolve()
    mod_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_mod = str(mod_p) + f".tmp.{os.getpid()}"
    joblib.dump(clf, tmp_mod)
    os.replace(tmp_mod, str(mod_p))
    logger.info("Saved visual model to: %s", mod_p)

    # B. Feature Names
    feat_p = Path(features_path).resolve()
    feat_p.parent.mkdir(parents=True, exist_ok=True)
    with open(feat_p, "w", encoding="utf-8") as f:
        json.dump(VISUAL_FEATURE_NAMES, f, indent=2)
    logger.info("Saved visual feature names to: %s", feat_p)

    # C. Class Mapping
    cls_p = Path(class_mapping_path).resolve()
    cls_p.parent.mkdir(parents=True, exist_ok=True)
    with open(cls_p, "w", encoding="utf-8") as f:
        json.dump(CLASS_MAP, f, indent=2)
    logger.info("Saved visual class mapping to: %s", cls_p)

    # D. JSON Evaluation Report
    eval_p = Path(eval_path).resolve()
    eval_p.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_p, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)
    logger.info("Saved visual model evaluation report to: %s", eval_p)

    # E. Markdown Evaluation Report
    docs_p = Path(docs_eval_path).resolve()
    docs_p.parent.mkdir(parents=True, exist_ok=True)
    md_content = _build_markdown_eval_report(evaluation_report, clf_rep_text)
    with open(docs_p, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved visual model evaluation markdown to: %s", docs_p)

    return clf, evaluation_report


def _build_markdown_eval_report(eval_dict: dict[str, Any], clf_rep_text: str) -> str:
    """Generate human-readable Markdown evaluation documentation."""
    m = eval_dict["test_metrics"]
    cv = eval_dict["cross_validation_train_5fold"]
    cm = m["confusion_matrix"]
    top_feats = eval_dict["top_10_feature_importances"]

    md = f"""# PhishGuard – Visual Classifier Evaluation Report (Phase 2, Step 2.4)

**Evaluation Date**: {eval_dict['metadata']['execution_timestamp']}  
**Model Architecture**: `RandomForestClassifier(n_estimators=100, random_state=42)`  
**Input Features**: Exactly 30 canonical visual features (0 metadata fields)  
**Partitioning**: Registered-domain grouped splitting (0% domain overlap)  

> [!WARNING]
> **Data Governance Notice & Provenance Limitation**:  
> {eval_dict['metadata']['limitation_notice']}  
> All metrics reported below represent classification performance against the candidate dataset baseline.

---

## 1. Executive Performance Summary

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **{m['accuracy'] * 100:.2f}%** | Overall classification accuracy on held-out domains |
| **Macro F1-Score** | **{m['macro_f1']:.4f}** | Unweighted mean of both classes |
| **Phishing F1-Score** | **{m['f1_score']:.4f}** | Primary threat detection capability |
| **Phishing Precision** | **{m['precision']:.4f}** | Proportion of visual phishing alarms that are true threats |
| **Phishing Recall** | **{m['recall']:.4f}** | Proportion of actual phishing screenshots detected |
| **ROC-AUC Score** | **{m['roc_auc']:.4f}** | Area under the Receiver Operating Characteristic curve |
| **5-Fold CV F1 (Train)** | **{cv['f1_mean']:.4f} (±{cv['f1_std']:.4f})** | 5-fold stratified cross-validation on training set |

---

## 2. Confusion Matrix (Held-Out Test Set: N = {eval_dict['dataset_split']['test_samples_count']})

| Ground Truth \\ Prediction | Predicted Candidate (0) | Predicted Phishing (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Candidate (Tranco)** | **{cm['true_negative_candidate']}** (TN) | {cm['false_positive_candidate_as_phish']} (FP) | {cm['true_negative_candidate'] + cm['false_positive_candidate_as_phish']} |
| **Actual Phishing (PhishTank)** | {cm['false_negative_phish_as_candidate']} (FN) | **{cm['true_positive_phish']}** (TP) | {cm['false_negative_phish_as_candidate'] + cm['true_positive_phish']} |

---

## 3. Detailed Classification Report

```
{clf_rep_text}
```

---

## 4. Top 10 Visual Feature Importances

| Rank | Feature Name | Gini Importance |
| :---: | :--- | :---: |
"""
    for rank, item in enumerate(top_feats, start=1):
        md += f"| {rank} | `{item['feature']}` | {item['importance']:.5f} |\n"

    md += """
---

## 5. Artifact Manifest

- Trained Model: `ml/models/visual_model.joblib`
- Feature Names: `ml/models/visual_feature_names.json`
- Class Mapping: `ml/models/visual_class_mapping.json`
- Evaluation JSON: `ml/evaluation/visual_model_eval.json`
- Train Split: `datasets/processed/visual_train.csv`
- Test Split: `datasets/processed/visual_test.csv`
"""
    return md


# ── Pipeline Runner & CLI ───────────────────────────────────────────

def run_pipeline(
    input_csv: str | os.PathLike = DEFAULT_INPUT_CSV,
    train_csv: str | os.PathLike = DEFAULT_TRAIN_CSV,
    test_csv: str | os.PathLike = DEFAULT_TEST_CSV,
    model_path: str | os.PathLike = DEFAULT_MODEL_PATH,
    features_path: str | os.PathLike = DEFAULT_FEATURES_PATH,
    class_mapping_path: str | os.PathLike = DEFAULT_CLASS_MAPPING_PATH,
    eval_path: str | os.PathLike = DEFAULT_EVAL_PATH,
    docs_eval_path: str | os.PathLike = DEFAULT_DOCS_EVAL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
) -> tuple[RandomForestClassifier | None, dict[str, Any]]:
    """Execute end-to-end splitting, training, and evaluation."""
    # 1. Split
    train_recs, test_recs, split_summary = split_visual_dataset(
        input_csv_path=input_csv,
        train_csv_path=train_csv,
        test_csv_path=test_csv,
        test_size=test_size,
        random_state=random_state,
    )

    print("\n" + "=" * 65)
    print(" PHISHGUARD VISUAL DATASET SPLITTING SUMMARY (Step 2.3)")
    print("=" * 65)
    print(f" Total Input Samples         : {split_summary['total_samples']}")
    print(f" Train Split (80%)           : {split_summary['train_samples']} samples ({split_summary['train_unique_domains']} domains)")
    print(f"   ├─ Phishing               : {split_summary['train_class_distribution'].get('phishing', 0)}")
    print(f"   └─ Unverified Candidates  : {split_summary['train_class_distribution'].get('unverified_candidate', 0)}")
    print(f" Test Split (20%)            : {split_summary['test_samples']} samples ({split_summary['test_unique_domains']} domains)")
    print(f"   ├─ Phishing               : {split_summary['test_class_distribution'].get('phishing', 0)}")
    print(f"   └─ Unverified Candidates  : {split_summary['test_class_distribution'].get('unverified_candidate', 0)}")
    print(f" Domain Leakage (Overlap)    : {split_summary['domain_overlap_count']} domains")
    print(f" Screenshot Hash Overlap     : {split_summary['hash_overlap_count']} hashes")
    print("=" * 65)

    if dry_run:
        print("[DRY-RUN COMPLETE] Skipping model training and persistence.\n")
        return None, split_summary

    # 2. Train & Evaluate
    clf, eval_report = train_visual_classifier(
        train_records=train_recs,
        test_records=test_recs,
        model_path=model_path,
        features_path=features_path,
        class_mapping_path=class_mapping_path,
        eval_path=eval_path,
        docs_eval_path=docs_eval_path,
        random_state=random_state,
    )

    m = eval_report["test_metrics"]
    cv = eval_report["cross_validation_train_5fold"]
    cm = m["confusion_matrix"]

    print("\n" + "=" * 65)
    print(" PHISHGUARD VISUAL CLASSIFIER EVALUATION (Step 2.4)")
    print("=" * 65)
    print(f" 5-Fold CV F1 on Train (Mean): {cv['f1_mean']:.4f} (±{cv['f1_std']:.4f})")
    print(f" Test Accuracy               : {m['accuracy'] * 100:.2f}%")
    print(f" Test Macro F1-Score         : {m['macro_f1']:.4f}")
    print(f" Test Phishing F1-Score      : {m['f1_score']:.4f}")
    print(f" Test Phishing Precision     : {m['precision']:.4f}")
    print(f" Test Phishing Recall        : {m['recall']:.4f}")
    print(f" Test ROC-AUC Score          : {m['roc_auc']:.4f}")
    print(" Confusion Matrix (Test Set):")
    print(f"   TN (Candidate): {cm['true_negative_candidate']:<4} | FP (as Phish): {cm['false_positive_candidate_as_phish']}")
    print(f"   FN (as Cand)  : {cm['false_negative_phish_as_candidate']:<4} | TP (Phish)   : {cm['true_positive_phish']}")
    print("\n Top 5 Visual Features by Importance:")
    for f in eval_report["top_10_feature_importances"][:5]:
        print(f"   - {f['feature']:<25}: {f['importance']:.5f}")
    print("=" * 65)
    print(f"[LIMITATION] {DATASET_LIMITATION_NOTICE}\n")

    return clf, eval_report


def build_cli() -> argparse.ArgumentParser:
    """Build CLI parser for visual dataset splitting and model training."""
    parser = argparse.ArgumentParser(
        description="PhishGuard – Visual Dataset Splitting and Classifier Training (Steps 2.3 & 2.4)"
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help=f"Path to visual_features.csv (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--train-csv",
        default=DEFAULT_TRAIN_CSV,
        help=f"Path to output visual_train.csv (default: {DEFAULT_TRAIN_CSV})",
    )
    parser.add_argument(
        "--test-csv",
        default=DEFAULT_TEST_CSV,
        help=f"Path to output visual_test.csv (default: {DEFAULT_TEST_CSV})",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to output visual_model.joblib (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--eval-path",
        default=DEFAULT_EVAL_PATH,
        help=f"Path to output visual_model_eval.json (default: {DEFAULT_EVAL_PATH})",
    )
    parser.add_argument(
        "--docs-eval-path",
        default=DEFAULT_DOCS_EVAL_PATH,
        help=f"Path to output visual_model_evaluation.md (default: {DEFAULT_DOCS_EVAL_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dataset splitting only without training or saving model files.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_cli()
    args = parser.parse_args()

    try:
        run_pipeline(
            input_csv=args.input_csv,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            model_path=args.model_path,
            eval_path=args.eval_path,
            docs_eval_path=args.docs_eval_path,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as exc:
        logger.exception("Visual training pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

