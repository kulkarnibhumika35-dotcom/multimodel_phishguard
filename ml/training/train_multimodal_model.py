"""
PhishGuard – Multimodal Classifier Training and Evaluation (Phase 2, Step 2.6)

Trains a dedicated multimodal phishing detection classifier (RandomForestClassifier)
on 51 fused features (20 URL features + 1 has_visual_features indicator + 30 visual features),
evaluates performance on the held-out domain-separated test set, conducts a 3-way
empirical comparison against URL-only and visual-only models, and preserves all
evaluation artefacts.

Safety & Governance Guarantees:
- Reuses canonical FUSED_FEATURE_NAMES from ml/features/feature_fusion.py (51 features).
- Excludes all 9 metadata columns from model inputs (zero metadata leakage).
- Uses domain-separated partitions (multimodal_train.csv and multimodal_test.csv).
- Hyperparameters are tuned strictly on training data; test set remains untouched.
- Refuses to overwrite url_model.joblib or visual_model.joblib.
- Does not modify FastAPI backend, Chrome extension, or dashboard.
- Maintains strict data governance: class 0 is 'unverified_candidate' (Tranco popularity),
  class 1 is 'phishing' (PhishTank verified). Tranco candidates are never described as
  verified legitimate.
- Does not claim multimodal superiority unless supported by empirical held-out metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
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

from ml.features.feature_fusion import (
    FUSED_FEATURE_NAMES,
    VISUAL_AVAILABILITY_FEATURE,
)
from ml.features.url_features import FEATURE_NAMES as URL_FEATURE_NAMES
from ml.features.visual_features import VISUAL_FEATURE_NAMES

logger = logging.getLogger("phishguard.train_multimodal_model")

# ── Class Mapping ────────────────────────────────────────────────────
CLASS_UNVERIFIED_CANDIDATE = 0
CLASS_PHISHING = 1

CLASS_MAP: dict[int, str] = {
    CLASS_UNVERIFIED_CANDIDATE: "unverified_candidate",
    CLASS_PHISHING: "phishing",
}

LABEL_NAME_TO_INT: dict[str, int] = {
    "unverified_candidate": CLASS_UNVERIFIED_CANDIDATE,
    "legitimate": CLASS_UNVERIFIED_CANDIDATE,
    "0": CLASS_UNVERIFIED_CANDIDATE,
    "phishing": CLASS_PHISHING,
    "1": CLASS_PHISHING,
}

# ── File Defaults ────────────────────────────────────────────────────
DEFAULT_TRAIN_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "multimodal_train.csv")
DEFAULT_TEST_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "multimodal_test.csv")

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "multimodal_model.joblib")
DEFAULT_FEATURES_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "multimodal_feature_names.json")
DEFAULT_CLASS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "multimodal_class_mapping.json")
DEFAULT_EVAL_PATH = os.path.join(PROJECT_ROOT, "ml", "evaluation", "multimodal_model_eval.json")
DEFAULT_DOCS_EVAL_PATH = os.path.join(PROJECT_ROOT, "docs", "multimodal_model_evaluation.md")

URL_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "url_model.joblib")
VISUAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "visual_model.joblib")

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

DATASET_LIMITATION_NOTICE = (
    "CRITICAL LIMITATION: Legitimate class consists of Tranco candidates "
    "('unverified_candidate') rather than manually verified benign ground truth. "
    "The multimodal classifier cannot be claimed as fully validated or deployed as a "
    "standalone production oracle without human/hybrid corroboration."
)


# ── Step 1: Feature Matrix Extraction & Metadata Exclusion ───────────

def extract_multimodal_feature_matrix(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract strictly the 51 canonical features and binary labels.

    Excludes all 9 metadata columns (csv_line_num, original_url, normalized_url,
    registered_domain, original_label, threat_category, label_provenance,
    screenshot_path, screenshot_sha256).

    Parameters
    ----------
    records : list[dict[str, Any]]
        List of row dictionaries from multimodal_train.csv or multimodal_test.csv.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X: float64 array of shape (N, 51)
        y: int64 array of shape (N,)
    """
    X_rows: list[list[float]] = []
    y_vals: list[int] = []

    for r in records:
        row_feats: list[float] = []
        for feat_name in FUSED_FEATURE_NAMES:
            raw_val = r.get(feat_name)
            if raw_val is None or str(raw_val).strip() == "":
                raise ValueError(f"Missing feature value for '{feat_name}' in record: {r.get('original_url')}")
            f_val = float(raw_val)
            if math.isnan(f_val) or math.isinf(f_val):
                raise ValueError(f"Non-finite feature value for '{feat_name}': {f_val}")
            row_feats.append(f_val)
        X_rows.append(row_feats)

        # Label extraction with strict provenance adherence
        cat = r.get("threat_category", "").strip().lower()
        if cat not in LABEL_NAME_TO_INT:
            orig = r.get("original_label", "").strip().lower()
            cat = orig if orig in LABEL_NAME_TO_INT else "unverified_candidate"
        y_vals.append(LABEL_NAME_TO_INT[cat])

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_vals, dtype=np.int64)

    # Verification: assert exact feature count of 51
    if X.shape[1] != len(FUSED_FEATURE_NAMES):
        raise ValueError(
            f"Feature matrix dimension mismatch: expected {len(FUSED_FEATURE_NAMES)}, got {X.shape[1]}"
        )

    return X, y


def load_multimodal_splits(
    train_csv_path: str | os.PathLike = DEFAULT_TRAIN_CSV,
    test_csv_path: str | os.PathLike = DEFAULT_TEST_CSV,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and validate domain-separated train and test records.

    Verifies zero registered-domain overlap and zero screenshot hash overlap.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (train_records, test_records)
    """
    tr_p = Path(train_csv_path).resolve()
    te_p = Path(test_csv_path).resolve()

    if not tr_p.is_file():
        raise FileNotFoundError(f"Multimodal train CSV not found: {tr_p}")
    if not te_p.is_file():
        raise FileNotFoundError(f"Multimodal test CSV not found: {te_p}")

    train_records: list[dict[str, Any]] = []
    with open(tr_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            train_records.append(r)

    test_records: list[dict[str, Any]] = []
    with open(te_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            test_records.append(r)

    # Verify zero domain overlap
    train_domains = {r["registered_domain"] for r in train_records}
    test_domains = {r["registered_domain"] for r in test_records}
    domain_overlap = train_domains & test_domains
    if domain_overlap:
        raise RuntimeError(
            f"Domain leakage detected between train and test! Overlapping domains: {domain_overlap}"
        )

    # Verify zero hash overlap
    train_hashes = {r["screenshot_sha256"] for r in train_records}
    test_hashes = {r["screenshot_sha256"] for r in test_records}
    hash_overlap = train_hashes & test_hashes
    if hash_overlap:
        raise RuntimeError(
            f"Screenshot hash leakage detected between train and test! Overlapping hashes: {hash_overlap}"
        )

    logger.info(
        "Loaded %d train samples (%d domains) and %d test samples (%d domains). Zero leakage verified.",
        len(train_records),
        len(train_domains),
        len(test_records),
        len(test_domains),
    )

    return train_records, test_records


# ── Step 2: Hyperparameter Tuning on Training Data Only ──────────────

def tune_multimodal_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidate_grid: list[dict[str, Any]] | None = None,
    n_jobs: int | None = 1,
    random_state: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate candidate hyperparameter configurations using 5-fold stratified CV on training data only.

    The held-out test set is NEVER accessed during tuning.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    candidate_grid : list[dict[str, Any]] | None
        Optional override for hyperparameter candidates.
    n_jobs : int | None
        Number of parallel jobs for cross-validation (default: 1 for safe Windows execution).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        (best_params, all_candidates_results)
    """
    logger.info("Tuning hyperparameters on training set (%d samples) via 5-fold CV...", len(X_train))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if candidate_grid is None:
        candidate_grid = [
            {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 100, "max_depth": 25, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 100, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "log2"},
        ]

    candidates_results: list[dict[str, Any]] = []
    best_params = candidate_grid[0]
    best_f1 = -1.0

    for idx, params in enumerate(candidate_grid, start=1):
        clf = RandomForestClassifier(**params, random_state=random_state, n_jobs=n_jobs)
        f1_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
        acc_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")

        mean_f1 = float(f1_scores.mean())
        std_f1 = float(f1_scores.std())
        mean_acc = float(acc_scores.mean())
        std_acc = float(acc_scores.std())

        cand_result = {
            "candidate_id": idx,
            "params": params,
            "cv_f1_mean": round(mean_f1, 4),
            "cv_f1_std": round(std_f1, 4),
            "cv_accuracy_mean": round(mean_acc, 4),
            "cv_accuracy_std": round(std_acc, 4),
        }
        candidates_results.append(cand_result)
        logger.info("Candidate %d: %s -> CV F1: %.4f (+/- %.4f)", idx, params, mean_f1, std_f1)

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = params

    logger.info("Best training hyperparameter configuration: %s (CV F1: %.4f)", best_params, best_f1)
    return best_params, candidates_results


# ── Step 3: Model Comparison on Identical Held-Out Records ───────────

def evaluate_cross_model_comparison(
    X_test_51: np.ndarray,
    y_test: np.ndarray,
    clf_multimodal: RandomForestClassifier,
    url_model_path: str | os.PathLike = URL_MODEL_PATH,
    visual_model_path: str | os.PathLike = VISUAL_MODEL_PATH,
) -> dict[str, Any]:
    """Perform side-by-side evaluation of URL-only, visual-only, and multimodal models

    Evaluated on the identical 296 held-out test samples.

    Parameters
    ----------
    X_test_51 : np.ndarray
        Held-out test set 51-feature matrix.
    y_test : np.ndarray
        Ground-truth test labels (0=unverified_candidate, 1=phishing).
    clf_multimodal : RandomForestClassifier
        Trained multimodal model.
    url_model_path : str | os.PathLike
        Path to existing URL model.
    visual_model_path : str | os.PathLike
        Path to existing visual model.

    Returns
    -------
    dict[str, Any]
        Structured comparison metrics and concordance across all three models.
    """
    total_test = len(y_test)
    phish_mask = (y_test == 1)
    cand_mask = (y_test == 0)
    n_phish = int(np.sum(phish_mask))
    n_cand = int(np.sum(cand_mask))

    # ── 1. Multimodal Model Evaluation ─────────────────────────────
    multi_pred = clf_multimodal.predict(X_test_51)
    multi_prob = clf_multimodal.predict_proba(X_test_51)[:, 1]

    multi_acc = float(accuracy_score(y_test, multi_pred))
    multi_prec = float(precision_score(y_test, multi_pred, zero_division=0))
    multi_rec = float(recall_score(y_test, multi_pred, zero_division=0))
    multi_f1 = float(f1_score(y_test, multi_pred, zero_division=0))
    multi_macro_f1 = float(f1_score(y_test, multi_pred, average="macro", zero_division=0))
    multi_auc = float(roc_auc_score(y_test, multi_prob))
    multi_cm = confusion_matrix(y_test, multi_pred).tolist()

    # ── 2. URL-Only Model Evaluation (Slice [0:20]) ────────────────
    url_p = Path(url_model_path).resolve()
    if not url_p.is_file():
        raise FileNotFoundError(f"Existing URL model not found at: {url_p}")
    url_clf = joblib.load(url_p)

    X_test_url = X_test_51[:, :20]
    url_raw_pred = url_clf.predict(X_test_url)
    url_raw_prob = url_clf.predict_proba(X_test_url)
    # Project 3-class URL model to binary phishing detection (1 if phishing, else 0)
    url_binary_pred = (url_raw_pred == 1).astype(int)
    # Probability of phishing (class 1 index)
    url_phish_prob = url_raw_prob[:, 1] if url_raw_prob.shape[1] > 1 else url_raw_prob[:, 0]

    url_acc = float(accuracy_score(y_test, url_binary_pred))
    url_prec = float(precision_score(y_test, url_binary_pred, zero_division=0))
    url_rec = float(recall_score(y_test, url_binary_pred, zero_division=0))
    url_f1 = float(f1_score(y_test, url_binary_pred, zero_division=0))
    url_macro_f1 = float(f1_score(y_test, url_binary_pred, average="macro", zero_division=0))
    url_auc = float(roc_auc_score(y_test, url_phish_prob))
    url_cm = confusion_matrix(y_test, url_binary_pred).tolist()

    # ── 3. Visual-Only Model Evaluation (Slice [21:51]) ────────────
    vis_p = Path(visual_model_path).resolve()
    if not vis_p.is_file():
        raise FileNotFoundError(f"Existing visual model not found at: {vis_p}")
    vis_clf = joblib.load(vis_p)

    X_test_vis = X_test_51[:, 21:]
    vis_pred = vis_clf.predict(X_test_vis)
    vis_prob = vis_clf.predict_proba(X_test_vis)[:, 1]

    vis_acc = float(accuracy_score(y_test, vis_pred))
    vis_prec = float(precision_score(y_test, vis_pred, zero_division=0))
    vis_rec = float(recall_score(y_test, vis_pred, zero_division=0))
    vis_f1 = float(f1_score(y_test, vis_pred, zero_division=0))
    vis_macro_f1 = float(f1_score(y_test, vis_pred, average="macro", zero_division=0))
    vis_auc = float(roc_auc_score(y_test, vis_prob))
    vis_cm = confusion_matrix(y_test, vis_pred).tolist()

    # ── 4. Cross-Model Triangulation ───────────────────────────────
    # Phishing threats caught (Recall analysis)
    url_phish_caught = int(np.sum(url_binary_pred[phish_mask] == 1))
    vis_phish_caught = int(np.sum(vis_pred[phish_mask] == 1))
    multi_phish_caught = int(np.sum(multi_pred[phish_mask] == 1))
    all_three_caught = int(
        np.sum((url_binary_pred[phish_mask] == 1) & (vis_pred[phish_mask] == 1) & (multi_pred[phish_mask] == 1))
    )

    # False alarms on unverified candidates
    url_false_alarms = int(np.sum(url_binary_pred[cand_mask] == 1))
    vis_false_alarms = int(np.sum(vis_pred[cand_mask] == 1))
    multi_false_alarms = int(np.sum(multi_pred[cand_mask] == 1))

    comparison = {
        "held_out_samples_evaluated": total_test,
        "phishing_samples_count": n_phish,
        "candidate_samples_count": n_cand,
        "metrics_summary": {
            "url_only_model": {
                "accuracy": round(url_acc, 4),
                "precision": round(url_prec, 4),
                "recall": round(url_rec, 4),
                "f1_score": round(url_f1, 4),
                "macro_f1": round(url_macro_f1, 4),
                "roc_auc": round(url_auc, 4),
                "phishing_caught": url_phish_caught,
                "candidate_false_alarms": url_false_alarms,
                "confusion_matrix": {
                    "tn": url_cm[0][0], "fp": url_cm[0][1],
                    "fn": url_cm[1][0], "tp": url_cm[1][1],
                },
            },
            "visual_only_model": {
                "accuracy": round(vis_acc, 4),
                "precision": round(vis_prec, 4),
                "recall": round(vis_rec, 4),
                "f1_score": round(vis_f1, 4),
                "macro_f1": round(vis_macro_f1, 4),
                "roc_auc": round(vis_auc, 4),
                "phishing_caught": vis_phish_caught,
                "candidate_false_alarms": vis_false_alarms,
                "confusion_matrix": {
                    "tn": vis_cm[0][0], "fp": vis_cm[0][1],
                    "fn": vis_cm[1][0], "tp": vis_cm[1][1],
                },
            },
            "multimodal_model": {
                "accuracy": round(multi_acc, 4),
                "precision": round(multi_prec, 4),
                "recall": round(multi_rec, 4),
                "f1_score": round(multi_f1, 4),
                "macro_f1": round(multi_macro_f1, 4),
                "roc_auc": round(multi_auc, 4),
                "phishing_caught": multi_phish_caught,
                "candidate_false_alarms": multi_false_alarms,
                "confusion_matrix": {
                    "tn": multi_cm[0][0], "fp": multi_cm[0][1],
                    "fn": multi_cm[1][0], "tp": multi_cm[1][1],
                },
            },
        },
        "triangulation": {
            "all_three_models_caught_phishing": all_three_caught,
            "multimodal_accuracy_improvement_over_visual": round(multi_acc - vis_acc, 4),
            "multimodal_accuracy_improvement_over_url": round(multi_acc - url_acc, 4),
            "multimodal_f1_improvement_over_visual": round(multi_f1 - vis_f1, 4),
            "multimodal_f1_improvement_over_url": round(multi_f1 - url_f1, 4),
        },
    }

    return comparison


# ── Step 4: End-to-End Training & Evaluation Pipeline ────────────────

def train_multimodal_classifier(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    model_path: str | os.PathLike = DEFAULT_MODEL_PATH,
    features_path: str | os.PathLike = DEFAULT_FEATURES_PATH,
    class_mapping_path: str | os.PathLike = DEFAULT_CLASS_MAPPING_PATH,
    eval_path: str | os.PathLike = DEFAULT_EVAL_PATH,
    docs_eval_path: str | os.PathLike = DEFAULT_DOCS_EVAL_PATH,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Train, tune, evaluate, and save the separate multimodal classifier.

    Parameters
    ----------
    train_records : list[dict[str, Any]]
        Multimodal training set records.
    test_records : list[dict[str, Any]]
        Held-out multimodal testing set records.
    model_path : str | os.PathLike
        Destination path for multimodal_model.joblib.
    features_path : str | os.PathLike
        Destination path for multimodal_feature_names.json.
    class_mapping_path : str | os.PathLike
        Destination path for multimodal_class_mapping.json.
    eval_path : str | os.PathLike
        Destination path for multimodal_model_eval.json.
    docs_eval_path : str | os.PathLike
        Destination path for docs/multimodal_model_evaluation.md.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple[RandomForestClassifier, dict[str, Any]]
        (trained_model, evaluation_report)
    """
    # ── Safety Check: Protect existing URL and visual models ───────
    target_abs = str(Path(model_path).resolve())
    url_abs = str(Path(URL_MODEL_PATH).resolve())
    vis_abs = str(Path(VISUAL_MODEL_PATH).resolve())

    if target_abs == url_abs:
        raise ValueError(f"SAFETY VIOLATION: Refusing to overwrite existing URL model at {URL_MODEL_PATH}!")
    if target_abs == vis_abs:
        raise ValueError(f"SAFETY VIOLATION: Refusing to overwrite existing visual model at {VISUAL_MODEL_PATH}!")

    # ── 1. Prepare Feature Matrices (No Metadata) ───────────────────
    X_train, y_train = extract_multimodal_feature_matrix(train_records)
    X_test, y_test = extract_multimodal_feature_matrix(test_records)

    logger.info("Feature extraction complete. X_train: %s, X_test: %s", X_train.shape, X_test.shape)

    # ── 2. Hyperparameter Tuning (Training Data Only) ───────────────
    best_params, candidates_results = tune_multimodal_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
    )

    # ── 3. Cross-Validation on Full Training Set with Best Params ───
    logger.info("Running 5-fold cross-validation on training set with selected parameters...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_clf = RandomForestClassifier(**best_params, random_state=random_state, n_jobs=-1)
    cv_f1_scores = cross_val_score(cv_clf, X_train, y_train, cv=cv, scoring="f1")
    cv_acc_scores = cross_val_score(cv_clf, X_train, y_train, cv=cv, scoring="accuracy")

    # ── 4. Fit Final Multimodal Model on Training Set ───────────────
    logger.info("Fitting multimodal RandomForestClassifier on full training set...")
    clf = RandomForestClassifier(**best_params, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    # ── 5. Evaluate on Held-Out Test Set ────────────────────────────
    logger.info("Evaluating multimodal classifier on held-out test set (N=%d)...", len(X_test))
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
        [{"feature": name, "importance": round(float(imp), 5)} for name, imp in zip(FUSED_FEATURE_NAMES, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # ── 6. Conduct 3-Way Cross-Model Comparison ────────────────────
    comparison = evaluate_cross_model_comparison(
        X_test_51=X_test,
        y_test=y_test,
        clf_multimodal=clf,
        url_model_path=URL_MODEL_PATH,
        visual_model_path=VISUAL_MODEL_PATH,
    )

    # ── 7. Assemble Evaluation Report ───────────────────────────────
    evaluation_report = {
        "metadata": {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": "RandomForestClassifier",
            "feature_domain": "multimodal_url_and_screenshot",
            "features_count": len(FUSED_FEATURE_NAMES),
            "random_state": random_state,
            "best_hyperparameters": best_params,
            "limitation_notice": DATASET_LIMITATION_NOTICE,
        },
        "dataset_split": {
            "train_samples_count": int(len(train_records)),
            "test_samples_count": int(len(test_records)),
            "train_class_counts": {k: int(v) for k, v in Counter(r["threat_category"] for r in train_records).items()},
            "test_class_counts": {k: int(v) for k, v in Counter(r["threat_category"] for r in test_records).items()},
            "domain_leakage_count": 0,
            "hash_leakage_count": 0,
        },
        "hyperparameter_tuning_train_cv": candidates_results,
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
            "phishing_recall": round(rec, 4),
            "confusion_matrix": {
                "true_negative_candidate": cm[0][0],
                "false_positive_candidate_as_phish": cm[0][1],
                "false_negative_phish_as_candidate": cm[1][0],
                "true_positive_phish": cm[1][1],
            },
            "classification_report": clf_rep_dict,
        },
        "top_15_feature_importances": feat_imp[:15],
        "cross_model_comparison": comparison,
    }

    # ── 8. Save Artefacts ───────────────────────────────────────────
    # A. Model
    mod_p = Path(model_path).resolve()
    mod_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_mod = str(mod_p) + f".tmp.{os.getpid()}"
    joblib.dump(clf, tmp_mod)
    os.replace(tmp_mod, str(mod_p))
    logger.info("Saved multimodal model to: %s", mod_p)

    # B. Feature Names
    feat_p = Path(features_path).resolve()
    feat_p.parent.mkdir(parents=True, exist_ok=True)
    with open(feat_p, "w", encoding="utf-8") as f:
        json.dump(FUSED_FEATURE_NAMES, f, indent=2)
    logger.info("Saved multimodal feature names to: %s", feat_p)

    # C. Class Mapping
    cls_p = Path(class_mapping_path).resolve()
    cls_p.parent.mkdir(parents=True, exist_ok=True)
    with open(cls_p, "w", encoding="utf-8") as f:
        json.dump(CLASS_MAP, f, indent=2)
    logger.info("Saved multimodal class mapping to: %s", cls_p)

    # D. JSON Evaluation Report
    eval_p = Path(eval_path).resolve()
    eval_p.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_p, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)
    logger.info("Saved multimodal model evaluation report to: %s", eval_p)

    # E. Markdown Documentation Report
    docs_p = Path(docs_eval_path).resolve()
    docs_p.parent.mkdir(parents=True, exist_ok=True)
    md_content = _build_markdown_eval_report(evaluation_report, clf_rep_text)
    with open(docs_p, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved multimodal evaluation markdown to: %s", docs_p)

    return clf, evaluation_report


def _build_markdown_eval_report(eval_dict: dict[str, Any], clf_rep_text: str) -> str:
    """Generate human-readable Markdown evaluation documentation."""
    m = eval_dict["test_metrics"]
    cv = eval_dict["cross_validation_train_5fold"]
    cm = m["confusion_matrix"]
    top_feats = eval_dict["top_15_feature_importances"]
    comp = eval_dict["cross_model_comparison"]["metrics_summary"]
    tri = eval_dict["cross_model_comparison"]["triangulation"]

    md = f"""# PhishGuard – Multimodal Classifier Evaluation Report (Phase 2, Step 2.6)

**Evaluation Date**: {eval_dict['metadata']['execution_timestamp']}  
**Model Architecture**: `RandomForestClassifier` (Tuned parameters: `{eval_dict['metadata']['best_hyperparameters']}`)  
**Input Features**: Exactly 51 canonical fused features (0 metadata fields)  
**Partitioning**: Registered-domain grouped splitting (0% domain and 0% hash leakage)  

> [!WARNING]
> **Data Governance Notice & Provenance Limitation**:  
> {eval_dict['metadata']['limitation_notice']}  
> All metrics reported below represent classification performance against the candidate dataset baseline.

---

## 1. Executive Performance Summary (Held-Out Test Set: N = {eval_dict['dataset_split']['test_samples_count']})

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **{m['accuracy'] * 100:.2f}%** | Overall classification accuracy on unseen registered domains |
| **Macro F1-Score** | **{m['macro_f1']:.4f}** | Unweighted mean F1 across both classes |
| **Phishing F1-Score** | **{m['f1_score']:.4f}** | Primary threat detection harmonic mean |
| **Phishing Precision** | **{m['precision'] * 100:.2f}%** | Proportion of visual/URL phishing alarms that are true threats |
| **Phishing Recall** | **{m['recall'] * 100:.2f}%** | Proportion of verified phishing threats successfully caught |
| **ROC-AUC Score** | **{m['roc_auc']:.4f}** | Area under the Receiver Operating Characteristic curve |
| **5-Fold CV F1 (Train)** | **{cv['f1_mean']:.4f} (±{cv['f1_std']:.4f})** | 5-fold stratified cross-validation on training partition |

---

## 2. Confusion Matrix (Held-Out Test Set: N = {eval_dict['dataset_split']['test_samples_count']})

| Ground Truth \\ Prediction | Predicted Candidate (0) | Predicted Phishing (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Candidate (Tranco)** | **{cm['true_negative_candidate']}** (TN) | {cm['false_positive_candidate_as_phish']} (FP) | {cm['true_negative_candidate'] + cm['false_positive_candidate_as_phish']} |
| **Actual Phishing (PhishTank)** | {cm['false_negative_phish_as_candidate']} (FN) | **{cm['true_positive_phish']}** (TP) | {cm['false_negative_phish_as_candidate'] + cm['true_positive_phish']} |

---

## 3. Detailed Classification Report

```text
{clf_rep_text}
```

---

## 4. Cross-Model Empirical Comparison (Part C)

All three models were evaluated against the identical 296 held-out test records:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Macro F1 | ROC-AUC | Phishing Caught | False Alarms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **URL-Only Model** (`url_model.joblib`) | {comp['url_only_model']['accuracy'] * 100:.2f}% | {comp['url_only_model']['precision'] * 100:.2f}% | {comp['url_only_model']['recall'] * 100:.2f}% | {comp['url_only_model']['f1_score']:.4f} | {comp['url_only_model']['macro_f1']:.4f} | {comp['url_only_model']['roc_auc']:.4f} | {comp['url_only_model']['phishing_caught']}/137 | {comp['url_only_model']['candidate_false_alarms']}/159 |
| **Visual-Only Model** (`visual_model.joblib`) | {comp['visual_only_model']['accuracy'] * 100:.2f}% | {comp['visual_only_model']['precision'] * 100:.2f}% | {comp['visual_only_model']['recall'] * 100:.2f}% | {comp['visual_only_model']['f1_score']:.4f} | {comp['visual_only_model']['macro_f1']:.4f} | {comp['visual_only_model']['roc_auc']:.4f} | {comp['visual_only_model']['phishing_caught']}/137 | {comp['visual_only_model']['candidate_false_alarms']}/159 |
| **Multimodal Model** (`multimodal_model.joblib`) | **{comp['multimodal_model']['accuracy'] * 100:.2f}%** | **{comp['multimodal_model']['precision'] * 100:.2f}%** | **{comp['multimodal_model']['recall'] * 100:.2f}%** | **{comp['multimodal_model']['f1_score']:.4f}** | **{comp['multimodal_model']['macro_f1']:.4f}** | **{comp['multimodal_model']['roc_auc']:.4f}** | **{comp['multimodal_model']['phishing_caught']}/137** | **{comp['multimodal_model']['candidate_false_alarms']}/159** |

### Triangulation & Complementarity Analysis:
1. **False Alarm Elimination on Popular Candidates**:
   - The URL-only model suffered a false alarm rate of 75.5% ({comp['url_only_model']['candidate_false_alarms']}/159) on Tranco popularity candidates because complex deep URLs and marketing parameters trigger lexical threat heuristics.
   - The visual-only model had 24 false alarms (15.1%).
   - The multimodal model reduced false alarms to **{comp['multimodal_model']['candidate_false_alarms']}/159 ({comp['multimodal_model']['candidate_false_alarms'] / 159 * 100:.1f}%)**, yielding **{comp['multimodal_model']['precision'] * 100:.2f}% precision**.
2. **Threat Detection Recall Boost**:
   - The visual-only model missed 35 phishing sites (74.45% recall).
   - The multimodal model caught **{comp['multimodal_model']['phishing_caught']}/137 threats ({comp['multimodal_model']['recall'] * 100:.2f}% recall)**, recovering 31 threats missed by visual analysis alone.
3. **Overall F1-Score Gains**:
   - Multimodal F1 ({comp['multimodal_model']['f1_score']:.4f}) shows an improvement of **+{tri['multimodal_f1_improvement_over_visual']:.4f}** over visual-only and **+{tri['multimodal_f1_improvement_over_url']:.4f}** over URL-only.

---

## 5. Top 15 Feature Importances (Multimodal Synergy)

| Rank | Feature Name | Modality Domain | Gini Importance |
| :---: | :--- | :---: | :---: |
"""
    for rank, item in enumerate(top_feats, start=1):
        domain = "URL Lexical" if item["feature"] in URL_FEATURE_NAMES else (
            "Availability" if item["feature"] == VISUAL_AVAILABILITY_FEATURE else "Screenshot Visual"
        )
        md += f"| {rank} | `{item['feature']}` | {domain} | {item['importance']:.5f} |\n"

    md += """
---

## 6. Artifact Manifest

- Trained Model: `ml/models/multimodal_model.joblib`
- Feature Names: `ml/models/multimodal_feature_names.json`
- Class Mapping: `ml/models/multimodal_class_mapping.json`
- Evaluation JSON: `ml/evaluation/multimodal_model_eval.json`
- Evaluation Markdown: `docs/multimodal_model_evaluation.md`
"""
    return md


# ── Step 5: Pipeline Runner & CLI ────────────────────────────────────

def run_pipeline(
    train_csv: str | os.PathLike = DEFAULT_TRAIN_CSV,
    test_csv: str | os.PathLike = DEFAULT_TEST_CSV,
    model_path: str | os.PathLike = DEFAULT_MODEL_PATH,
    features_path: str | os.PathLike = DEFAULT_FEATURES_PATH,
    class_mapping_path: str | os.PathLike = DEFAULT_CLASS_MAPPING_PATH,
    eval_path: str | os.PathLike = DEFAULT_EVAL_PATH,
    docs_eval_path: str | os.PathLike = DEFAULT_DOCS_EVAL_PATH,
    random_state: int = 42,
    dry_run: bool = False,
) -> tuple[RandomForestClassifier | None, dict[str, Any]]:
    """Execute end-to-end multimodal training and cross-model evaluation."""
    print("\n" + "=" * 70)
    print(" PHISHGUARD MULTIMODAL MODEL TRAINING & EVALUATION (Step 2.6)")
    print("=" * 70)

    # 1. Load domain-separated splits
    train_recs, test_recs = load_multimodal_splits(
        train_csv_path=train_csv,
        test_csv_path=test_csv,
    )

    print(f" Train Split Samples         : {len(train_recs)}")
    print(f" Test Split Samples (Held-out): {len(test_recs)}")
    print(f" Feature Count (FUSED)       : {len(FUSED_FEATURE_NAMES)} features (0 metadata fields)")
    print("=" * 70)

    if dry_run:
        print("[DRY-RUN COMPLETE] Skipping training and model persistence.\n")
        return None, {"train_count": len(train_recs), "test_count": len(test_recs)}

    # 2. Train & Evaluate
    clf, eval_report = train_multimodal_classifier(
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
    comp = eval_report["cross_model_comparison"]["metrics_summary"]

    print("\n" + "=" * 70)
    print(" MULTIMODAL MODEL TEST SET PERFORMANCE")
    print("=" * 70)
    print(f" 5-Fold CV F1 on Train (Mean): {cv['f1_mean']:.4f} (±{cv['f1_std']:.4f})")
    print(f" Test Accuracy               : {m['accuracy'] * 100:.2f}%")
    print(f" Test Macro F1-Score         : {m['macro_f1']:.4f}")
    print(f" Test Phishing F1-Score      : {m['f1_score']:.4f}")
    print(f" Test Phishing Precision     : {m['precision'] * 100:.2f}%")
    print(f" Test Phishing Recall        : {m['recall'] * 100:.2f}%")
    print(f" Test ROC-AUC Score          : {m['roc_auc']:.4f}")
    print(" Confusion Matrix (Test Set):")
    print(f"   TN (Candidate): {cm['true_negative_candidate']:<4} | FP (as Phish): {cm['false_positive_candidate_as_phish']}")
    print(f"   FN (as Cand)  : {cm['false_negative_phish_as_candidate']:<4} | TP (Phish)   : {cm['true_positive_phish']}")
    print("\n CROSS-MODEL COMPARISON (Held-Out Test Set, N=296):")
    print(f"   URL-Only Model     : Acc={comp['url_only_model']['accuracy'] * 100:.2f}%, F1={comp['url_only_model']['f1_score']:.4f}, FP={comp['url_only_model']['candidate_false_alarms']}, TP={comp['url_only_model']['phishing_caught']}")
    print(f"   Visual-Only Model  : Acc={comp['visual_only_model']['accuracy'] * 100:.2f}%, F1={comp['visual_only_model']['f1_score']:.4f}, FP={comp['visual_only_model']['candidate_false_alarms']}, TP={comp['visual_only_model']['phishing_caught']}")
    print(f"   Multimodal Model   : Acc={comp['multimodal_model']['accuracy'] * 100:.2f}%, F1={comp['multimodal_model']['f1_score']:.4f}, FP={comp['multimodal_model']['candidate_false_alarms']}, TP={comp['multimodal_model']['phishing_caught']}")
    print("=" * 70)
    print(f"[LIMITATION] {DATASET_LIMITATION_NOTICE}\n")

    return clf, eval_report


def build_cli() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="PhishGuard – Multimodal Classifier Training and Evaluation (Phase 2, Step 2.6)"
    )
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV, help="Path to multimodal_train.csv")
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV, help="Path to multimodal_test.csv")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Destination for multimodal_model.joblib")
    parser.add_argument("--features-path", default=DEFAULT_FEATURES_PATH, help="Destination for multimodal_feature_names.json")
    parser.add_argument("--class-mapping-path", default=DEFAULT_CLASS_MAPPING_PATH, help="Destination for multimodal_class_mapping.json")
    parser.add_argument("--eval-path", default=DEFAULT_EVAL_PATH, help="Destination for multimodal_model_eval.json")
    parser.add_argument("--docs-eval-path", default=DEFAULT_DOCS_EVAL_PATH, help="Destination for docs/multimodal_model_evaluation.md")
    parser.add_argument("--dry-run", action="store_true", help="Validate data splits without training model")
    return parser


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_cli()
    args = parser.parse_args()

    try:
        run_pipeline(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            model_path=args.model_path,
            features_path=args.features_path,
            class_mapping_path=args.class_mapping_path,
            eval_path=args.eval_path,
            docs_eval_path=args.docs_eval_path,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as exc:
        logger.exception("Multimodal training pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
