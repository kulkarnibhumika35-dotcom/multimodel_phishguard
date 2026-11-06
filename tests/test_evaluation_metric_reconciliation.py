"""
Regression tests for Evaluation Metric Reconciliation (Phase 2, Step 2.7.1).

Verifies metric reproducibility, ROC-AUC score calculations, class mapping
integrity, metadata isolation, and documentation artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ml.training.train_multimodal_model import (
    load_multimodal_splits,
    extract_multimodal_feature_matrix,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
)
from ml.features.feature_fusion import FUSED_FEATURE_NAMES, URL_FEATURE_NAMES, VISUAL_FEATURE_NAMES

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def evaluation_artifacts() -> dict:
    """Load test records, feature matrix, and models once for the test module."""
    train_recs, test_recs = load_multimodal_splits()
    X_test, y_test = extract_multimodal_feature_matrix(test_recs)

    url_clf = joblib.load(ROOT_DIR / "ml" / "models" / "url_model.joblib")
    vis_clf = joblib.load(ROOT_DIR / "ml" / "models" / "visual_model.joblib")
    multi_clf = joblib.load(ROOT_DIR / "ml" / "models" / "multimodal_model.joblib")

    return {
        "train_recs": train_recs,
        "test_recs": test_recs,
        "X_test": X_test,
        "y_test": y_test,
        "url_clf": url_clf,
        "vis_clf": vis_clf,
        "multi_clf": multi_clf,
    }


def test_saved_eval_json_metric_consistency():
    """Verify that stored evaluation JSON files report exactly 0.4664 and 0.8491 for ROC-AUC."""
    multi_eval_path = ROOT_DIR / "ml" / "evaluation" / "multimodal_model_eval.json"
    audit_eval_path = ROOT_DIR / "ml" / "evaluation" / "multimodal_robustness_audit.json"

    assert multi_eval_path.is_file(), "multimodal_model_eval.json must exist"
    assert audit_eval_path.is_file(), "multimodal_robustness_audit.json must exist"

    with open(multi_eval_path, "r", encoding="utf-8") as f:
        multi_eval = json.load(f)
    with open(audit_eval_path, "r", encoding="utf-8") as f:
        audit_eval = json.load(f)

    # Check Step 2.6 cross-model comparison
    url_metrics_26 = multi_eval["cross_model_comparison"]["metrics_summary"]["url_only_model"]
    vis_metrics_26 = multi_eval["cross_model_comparison"]["metrics_summary"]["visual_only_model"]
    assert url_metrics_26["roc_auc"] == 0.4664
    assert vis_metrics_26["roc_auc"] == 0.8491

    # Check Step 2.7 audit report
    standalone_models = audit_eval["feature_ablation_audit"]["standalone_single_modality_models"]
    assert standalone_models["standalone_url_model"]["roc_auc"] == 0.4664
    assert standalone_models["standalone_visual_model"]["roc_auc"] == 0.8491


def test_recalculated_standalone_url_metrics(evaluation_artifacts):
    """Verify reproducible calculation of standalone URL model metrics on held-out test set."""
    X_test = evaluation_artifacts["X_test"]
    y_test = evaluation_artifacts["y_test"]
    url_clf = evaluation_artifacts["url_clf"]

    X_url = X_test[:, :20]
    raw_pred = url_clf.predict(X_url)
    bin_pred = (raw_pred == 1).astype(int)
    raw_prob = url_clf.predict_proba(X_url)
    phish_prob = raw_prob[:, 1]

    acc = accuracy_score(y_test, bin_pred)
    rec = recall_score(y_test, bin_pred)
    auc = roc_auc_score(y_test, phish_prob)

    assert round(acc, 4) == 0.5946
    assert round(rec, 4) == 1.0000
    assert round(auc, 4) == 0.4664
    # Verify high candidate false positive rate
    cand_fp = np.sum((y_test == 0) & (bin_pred == 1))
    assert cand_fp == 120


def test_recalculated_standalone_visual_metrics(evaluation_artifacts):
    """Verify reproducible calculation of standalone visual model metrics on held-out test set."""
    X_test = evaluation_artifacts["X_test"]
    y_test = evaluation_artifacts["y_test"]
    vis_clf = evaluation_artifacts["vis_clf"]

    X_vis = X_test[:, 21:]
    pred = vis_clf.predict(X_vis)
    prob = vis_clf.predict_proba(X_vis)[:, 1]

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    assert round(acc, 4) == 0.8007
    assert round(prec, 4) == 0.8095
    assert round(rec, 4) == 0.7445
    assert round(auc, 4) == 0.8491


def test_recalculated_multimodal_metrics(evaluation_artifacts):
    """Verify full multimodal model performance metrics."""
    X_test = evaluation_artifacts["X_test"]
    y_test = evaluation_artifacts["y_test"]
    multi_clf = evaluation_artifacts["multi_clf"]

    pred = multi_clf.predict(X_test)
    prob = multi_clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    assert round(acc, 4) == 0.9865
    assert round(prec, 4) == 1.0000
    assert round(rec, 4) == 0.9708
    assert round(f1, 4) == 0.9852
    assert round(auc, 4) == 0.9987


def test_true_url_ablation_performance(evaluation_artifacts):
    """Verify that true URL-only ablation of the multimodal model achieves 96.62% accuracy."""
    X_test = evaluation_artifacts["X_test"]
    y_test = evaluation_artifacts["y_test"]
    multi_clf = evaluation_artifacts["multi_clf"]

    # Mask visual features to 0.0 and availability flag to 0.0
    X_abl_url = X_test.copy()
    X_abl_url[:, 20] = 0.0
    X_abl_url[:, 21:] = 0.0

    pred = multi_clf.predict(X_abl_url)
    prob = multi_clf.predict_proba(X_abl_url)[:, 1]

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    assert round(acc, 4) == 0.9662
    assert round(prec, 4) == 0.9704
    assert round(rec, 4) == 0.9562
    assert round(auc, 4) == 0.9944


def test_positive_class_definition_consistency(evaluation_artifacts):
    """Verify that class 1 is consistently encoded as phishing across models."""
    multi_clf = evaluation_artifacts["multi_clf"]
    vis_clf = evaluation_artifacts["vis_clf"]
    url_clf = evaluation_artifacts["url_clf"]

    assert list(multi_clf.classes_) == [0, 1]
    assert list(vis_clf.classes_) == [0, 1]
    assert list(url_clf.classes_) == [0, 1, 2]

    # Verify class mapping files
    with open(ROOT_DIR / "ml" / "models" / "multimodal_class_mapping.json", "r", encoding="utf-8") as f:
        multi_map = json.load(f)
    assert multi_map["1"] == "phishing"
    assert multi_map["0"] == "unverified_candidate"

    with open(ROOT_DIR / "ml" / "models" / "visual_class_mapping.json", "r", encoding="utf-8") as f:
        vis_map = json.load(f)
    assert vis_map["1"] == "phishing"
    assert vis_map["0"] == "unverified_candidate"

    with open(ROOT_DIR / "ml" / "models" / "class_mapping.json", "r", encoding="utf-8") as f:
        url_map = json.load(f)
    assert url_map["class_to_name"]["1"] == "phishing"
    assert url_map["class_to_name"]["0"] == "legitimate"
    assert url_map["class_to_name"]["2"] == "malware_distribution"


def test_metadata_exclusion_from_multimodal_features():
    """Verify that csv_line_num and other metadata columns are excluded from features."""
    feat_names_path = ROOT_DIR / "ml" / "models" / "multimodal_feature_names.json"
    with open(feat_names_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    assert len(features) == 51
    assert "csv_line_num" not in features
    assert "original_url" not in features
    assert "registered_domain" not in features
    assert "label" not in features
    assert "binary_target" not in features


def test_reconciliation_documentation_exists():
    """Verify that reconciliation report markdown file exists and has substantive sections."""
    doc_path = ROOT_DIR / "docs" / "evaluation_metric_reconciliation.md"
    assert doc_path.is_file(), "docs/evaluation_metric_reconciliation.md must exist"

    content = doc_path.read_text(encoding="utf-8")
    assert "0.4664" in content
    assert "0.8491" in content
    assert "0.8988" in content
    assert "0.8872" in content
    assert "96.62%" in content
    assert "59.46%" in content
    assert "Pearson" in content

