"""
Unit Tests for PhishGuard Visual Classifier Training Pipeline
(ml/training/train_visual_model.py)
"""

import csv
import json
import math
import os
import joblib
import numpy as np
import pytest

from ml.features.visual_features import VISUAL_FEATURE_NAMES
from ml.training.train_visual_model import (
    CLASS_MAP,
    CLASS_PHISHING,
    CLASS_UNVERIFIED_CANDIDATE,
    CSV_FIELDNAMES,
    DATASET_LIMITATION_NOTICE,
    URL_MODEL_PATH,
    extract_feature_matrix,
    run_pipeline,
    split_visual_dataset,
    train_visual_classifier,
)


@pytest.fixture
def synthetic_visual_csv(tmp_path):
    """Create a synthetic visual_features.csv with known labels and 30 numerical features."""
    csv_path = tmp_path / "synthetic_visual_features.csv"
    rng = np.random.RandomState(42)

    rows = []
    # 20 phishing, 20 unverified candidates
    for i in range(20):
        # Phishing sample
        p_row = {
            "csv_line_num": i + 1,
            "original_url": f"https://phish-domain-{i}.com/login",
            "normalized_url": f"https://phish-domain-{i}.com/login",
            "registered_domain": f"phish-domain-{i}.com",
            "original_label": "phishing",
            "threat_category": "phishing",
            "label_provenance": "phishtank_verified",
            "screenshot_path": f"datasets/screenshots/raw/phish_{i}.png",
            "screenshot_sha256": f"phish_sha_{i:04d}",
        }
        # Add 30 visual features
        for feat in VISUAL_FEATURE_NAMES:
            p_row[feat] = round(float(rng.uniform(10.0, 50.0)), 4)
        rows.append(p_row)

        # Legitimate candidate sample
        l_row = {
            "csv_line_num": i + 21,
            "original_url": f"https://legit-domain-{i}.org",
            "normalized_url": f"https://legit-domain-{i}.org",
            "registered_domain": f"legit-domain-{i}.org",
            "original_label": "legitimate",
            "threat_category": "unverified_candidate",
            "label_provenance": "tranco_candidate_unverified",
            "screenshot_path": f"datasets/screenshots/raw/legit_{i}.png",
            "screenshot_sha256": f"legit_sha_{i:04d}",
        }
        for feat in VISUAL_FEATURE_NAMES:
            l_row[feat] = round(float(rng.uniform(60.0, 100.0)), 4)
        rows.append(l_row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path, rows


def test_split_visual_dataset_domain_separation(tmp_path, synthetic_visual_csv):
    """Verify zero domain leakage and zero hash leakage between train and test sets."""
    csv_path, _ = synthetic_visual_csv
    train_path = tmp_path / "visual_train.csv"
    test_path = tmp_path / "visual_test.csv"

    train_recs, test_recs, summary = split_visual_dataset(
        input_csv_path=csv_path,
        train_csv_path=train_path,
        test_csv_path=test_path,
        test_size=0.25,
        random_state=42,
    )

    assert len(train_recs) + len(test_recs) == 40
    assert summary["domain_overlap_count"] == 0
    assert summary["hash_overlap_count"] == 0

    train_domains = {r["registered_domain"] for r in train_recs}
    test_domains = {r["registered_domain"] for r in test_recs}
    assert train_domains.isdisjoint(test_domains)

    train_hashes = {r["screenshot_sha256"] for r in train_recs}
    test_hashes = {r["screenshot_sha256"] for r in test_recs}
    assert train_hashes.isdisjoint(test_hashes)

    # Verify output files exist and preserve all 39 columns
    assert os.path.isfile(train_path)
    assert os.path.isfile(test_path)

    with open(train_path, "r", encoding="utf-8") as f:
        r_train = list(csv.DictReader(f))
    assert len(r_train) == len(train_recs)
    assert set(r_train[0].keys()) == set(CSV_FIELDNAMES)


def test_extract_feature_matrix_dimensionality_and_metadata_exclusion(synthetic_visual_csv):
    """Verify feature matrix contains strictly 30 canonical features and excludes all metadata."""
    _, rows = synthetic_visual_csv
    X, y = extract_feature_matrix(rows)

    # Exactly 30 features
    assert X.shape == (40, 30)
    assert y.shape == (40,)

    # Labels map correctly: 0 for candidate, 1 for phishing
    assert set(np.unique(y)) == {CLASS_UNVERIFIED_CANDIDATE, CLASS_PHISHING}
    assert np.count_nonzero(y == CLASS_PHISHING) == 20
    assert np.count_nonzero(y == CLASS_UNVERIFIED_CANDIDATE) == 20

    # No NaN or Inf
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_safety_check_refuses_overwriting_url_model(tmp_path, synthetic_visual_csv):
    """Verify safety check rejects any attempt to overwrite url_model.joblib."""
    _, rows = synthetic_visual_csv
    with pytest.raises(ValueError, match="SAFETY VIOLATION"):
        train_visual_classifier(
            train_records=rows[:30],
            test_records=rows[30:],
            model_path=URL_MODEL_PATH,
        )


def test_train_visual_classifier_persistence_and_evaluation(tmp_path, synthetic_visual_csv):
    """Verify visual model training, persistence, and complete evaluation report generation."""
    _, rows = synthetic_visual_csv
    train_recs = rows[:30]
    test_recs = rows[30:]

    model_path = tmp_path / "visual_model.joblib"
    features_path = tmp_path / "visual_feature_names.json"
    class_mapping_path = tmp_path / "visual_class_mapping.json"
    eval_path = tmp_path / "visual_model_eval.json"
    docs_eval_path = tmp_path / "visual_model_evaluation.md"

    clf, report = train_visual_classifier(
        train_records=train_recs,
        test_records=test_recs,
        model_path=model_path,
        features_path=features_path,
        class_mapping_path=class_mapping_path,
        eval_path=eval_path,
        docs_eval_path=docs_eval_path,
        n_estimators=20,
        random_state=42,
    )

    # 1. Model file persisted and loadable
    assert os.path.isfile(model_path)
    loaded_clf = joblib.load(model_path)
    assert hasattr(loaded_clf, "predict")
    preds = loaded_clf.predict(np.zeros((1, 30), dtype=np.float32))
    assert preds.shape == (1,)

    # 2. Features JSON
    assert os.path.isfile(features_path)
    with open(features_path, "r", encoding="utf-8") as f:
        saved_feats = json.load(f)
    assert saved_feats == VISUAL_FEATURE_NAMES

    # 3. Class mapping JSON
    assert os.path.isfile(class_mapping_path)
    with open(class_mapping_path, "r", encoding="utf-8") as f:
        saved_cls = json.load(f)
    assert saved_cls["0"] == "unverified_candidate"
    assert saved_cls["1"] == "phishing"

    # 4. JSON Evaluation Report
    assert os.path.isfile(eval_path)
    assert "test_metrics" in report
    assert "accuracy" in report["test_metrics"]
    assert "precision" in report["test_metrics"]
    assert "recall" in report["test_metrics"]
    assert "f1_score" in report["test_metrics"]
    assert "macro_f1" in report["test_metrics"]
    assert "roc_auc" in report["test_metrics"]
    assert "confusion_matrix" in report["test_metrics"]
    assert "cross_validation_train_5fold" in report
    assert "top_10_feature_importances" in report
    assert report["metadata"]["limitation_notice"] == DATASET_LIMITATION_NOTICE

    # 5. Markdown documentation report
    assert os.path.isfile(docs_eval_path)
    md_text = docs_eval_path.read_text(encoding="utf-8")
    assert "Visual Classifier Evaluation Report" in md_text
    assert "CRITICAL LIMITATION" in md_text


def test_reproducibility(synthetic_visual_csv, tmp_path):
    """Verify that identical random seeds produce identical splits and identical predictions."""
    csv_path, _ = synthetic_visual_csv

    # Run 1
    t1, te1, _ = split_visual_dataset(
        input_csv_path=csv_path,
        train_csv_path=tmp_path / "tr1.csv",
        test_csv_path=tmp_path / "te1.csv",
        random_state=42,
    )
    clf1, _ = train_visual_classifier(
        train_records=t1,
        test_records=te1,
        model_path=tmp_path / "m1.joblib",
        features_path=tmp_path / "f1.json",
        class_mapping_path=tmp_path / "c1.json",
        eval_path=tmp_path / "e1.json",
        docs_eval_path=tmp_path / "d1.md",
        n_estimators=20,
        random_state=42,
    )

    # Run 2
    t2, te2, _ = split_visual_dataset(
        input_csv_path=csv_path,
        train_csv_path=tmp_path / "tr2.csv",
        test_csv_path=tmp_path / "te2.csv",
        random_state=42,
    )
    clf2, _ = train_visual_classifier(
        train_records=t2,
        test_records=te2,
        model_path=tmp_path / "m2.joblib",
        features_path=tmp_path / "f2.json",
        class_mapping_path=tmp_path / "c2.json",
        eval_path=tmp_path / "e2.json",
        docs_eval_path=tmp_path / "d2.md",
        n_estimators=20,
        random_state=42,
    )

    # Verify identical records
    assert [r["registered_domain"] for r in t1] == [r["registered_domain"] for r in t2]
    assert [r["registered_domain"] for r in te1] == [r["registered_domain"] for r in te2]

    # Verify identical predictions on test set
    X_test, _ = extract_feature_matrix(te1)
    np.testing.assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))
    np.testing.assert_allclose(clf1.predict_proba(X_test), clf2.predict_proba(X_test))


def test_dry_run_pipeline(tmp_path, synthetic_visual_csv):
    """Verify run_pipeline in dry-run mode does not write model files."""
    csv_path, _ = synthetic_visual_csv
    model_path = tmp_path / "no_model.joblib"
    eval_path = tmp_path / "no_eval.json"

    clf, summary = run_pipeline(
        input_csv=csv_path,
        train_csv=tmp_path / "dry_train.csv",
        test_csv=tmp_path / "dry_test.csv",
        model_path=model_path,
        eval_path=eval_path,
        dry_run=True,
    )

    assert clf is None
    assert summary["total_samples"] == 40
    assert not os.path.exists(model_path)
    assert not os.path.exists(eval_path)

