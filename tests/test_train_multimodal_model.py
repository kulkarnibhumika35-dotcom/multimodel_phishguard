"""
Unit tests for PhishGuard Multimodal Classifier Training and Evaluation
(ml/training/train_multimodal_model.py, Phase 2 Step 2.6).
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features.feature_fusion import FUSED_FEATURE_NAMES, VISUAL_AVAILABILITY_FEATURE
from ml.features.url_features import FEATURE_NAMES as URL_FEATURE_NAMES
from ml.features.visual_features import VISUAL_FEATURE_NAMES
from ml.training.train_multimodal_model import (
    CLASS_MAP,
    CLASS_PHISHING,
    CLASS_UNVERIFIED_CANDIDATE,
    DATASET_LIMITATION_NOTICE,
    DEFAULT_CLASS_MAPPING_PATH,
    DEFAULT_DOCS_EVAL_PATH,
    DEFAULT_EVAL_PATH,
    DEFAULT_FEATURES_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    METADATA_COLUMNS,
    URL_MODEL_PATH,
    VISUAL_MODEL_PATH,
    evaluate_cross_model_comparison,
    extract_multimodal_feature_matrix,
    load_multimodal_splits,
    train_multimodal_classifier,
    tune_multimodal_hyperparameters,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_multimodal_records():
    """Create a minimal synthetic multimodal records list with 51 features and metadata."""
    rng = np.random.RandomState(42)
    records = []

    # 10 Phishing records
    for i in range(10):
        r = {
            "csv_line_num": str(i + 1),
            "original_url": f"https://phish-example-{i}.net/login?token=abc",
            "normalized_url": f"https://phish-example-{i}.net/login?token=abc",
            "registered_domain": f"phish-example-{i}.net",
            "original_label": "phishing",
            "threat_category": "phishing",
            "label_provenance": "phishtank_verified",
            "screenshot_path": f"datasets/screenshots/raw/p_{i}.png",
            "screenshot_sha256": f"p_sha_{i:04d}",
        }
        # Add 51 features
        for fn in FUSED_FEATURE_NAMES:
            if fn == VISUAL_AVAILABILITY_FEATURE:
                r[fn] = "1"
            elif fn in ("width", "height"):
                r[fn] = "1920" if fn == "width" else "1080"
            else:
                r[fn] = str(round(float(rng.uniform(1.0, 50.0)), 4))
        records.append(r)

    # 10 Unverified candidate records
    for i in range(10):
        r = {
            "csv_line_num": str(i + 11),
            "original_url": f"https://cand-example-{i}.org",
            "normalized_url": f"https://cand-example-{i}.org",
            "registered_domain": f"cand-example-{i}.org",
            "original_label": "legitimate",
            "threat_category": "unverified_candidate",
            "label_provenance": "tranco_candidate_unverified",
            "screenshot_path": f"datasets/screenshots/raw/c_{i}.png",
            "screenshot_sha256": f"c_sha_{i:04d}",
        }
        for fn in FUSED_FEATURE_NAMES:
            if fn == VISUAL_AVAILABILITY_FEATURE:
                r[fn] = "1"
            elif fn in ("width", "height"):
                r[fn] = "1920" if fn == "width" else "1080"
            else:
                r[fn] = str(round(float(rng.uniform(50.0, 100.0)), 4))
        records.append(r)

    return records


# ── 1. 51-Feature Input Validation & Slices ──────────────────────────

class TestFeatureValidation:
    def test_fused_feature_count(self):
        """Assert exactly 51 fused features."""
        assert len(FUSED_FEATURE_NAMES) == 51

    def test_feature_canonical_slices(self):
        """Verify URL features [0:20], availability flag [20], and visual features [21:51]."""
        assert FUSED_FEATURE_NAMES[:20] == list(URL_FEATURE_NAMES)
        assert FUSED_FEATURE_NAMES[20] == VISUAL_AVAILABILITY_FEATURE
        assert FUSED_FEATURE_NAMES[21:] == list(VISUAL_FEATURE_NAMES)

    def test_extract_feature_matrix_dimensions(self, synthetic_multimodal_records):
        """Verify feature matrix has shape (N, 51) and labels have shape (N,)."""
        X, y = extract_multimodal_feature_matrix(synthetic_multimodal_records)
        assert X.shape == (20, 51)
        assert y.shape == (20,)
        assert X.dtype == np.float64
        assert y.dtype == np.int64
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()

    def test_extract_feature_matrix_rejects_missing_feature(self, synthetic_multimodal_records):
        """Verify extraction raises ValueError when any canonical feature is missing."""
        del synthetic_multimodal_records[0]["url_entropy"]
        with pytest.raises(ValueError, match="Missing feature value for 'url_entropy'"):
            extract_multimodal_feature_matrix(synthetic_multimodal_records)

    def test_extract_feature_matrix_rejects_nan(self, synthetic_multimodal_records):
        """Verify extraction raises ValueError on NaN values."""
        synthetic_multimodal_records[0]["mean_brightness"] = "nan"
        with pytest.raises(ValueError, match="Non-finite feature value"):
            extract_multimodal_feature_matrix(synthetic_multimodal_records)


# ── 2. Metadata Exclusion Tests ──────────────────────────────────────

class TestMetadataExclusion:
    def test_metadata_fields_excluded_from_features(self, synthetic_multimodal_records):
        """Ensure none of the 9 metadata columns are included in the feature matrix."""
        X, _ = extract_multimodal_feature_matrix(synthetic_multimodal_records)

        # Check that X only has 51 columns (which correspond to FUSED_FEATURE_NAMES)
        assert X.shape[1] == len(FUSED_FEATURE_NAMES)

        # Verify none of the metadata column names are in FUSED_FEATURE_NAMES
        for col in METADATA_COLUMNS:
            assert col not in FUSED_FEATURE_NAMES


# ── 3. Domain-Separated Partitioning & Zero Leakage ──────────────────

class TestDomainSeparation:
    def test_zero_leakage_between_train_and_test(self):
        """Verify zero domain and zero hash overlap in the actual dataset files."""
        train_recs, test_recs = load_multimodal_splits(DEFAULT_TRAIN_CSV, DEFAULT_TEST_CSV)
        assert len(train_recs) == 1190
        assert len(test_recs) == 296

        train_domains = {r["registered_domain"] for r in train_recs}
        test_domains = {r["registered_domain"] for r in test_recs}
        assert train_domains.isdisjoint(test_domains)

        train_hashes = {r["screenshot_sha256"] for r in train_recs}
        test_hashes = {r["screenshot_sha256"] for r in test_recs}
        assert train_hashes.isdisjoint(test_hashes)


# ── 4. Safety Checks: Protection of Existing Models ──────────────────

class TestSafetyProtections:
    def test_safety_check_refuses_overwriting_url_model(self, synthetic_multimodal_records):
        """Verify train_multimodal_classifier refuses to overwrite url_model.joblib."""
        with pytest.raises(ValueError, match="Refusing to overwrite existing URL model"):
            train_multimodal_classifier(
                train_records=synthetic_multimodal_records[:16],
                test_records=synthetic_multimodal_records[16:],
                model_path=URL_MODEL_PATH,
            )

    def test_safety_check_refuses_overwriting_visual_model(self, synthetic_multimodal_records):
        """Verify train_multimodal_classifier refuses to overwrite visual_model.joblib."""
        with pytest.raises(ValueError, match="Refusing to overwrite existing visual model"):
            train_multimodal_classifier(
                train_records=synthetic_multimodal_records[:16],
                test_records=synthetic_multimodal_records[16:],
                model_path=VISUAL_MODEL_PATH,
            )


# ── 5. Hyperparameter Tuning on Training Data Only ───────────────────

class TestHyperparameterTuning:
    def test_tune_hyperparameters_reproducibility(self, synthetic_multimodal_records):
        """Verify hyperparameter tuning with fixed random_state produces identical best_params."""
        X, y = extract_multimodal_feature_matrix(synthetic_multimodal_records)
        custom_grid = [
            {"n_estimators": 5, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 10, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "log2"},
        ]
        best1, cands1 = tune_multimodal_hyperparameters(X, y, candidate_grid=custom_grid, n_jobs=1, random_state=42)
        best2, cands2 = tune_multimodal_hyperparameters(X, y, candidate_grid=custom_grid, n_jobs=1, random_state=42)

        assert best1 == best2
        assert len(cands1) == 2
        assert len(cands2) == 2
        assert cands1[0]["cv_f1_mean"] == cands2[0]["cv_f1_mean"]


# ── 6. Model Persistence and Loading ─────────────────────────────────

class TestModelPersistenceAndLoading:
    @pytest.mark.skipif(not os.path.exists(DEFAULT_MODEL_PATH), reason="multimodal_model.joblib not yet created")
    def test_loaded_model_interface_and_features(self):
        """Verify loaded multimodal model conforms to 51 features and correct classes."""
        clf = joblib.load(DEFAULT_MODEL_PATH)
        assert hasattr(clf, "predict")
        assert hasattr(clf, "predict_proba")
        assert clf.n_features_in_ == 51
        assert list(clf.classes_) == [0, 1]

        # Verify inference on dummy 51-feature vector
        dummy = np.zeros((2, 51), dtype=np.float64)
        preds = clf.predict(dummy)
        probas = clf.predict_proba(dummy)

        assert preds.shape == (2,)
        assert set(preds).issubset({0, 1})
        assert probas.shape == (2, 2)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.skipif(not os.path.exists(DEFAULT_FEATURES_PATH), reason="multimodal_feature_names.json not yet created")
    def test_feature_names_json_alignment(self):
        """Verify persisted feature names JSON exactly matches FUSED_FEATURE_NAMES."""
        with open(DEFAULT_FEATURES_PATH, "r", encoding="utf-8") as f:
            names = json.load(f)
        assert names == FUSED_FEATURE_NAMES

    @pytest.mark.skipif(not os.path.exists(DEFAULT_CLASS_MAPPING_PATH), reason="multimodal_class_mapping.json not yet created")
    def test_class_mapping_json_alignment(self):
        """Verify persisted class mapping JSON maps 0 to unverified_candidate and 1 to phishing."""
        with open(DEFAULT_CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            cmap = json.load(f)
        assert cmap["0"] == "unverified_candidate"
        assert cmap["1"] == "phishing"


# ── 7. Evaluation Report & Cross-Model Comparison ────────────────────

class TestEvaluationArtifacts:
    @pytest.mark.skipif(not os.path.exists(DEFAULT_EVAL_PATH), reason="multimodal_model_eval.json not yet created")
    def test_evaluation_json_structure_and_metrics(self):
        """Verify evaluation JSON contains all required metrics, CV results, and comparisons."""
        with open(DEFAULT_EVAL_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        assert "metadata" in eval_data
        assert "dataset_split" in eval_data
        assert "cross_validation_train_5fold" in eval_data
        assert "test_metrics" in eval_data
        assert "cross_model_comparison" in eval_data
        assert "top_15_feature_importances" in eval_data

        m = eval_data["test_metrics"]
        assert m["accuracy"] > 0.90
        assert m["precision"] > 0.90
        assert m["recall"] > 0.90
        assert m["f1_score"] > 0.90
        assert m["roc_auc"] > 0.95

        # Check cross-model comparison section
        comp = eval_data["cross_model_comparison"]
        assert "url_only_model" in comp["metrics_summary"]
        assert "visual_only_model" in comp["metrics_summary"]
        assert "multimodal_model" in comp["metrics_summary"]
        assert "triangulation" in comp

    @pytest.mark.skipif(not os.path.exists(DEFAULT_DOCS_EVAL_PATH), reason="multimodal_model_evaluation.md not yet created")
    def test_evaluation_markdown_sections(self):
        """Verify markdown report exists and contains required sections."""
        with open(DEFAULT_DOCS_EVAL_PATH, "r", encoding="utf-8") as f:
            md = f.read()

        assert "# PhishGuard – Multimodal Classifier Evaluation Report" in md
        assert "Executive Performance Summary" in md
        assert "Confusion Matrix" in md
        assert "Detailed Classification Report" in md
        assert "Cross-Model Empirical Comparison (Part C)" in md
        assert "Top 15 Feature Importances" in md
        assert "Artifact Manifest" in md
        assert "CRITICAL LIMITATION: Legitimate class consists of Tranco candidates" in md
