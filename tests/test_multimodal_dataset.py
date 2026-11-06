"""
Unit tests for PhishGuard multimodal dataset preparation and compatibility audit
(Phase 2, Step 2.5).
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

from datasets.scripts.prepare_multimodal_dataset import (
    DEFAULT_AUDIT_JSON_PATH,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_REPORT_MD_PATH,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    METADATA_COLUMNS,
    MULTIMODAL_FIELDNAMES,
    PROVENANCE_LIMITATION_NOTICE,
    audit_multimodal_compatibility,
    build_multimodal_record,
    prepare_multimodal_dataset,
)
from ml.features.feature_fusion import (
    FUSED_FEATURE_NAMES,
    VISUAL_AVAILABILITY_FEATURE,
    fuse_features,
    fuse_feature_vector,
)
from ml.features.url_features import FEATURE_NAMES as URL_FEATURE_NAMES
from ml.features.visual_features import VISUAL_FEATURE_NAMES


# ── Fixtures & Mock Data ─────────────────────────────────────────────

@pytest.fixture
def sample_visual_record() -> dict[str, str]:
    """Sample valid record mimicking a row from datasets/processed/visual_features.csv."""
    rec: dict[str, str] = {
        "csv_line_num": "42",
        "original_url": "https://secure-login.example.com/auth?token=xyz",
        "normalized_url": "https://secure-login.example.com/auth?token=xyz",
        "registered_domain": "example.com",
        "original_label": "phishing",
        "threat_category": "phishing",
        "label_provenance": "phishtank_verified",
        "screenshot_path": "datasets/screenshots/raw/dummy.png",
        "screenshot_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "width": "1920",
        "height": "1080",
        "aspect_ratio": "1.7778",
        "mean_brightness": "180.25",
        "std_brightness": "45.12",
        "mean_blue": "190.10",
        "mean_green": "175.40",
        "mean_red": "165.20",
        "std_blue": "35.50",
        "std_green": "40.20",
        "std_red": "42.10",
        "color_variance": "2400.5",
        "edge_density": "0.035",
        "image_entropy": "4.21",
        "dominant_color_b": "245.0",
        "dominant_color_g": "240.0",
        "dominant_color_r": "235.0",
        "dominant_color_proportion": "0.72",
        "dominant_color_1_b": "245.0",
        "dominant_color_1_g": "240.0",
        "dominant_color_1_r": "235.0",
        "dominant_color_1_proportion": "0.72",
        "dominant_color_2_b": "120.0",
        "dominant_color_2_g": "110.0",
        "dominant_color_2_r": "100.0",
        "dominant_color_2_proportion": "0.18",
        "dominant_color_3_b": "50.0",
        "dominant_color_3_g": "40.0",
        "dominant_color_3_r": "30.0",
        "dominant_color_3_proportion": "0.10",
    }
    return rec


# ── 1. Canonical Ordering & Schema Tests ─────────────────────────────

class TestMultimodalSchema:
    def test_total_fused_features_count(self):
        """Assert exactly 51 fused features."""
        assert len(FUSED_FEATURE_NAMES) == 51

    def test_total_multimodal_fieldnames_count(self):
        """Assert exactly 9 metadata columns + 51 features = 60 columns."""
        assert len(METADATA_COLUMNS) == 9
        assert len(MULTIMODAL_FIELDNAMES) == 60
        assert MULTIMODAL_FIELDNAMES[:9] == METADATA_COLUMNS
        assert MULTIMODAL_FIELDNAMES[9:] == FUSED_FEATURE_NAMES

    def test_feature_canonical_slices(self):
        """Verify slices: [0:20] URL, [20] availability, [21:51] visual."""
        assert FUSED_FEATURE_NAMES[:20] == list(URL_FEATURE_NAMES)
        assert FUSED_FEATURE_NAMES[20] == VISUAL_AVAILABILITY_FEATURE
        assert FUSED_FEATURE_NAMES[21:] == list(VISUAL_FEATURE_NAMES)

    def test_url_feature_names_json_alignment(self):
        """Ensure feature_names.json exactly matches URL_FEATURE_NAMES."""
        json_path = os.path.join(PROJECT_ROOT, "ml", "models", "feature_names.json")
        with open(json_path, "r", encoding="utf-8") as f:
            names = json.load(f)
        assert names == list(URL_FEATURE_NAMES)

    def test_visual_feature_names_json_alignment(self):
        """Ensure visual_feature_names.json exactly matches VISUAL_FEATURE_NAMES."""
        json_path = os.path.join(PROJECT_ROOT, "ml", "models", "visual_feature_names.json")
        with open(json_path, "r", encoding="utf-8") as f:
            names = json.load(f)
        assert names == list(VISUAL_FEATURE_NAMES)


# ── 2. Record Construction Tests ─────────────────────────────────────

class TestRecordConstruction:
    def test_build_multimodal_record(self, sample_visual_record):
        """Test conversion of visual record into 51-feature multimodal record."""
        rec = build_multimodal_record(sample_visual_record)
        assert len(rec) == 60

        # Metadata checks
        for meta_k in METADATA_COLUMNS:
            assert rec[meta_k] == sample_visual_record[meta_k]

        # Feature checks
        assert rec[VISUAL_AVAILABILITY_FEATURE] == 1
        assert rec["url_length"] > 0
        assert rec["width"] == 1920
        assert rec["height"] == 1080

        # Vector conversion check
        fused_dict = {k: rec[k] for k in FUSED_FEATURE_NAMES}
        vec = fuse_feature_vector(fused_dict)
        assert vec.shape == (51,)
        assert vec.dtype == np.float64
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()

    def test_build_record_missing_visual_feature_raises(self, sample_visual_record):
        """Test that missing visual feature raises ValueError."""
        del sample_visual_record["std_brightness"]
        with pytest.raises(ValueError, match="Missing visual feature"):
            build_multimodal_record(sample_visual_record)


# ── 3. Dataset Integrity on Disk ─────────────────────────────────────

class TestDatasetIntegrityOnDisk:
    @pytest.mark.skipif(not os.path.exists(DEFAULT_OUTPUT_CSV), reason="multimodal_features.csv not generated yet")
    def test_multimodal_csv_structure(self):
        """Verify multimodal_features.csv row and column counts."""
        with open(DEFAULT_OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == MULTIMODAL_FIELDNAMES
            rows = list(reader)

        assert len(rows) == 1486

        # Check all rows have valid 51 features
        for r in rows:
            assert r[VISUAL_AVAILABILITY_FEATURE] in ("1", 1)
            for fn in FUSED_FEATURE_NAMES:
                val = float(r[fn])
                assert not math.isnan(val)
                assert not math.isinf(val)

    @pytest.mark.skipif(not os.path.exists(DEFAULT_TRAIN_CSV), reason="multimodal_train.csv not generated yet")
    def test_multimodal_split_proportions(self):
        """Verify train/test counts: 1190 train, 296 test, sum 1486."""
        with open(DEFAULT_TRAIN_CSV, "r", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))
        with open(DEFAULT_TEST_CSV, "r", encoding="utf-8") as f:
            test_rows = list(csv.DictReader(f))

        assert len(train_rows) == 1190
        assert len(test_rows) == 296
        assert len(train_rows) + len(test_rows) == 1486

    @pytest.mark.skipif(not os.path.exists(DEFAULT_TRAIN_CSV), reason="multimodal_train.csv not generated yet")
    def test_zero_domain_leakage(self):
        """Verify 0% domain overlap and 0% hash overlap between train and test."""
        with open(DEFAULT_TRAIN_CSV, "r", encoding="utf-8") as f:
            train_domains = {r["registered_domain"] for r in csv.DictReader(f)}
        with open(DEFAULT_TEST_CSV, "r", encoding="utf-8") as f:
            test_domains = {r["registered_domain"] for r in csv.DictReader(f)}

        assert train_domains & test_domains == set()

        with open(DEFAULT_TRAIN_CSV, "r", encoding="utf-8") as f:
            train_hashes = {r["screenshot_sha256"] for r in csv.DictReader(f)}
        with open(DEFAULT_TEST_CSV, "r", encoding="utf-8") as f:
            test_hashes = {r["screenshot_sha256"] for r in csv.DictReader(f)}

        assert train_hashes & test_hashes == set()

    @pytest.mark.skipif(not os.path.exists(DEFAULT_OUTPUT_CSV), reason="multimodal_features.csv not generated yet")
    def test_provenance_preservation(self):
        """Verify all rows maintain explicit provenance tags."""
        with open(DEFAULT_OUTPUT_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        provenance_set = {r["label_provenance"] for r in rows}
        assert provenance_set == {"phishtank_verified", "tranco_candidate_unverified"}

        category_set = {r["threat_category"] for r in rows}
        assert category_set == {"phishing", "unverified_candidate"}


# ── 4. Model Prediction Interface Compatibility ─────────────────────

class TestModelCompatibility:
    @pytest.mark.skipif(not os.path.exists(DEFAULT_OUTPUT_CSV), reason="multimodal_features.csv not generated yet")
    def test_url_model_prediction_on_multimodal_slice(self):
        """Verify url_model accepts the first 20 features and predicts valid classes."""
        url_model_path = os.path.join(PROJECT_ROOT, "ml", "models", "url_model.joblib")
        clf = joblib.load(url_model_path)

        with open(DEFAULT_OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            sample_rows = [next(reader) for _ in range(50)]

        X_url = np.array(
            [[float(r[fn]) for fn in URL_FEATURE_NAMES] for r in sample_rows],
            dtype=np.float64,
        )

        assert X_url.shape == (50, 20)
        preds = clf.predict(X_url)
        probas = clf.predict_proba(X_url)

        assert len(preds) == 50
        assert set(preds).issubset({0, 1, 2})
        assert probas.shape == (50, 3)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.skipif(not os.path.exists(DEFAULT_OUTPUT_CSV), reason="multimodal_features.csv not generated yet")
    def test_visual_model_prediction_on_multimodal_slice(self):
        """Verify visual_model accepts the last 30 features and predicts valid classes."""
        vis_model_path = os.path.join(PROJECT_ROOT, "ml", "models", "visual_model.joblib")
        clf = joblib.load(vis_model_path)

        with open(DEFAULT_OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            sample_rows = [next(reader) for _ in range(50)]

        X_vis = np.array(
            [[float(r[fn]) for fn in VISUAL_FEATURE_NAMES] for r in sample_rows],
            dtype=np.float64,
        )

        assert X_vis.shape == (50, 30)
        preds = clf.predict(X_vis)
        probas = clf.predict_proba(X_vis)

        assert len(preds) == 50
        assert set(preds).issubset({0, 1})
        assert probas.shape == (50, 2)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)


# ── 5. Audit Report & Documentation Verification ────────────────────

class TestAuditReportArtifacts:
    @pytest.mark.skipif(not os.path.exists(DEFAULT_AUDIT_JSON_PATH), reason="audit JSON not generated yet")
    def test_audit_json_validity(self):
        """Verify compatibility audit JSON report exists and contains required sections."""
        with open(DEFAULT_AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "schema_verification" in data
        assert "url_model_audit" in data
        assert "visual_model_audit" in data
        assert "label_compatibility_audit" in data
        assert "concordance_analysis" in data

        assert data["schema_verification"]["fused_features_exact_match"] is True
        assert data["schema_verification"]["total_records"] == 1486
        assert data["schema_verification"]["nan_count"] == 0

        # Check concordance fields
        c = data["concordance_analysis"]
        assert c["total_samples"] == 1486
        assert 0.0 < c["agreement_rate"] < 1.0

    @pytest.mark.skipif(not os.path.exists(DEFAULT_REPORT_MD_PATH), reason="report markdown not generated yet")
    def test_audit_markdown_report_validity(self):
        """Verify markdown report exists and contains required sections and tables."""
        with open(DEFAULT_REPORT_MD_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# PhishGuard – Multimodal Feature Compatibility Audit" in content
        assert "Feature Order & Schema Verification" in content
        assert "Canonical Feature Ordering Specification (51 Features)" in content
        assert "Model Prediction Interface & Compatibility Audit" in content
        assert "Label Compatibility & Non-Interchangeability Analysis" in content
        assert "Cross-Modality Concordance Analysis" in content
        assert "Artifact Manifest" in content
        assert "CRITICAL PROVENANCE & LABEL COMPATIBILITY NOTICE" in content

