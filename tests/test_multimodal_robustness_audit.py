"""
Unit tests for PhishGuard Multimodal Robustness, Data Leakage, and Ablation Audit
(ml/evaluation/audit_multimodal_model.py, Phase 2 Step 2.7).
"""

from __future__ import annotations

import csv
import json
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

from ml.evaluation.audit_multimodal_model import (
    DEFAULT_AUDIT_JSON_PATH,
    DEFAULT_AUDIT_MD_PATH,
    GOVERNANCE_LIMITATION_NOTICE,
    METADATA_COLUMNS,
    MULTIMODAL_CLASS_MAPPING_PATH,
    MULTIMODAL_FEATURE_NAMES_PATH,
    MULTIMODAL_MODEL_PATH,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    URL_CLASS_MAPPING_PATH,
    URL_MODEL_PATH,
    VISUAL_CLASS_MAPPING_PATH,
    VISUAL_MODEL_PATH,
    audit_class_mappings,
    audit_data_leakage,
    audit_feature_ablation,
    audit_robustness_and_errors,
)
from ml.features.feature_fusion import FUSED_FEATURE_NAMES, VISUAL_AVAILABILITY_FEATURE
from ml.features.url_features import FEATURE_NAMES as URL_FEATURE_NAMES
from ml.features.visual_features import VISUAL_FEATURE_NAMES


# ── 1. Data Leakage & Isolation Tests ────────────────────────────────

class TestDataLeakageAudit:
    def test_zero_domain_leakage(self):
        """Verify 0% registered domain overlap between train and test sets."""
        with open(TRAIN_CSV_PATH, "r", encoding="utf-8") as f:
            train_domains = {r["registered_domain"] for r in csv.DictReader(f)}
        with open(TEST_CSV_PATH, "r", encoding="utf-8") as f:
            test_domains = {r["registered_domain"] for r in csv.DictReader(f)}

        assert train_domains.isdisjoint(test_domains)

    def test_zero_url_leakage(self):
        """Verify 0% original and normalized URL overlap between train and test sets."""
        with open(TRAIN_CSV_PATH, "r", encoding="utf-8") as f:
            tr_r = list(csv.DictReader(f))
        with open(TEST_CSV_PATH, "r", encoding="utf-8") as f:
            te_r = list(csv.DictReader(f))

        train_urls = {r["original_url"] for r in tr_r}
        test_urls = {r["original_url"] for r in te_r}
        assert train_urls.isdisjoint(test_urls)

        train_norm_urls = {r["normalized_url"] for r in tr_r}
        test_norm_urls = {r["normalized_url"] for r in te_r}
        assert train_norm_urls.isdisjoint(test_norm_urls)

    def test_zero_screenshot_hash_leakage(self):
        """Verify 0% screenshot SHA-256 hash overlap between train and test sets."""
        with open(TRAIN_CSV_PATH, "r", encoding="utf-8") as f:
            train_hashes = {r["screenshot_sha256"] for r in csv.DictReader(f)}
        with open(TEST_CSV_PATH, "r", encoding="utf-8") as f:
            test_hashes = {r["screenshot_sha256"] for r in csv.DictReader(f)}

        assert train_hashes.isdisjoint(test_hashes)

    def test_zero_visual_feature_vector_exact_overlap(self):
        """Verify 0% identical 30-feature visual vectors between train and test."""
        with open(TRAIN_CSV_PATH, "r", encoding="utf-8") as f:
            tr_r = list(csv.DictReader(f))
        with open(TEST_CSV_PATH, "r", encoding="utf-8") as f:
            te_r = list(csv.DictReader(f))

        train_vis_vecs = {tuple(round(float(r[fn]), 4) for fn in VISUAL_FEATURE_NAMES) for r in tr_r}
        test_vis_vecs = {tuple(round(float(r[fn]), 4) for fn in VISUAL_FEATURE_NAMES) for r in te_r}
        assert train_vis_vecs.isdisjoint(test_vis_vecs)

    def test_metadata_and_target_columns_exclusion(self):
        """Verify all metadata and label columns are strictly excluded from model feature names."""
        with open(MULTIMODAL_FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
            model_feat_names = json.load(f)

        assert len(model_feat_names) == 51
        for meta_col in METADATA_COLUMNS:
            assert meta_col not in model_feat_names

        for target_col in ["threat_category", "label", "original_label"]:
            assert target_col not in model_feat_names


# ── 2. Class Mapping & Governance Tests ──────────────────────────────

class TestClassMappingAudit:
    def test_multimodal_class_mapping(self):
        """Verify class 0 is unverified_candidate and class 1 is phishing for multimodal model."""
        with open(MULTIMODAL_CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            multi_map = json.load(f)

        assert multi_map["0"] == "unverified_candidate"
        assert multi_map["1"] == "phishing"

    def test_url_model_three_class_mapping(self):
        """Verify 3-class URL model mapping defines 0=legitimate, 1=phishing, 2=malware_distribution."""
        with open(URL_CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            url_map = json.load(f)

        c2n = url_map.get("class_to_name", url_map)
        assert c2n["0"] == "legitimate"
        assert c2n["1"] == "phishing"
        assert c2n["2"] == "malware_distribution"

    def test_malware_class_isolated_from_non_threats(self):
        """Verify malware_distribution is never mapped to 0 (legitimate or unverified candidate)."""
        with open(URL_CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            url_map = json.load(f)

        c2n = url_map.get("class_to_name", url_map)
        assert c2n.get("0") != "malware_distribution"
        assert c2n.get("1") != "malware_distribution"
        assert c2n.get("2") == "malware_distribution"

    def test_positive_class_consistency(self):
        """Verify all models treat class 1 (phishing) as the primary positive detection target."""
        clf_multi = joblib.load(MULTIMODAL_MODEL_PATH)
        clf_vis = joblib.load(VISUAL_MODEL_PATH)

        assert list(clf_multi.classes_) == [0, 1]
        assert list(clf_vis.classes_) == [0, 1]


# ── 3. Feature Slicing & Ablation Invariants ─────────────────────────

class TestFeatureSlicingAndAblation:
    def test_canonical_feature_slices(self):
        """Verify slice [0:20] is URL, [20] is availability indicator, [21:51] is visual."""
        assert FUSED_FEATURE_NAMES[:20] == list(URL_FEATURE_NAMES)
        assert FUSED_FEATURE_NAMES[20] == VISUAL_AVAILABILITY_FEATURE
        assert FUSED_FEATURE_NAMES[21:] == list(VISUAL_FEATURE_NAMES)
        assert len(FUSED_FEATURE_NAMES) == 51

    def test_url_ablation_masking_runs_safely(self):
        """Verify model handles visual ablation (availability=0, visual=0.0) without error."""
        clf_multi = joblib.load(MULTIMODAL_MODEL_PATH)
        dummy_X = np.ones((5, 51), dtype=np.float64)
        dummy_X[:, 20] = 0.0
        dummy_X[:, 21:] = 0.0

        preds = clf_multi.predict(dummy_X)
        probas = clf_multi.predict_proba(dummy_X)

        assert preds.shape == (5,)
        assert probas.shape == (5, 2)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    def test_visual_ablation_masking_runs_safely(self):
        """Verify model handles URL ablation (URL=0.0, availability=1) without error."""
        clf_multi = joblib.load(MULTIMODAL_MODEL_PATH)
        dummy_X = np.ones((5, 51), dtype=np.float64)
        dummy_X[:, :20] = 0.0
        dummy_X[:, 20] = 1.0

        preds = clf_multi.predict(dummy_X)
        probas = clf_multi.predict_proba(dummy_X)

        assert preds.shape == (5,)
        assert probas.shape == (5, 2)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)


# ── 4. Metric Consistency & Audit Report Artifacts ───────────────────

class TestAuditReportArtifacts:
    def test_audit_json_exists_and_valid(self):
        """Verify ml/evaluation/multimodal_robustness_audit.json exists and has required sections."""
        assert os.path.isfile(DEFAULT_AUDIT_JSON_PATH)
        with open(DEFAULT_AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "governance_limitation_notice" in data
        assert "data_leakage_audit" in data
        assert "feature_ablation_audit" in data
        assert "class_mapping_audit" in data
        assert "robustness_and_error_analysis" in data
        assert "controlled_integration_verdict" in data

        assert data["data_leakage_audit"]["leakage_audit_passed"] is True
        assert data["controlled_integration_verdict"]["ready_for_controlled_canary_experiment"] is True

    def test_audit_markdown_exists_and_contains_sections(self):
        """Verify docs/multimodal_robustness_audit.md exists and contains required sections."""
        assert os.path.isfile(DEFAULT_AUDIT_MD_PATH)
        with open(DEFAULT_AUDIT_MD_PATH, "r", encoding="utf-8") as f:
            md = f.read()

        assert "# PhishGuard – Multimodal Model Robustness, Data Leakage & Ablation Audit" in md
        assert "Data Leakage Audit Findings" in md
        assert "Feature Ablation Audit (True Ablation vs Standalone Models)" in md
        assert "Class Mapping & Governance Audit" in md
        assert "Robustness Analysis & Error Deep Dive" in md
        assert "Controlled Integration Readiness Assessment" in md
        assert "CRITICAL GOVERNANCE LIMITATION" in md

    def test_false_negative_invariants(self):
        """Verify all false negatives in test set share apex / bare domain characteristics."""
        with open(DEFAULT_AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        fns = data["robustness_and_error_analysis"]["false_negatives"]
        assert len(fns) == 4
        for fn in fns:
            assert fn["path_length"] == 0
            assert fn["num_slashes"] == 2
            assert fn["multimodal_prob_phish"] < 0.50

    def test_ablation_graceful_degradation_metrics(self):
        """Verify URL ablation retains > 95% accuracy and high precision."""
        with open(DEFAULT_AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        t_abl = data["feature_ablation_audit"]["true_ablation_experiments"]
        full_acc = t_abl["full_multimodal_51_features"]["accuracy"]
        url_abl_acc = t_abl["url_only_visual_masked"]["accuracy"]

        assert full_acc > 0.98
        assert url_abl_acc > 0.95

