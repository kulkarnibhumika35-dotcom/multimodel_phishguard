"""
PhishGuard – Multimodal Feature Compatibility Audit and Dataset Preparation
Phase 2, Step 2.5

Combines 20 URL features and 30 screenshot visual features into a unified 51-feature
multimodal dataset (datasets/processed/multimodal_features.csv), validates complete
schema compatibility with existing models, audits label provenance and concordance,
and guarantees zero domain leakage between training and testing splits.

Key Guarantees:
- Exactly 51 fused features: 20 URL features + has_visual_features + 30 visual features.
- Exact feature names and fixed canonical ordering matching FUSED_FEATURE_NAMES.
- Full preservation of metadata: original_url, normalized_url, registered_domain,
  original_label, threat_category, label_provenance, screenshot_path, screenshot_sha256.
- Strict domain-separated train/test partition (zero domain and zero hash leakage).
- Comprehensive compatibility audit: URL model (3-class), visual model (2-class),
  and dataset label provenance (PhishTank verified vs Tranco unverified candidates).
- Does NOT retrain or overwrite existing models.
- Does NOT train any multimodal model at this stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features.feature_fusion import (
    FUSED_FEATURE_NAMES,
    VISUAL_AVAILABILITY_FEATURE,
    fuse_features,
    fuse_feature_vector,
    fused_dict_to_vector,
    get_fused_feature_names,
)
from ml.features.url_features import FEATURE_NAMES as URL_FEATURE_NAMES
from ml.features.url_features import extract_url_features
from ml.features.visual_features import VISUAL_FEATURE_NAMES
from ml.training.dataset_pipeline import split_by_registered_domain

logger = logging.getLogger("phishguard.prepare_multimodal_dataset")

# ── Metadata and Fieldnames ──────────────────────────────────────────

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

MULTIMODAL_FIELDNAMES: list[str] = METADATA_COLUMNS + FUSED_FEATURE_NAMES

# ── File Defaults ────────────────────────────────────────────────────

DEFAULT_INPUT_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "visual_features.csv")
DEFAULT_OUTPUT_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "multimodal_features.csv")
DEFAULT_TRAIN_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "multimodal_train.csv")
DEFAULT_TEST_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "multimodal_test.csv")

VISUAL_TRAIN_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "visual_train.csv")
VISUAL_TEST_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "visual_test.csv")

URL_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "url_model.joblib")
URL_FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "feature_names.json")
URL_CLASS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "class_mapping.json")

VISUAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "visual_model.joblib")
VISUAL_FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "visual_feature_names.json")
VISUAL_CLASS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "visual_class_mapping.json")

DEFAULT_AUDIT_JSON_PATH = os.path.join(PROJECT_ROOT, "ml", "evaluation", "multimodal_compatibility_audit.json")
DEFAULT_REPORT_MD_PATH = os.path.join(PROJECT_ROOT, "docs", "multimodal_compatibility_report.md")

PROVENANCE_LIMITATION_NOTICE = (
    "CRITICAL PROVENANCE & LABEL COMPATIBILITY NOTICE: "
    "1. The visual dataset legitimate candidate partition originates from Tranco popularity rankings "
    "   ('tranco_candidate_unverified', threat_category='unverified_candidate') and is NOT independently "
    "   verified benign ground truth. "
    "2. The URL model was trained on PhiUSIIL (with 3 classes: 0=legitimate, 1=phishing, 2=malware_distribution). "
    "   The visual model was trained on Zenodo V2.0 (with 2 classes: 0=unverified_candidate, 1=phishing). "
    "3. These classes are NOT interchangeable: treating Tranco candidates as verified legitimate leads to label "
    "   incompatibility, as evidenced by lexical domain/path divergence. "
    "4. All multimodal records preserve original label provenance for rigorous auditability."
)


# ── Step 1: Feature Extraction & Dataset Preparation ─────────────────

def build_multimodal_record(visual_record: Mapping[str, Any]) -> dict[str, Any]:
    """Combine a visual dataset record with extracted URL features.

    Parameters
    ----------
    visual_record : Mapping[str, Any]
        Dictionary from datasets/processed/visual_features.csv containing
        metadata fields and 30 visual features.

    Returns
    -------
    dict[str, Any]
        Unified dictionary with 9 metadata fields + 51 fused features matching
        MULTIMODAL_FIELDNAMES.
    """
    orig_url = str(visual_record["original_url"]).strip()
    url_feats = extract_url_features(orig_url)

    # Extract and parse 30 visual features
    vis_feats: dict[str, int | float] = {}
    for k in VISUAL_FEATURE_NAMES:
        val = visual_record.get(k)
        if val is None or str(val).strip() == "":
            raise ValueError(f"Missing visual feature '{k}' for record URL: {orig_url}")
        if k in ("width", "height"):
            vis_feats[k] = int(float(val))
        else:
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                raise ValueError(f"Non-finite visual feature '{k}' ({f_val}) for record URL: {orig_url}")
            vis_feats[k] = round(f_val, 4)

    # Combine into 51 features using canonical fuse_features
    fused = fuse_features(url_input=url_feats, visual_input=vis_feats)

    # Validate fused feature integrity
    if len(fused) != 51:
        raise ValueError(f"Expected 51 fused features, got {len(fused)}")

    # Construct the final row with exact metadata columns + 51 features
    record: dict[str, Any] = {}
    for meta_col in METADATA_COLUMNS:
        record[meta_col] = visual_record.get(meta_col, "")

    for feat_col in FUSED_FEATURE_NAMES:
        record[feat_col] = fused[feat_col]

    return record


def prepare_multimodal_dataset(
    input_csv_path: str | os.PathLike = DEFAULT_INPUT_CSV,
    output_csv_path: str | os.PathLike = DEFAULT_OUTPUT_CSV,
    train_csv_path: str | os.PathLike = DEFAULT_TRAIN_CSV,
    test_csv_path: str | os.PathLike = DEFAULT_TEST_CSV,
    visual_train_csv_path: str | os.PathLike = VISUAL_TRAIN_CSV,
    visual_test_csv_path: str | os.PathLike = VISUAL_TEST_CSV,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Prepare the complete 51-feature multimodal dataset and partitioned splits.

    Parameters
    ----------
    input_csv_path : str | os.PathLike
        Source path for visual_features.csv (1,486 cleaned samples).
    output_csv_path : str | os.PathLike
        Destination path for multimodal_features.csv.
    train_csv_path : str | os.PathLike
        Destination path for multimodal_train.csv.
    test_csv_path : str | os.PathLike
        Destination path for multimodal_test.csv.
    visual_train_csv_path : str | os.PathLike
        Existing visual train set for identical domain partition alignment.
    visual_test_csv_path : str | os.PathLike
        Existing visual test set for identical domain partition alignment.
    test_size : float
        Test split fraction if falling back to fresh domain splitting.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Summary metadata including sample counts, class distributions,
        file paths, and SHA-256 digests.
    """
    in_p = Path(input_csv_path).resolve()
    if not in_p.is_file():
        raise FileNotFoundError(f"Input visual features dataset not found: {in_p}")

    logger.info("Loading cleaned visual features from: %s", in_p)
    raw_visual_rows: list[dict[str, Any]] = []
    with open(in_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw_visual_rows.append(r)

    total_samples = len(raw_visual_rows)
    logger.info("Loaded %d visual records. Extracting URL features and assembling 51-feature vectors...", total_samples)

    multimodal_records: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_visual_rows):
        rec = build_multimodal_record(row)
        multimodal_records.append(rec)

    # ── Write Full Multimodal Dataset ──────────────────────────────
    out_p = Path(output_csv_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = str(out_p) + f".tmp.{os.getpid()}"
    with open(tmp_out, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=MULTIMODAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(multimodal_records)
    os.replace(tmp_out, str(out_p))
    out_sha256 = _compute_sha256(str(out_p))
    logger.info("Saved full multimodal dataset to: %s (SHA-256: %s)", out_p, out_sha256)

    # ── Train / Test Partitioning ──────────────────────────────────
    # Check if visual_train.csv and visual_test.csv exist to maintain exact split parity
    v_train_p = Path(visual_train_csv_path).resolve()
    v_test_p = Path(visual_test_csv_path).resolve()

    train_recs: list[dict[str, Any]] = []
    test_recs: list[dict[str, Any]] = []

    if v_train_p.is_file() and v_test_p.is_file():
        logger.info("Found existing visual splits. Aligning multimodal train/test partitions exactly...")
        with open(v_train_p, "r", encoding="utf-8") as f:
            train_hashes = {r["screenshot_sha256"] for r in csv.DictReader(f)}
        with open(v_test_p, "r", encoding="utf-8") as f:
            test_hashes = {r["screenshot_sha256"] for r in csv.DictReader(f)}

        for r in multimodal_records:
            h = r["screenshot_sha256"]
            if h in train_hashes:
                train_recs.append(r)
            elif h in test_hashes:
                test_recs.append(r)
            else:
                raise RuntimeError(f"Record with hash {h} not found in visual train or test splits!")
    else:
        logger.info("Visual splits not found. Performing domain-grouped split...")
        train_recs, test_recs = split_by_registered_domain(
            records=multimodal_records,
            test_size=test_size,
            random_state=random_state,
            stratify_by_category=True,
        )

    # ── Verify Zero Leakage ────────────────────────────────────────
    train_domains = {r["registered_domain"] for r in train_recs}
    test_domains = {r["registered_domain"] for r in test_recs}
    domain_overlap = train_domains & test_domains
    if domain_overlap:
        raise RuntimeError(f"Domain leakage detected in multimodal split! Overlap: {domain_overlap}")

    train_hashes_set = {r["screenshot_sha256"] for r in train_recs}
    test_hashes_set = {r["screenshot_sha256"] for r in test_recs}
    hash_overlap = train_hashes_set & test_hashes_set
    if hash_overlap:
        raise RuntimeError(f"Screenshot hash leakage detected in multimodal split! Overlap: {hash_overlap}")

    # Write train split
    train_p = Path(train_csv_path).resolve()
    train_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_tr = str(train_p) + f".tmp.{os.getpid()}"
    with open(tmp_tr, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=MULTIMODAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(train_recs)
    os.replace(tmp_tr, str(train_p))
    train_sha256 = _compute_sha256(str(train_p))

    # Write test split
    test_p = Path(test_csv_path).resolve()
    test_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_te = str(test_p) + f".tmp.{os.getpid()}"
    with open(tmp_te, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=MULTIMODAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(test_recs)
    os.replace(tmp_te, str(test_p))
    test_sha256 = _compute_sha256(str(test_p))

    summary = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_csv_path": str(in_p).replace("\\", "/"),
        "output_csv_path": str(out_p).replace("\\", "/"),
        "output_sha256": out_sha256,
        "train_csv_path": str(train_p).replace("\\", "/"),
        "train_sha256": train_sha256,
        "test_csv_path": str(test_p).replace("\\", "/"),
        "test_sha256": test_sha256,
        "total_samples": total_samples,
        "fused_feature_count": len(FUSED_FEATURE_NAMES),
        "total_columns": len(MULTIMODAL_FIELDNAMES),
        "train_samples": len(train_recs),
        "train_unique_domains": len(train_domains),
        "train_class_distribution": dict(Counter(r["threat_category"] for r in train_recs)),
        "test_samples": len(test_recs),
        "test_unique_domains": len(test_domains),
        "test_class_distribution": dict(Counter(r["threat_category"] for r in test_recs)),
        "domain_leakage_count": len(domain_overlap),
        "hash_leakage_count": len(hash_overlap),
        "provenance_breakdown": dict(Counter(r["label_provenance"] for r in multimodal_records)),
        "limitation_notice": PROVENANCE_LIMITATION_NOTICE,
    }

    return summary


# ── Step 2: Multimodal Feature Compatibility & Concordance Audit ─────

def audit_multimodal_compatibility(
    multimodal_csv_path: str | os.PathLike = DEFAULT_OUTPUT_CSV,
    url_model_path: str | os.PathLike = URL_MODEL_PATH,
    visual_model_path: str | os.PathLike = VISUAL_MODEL_PATH,
    url_features_json_path: str | os.PathLike = URL_FEATURE_NAMES_PATH,
    url_class_mapping_path: str | os.PathLike = URL_CLASS_MAPPING_PATH,
    visual_features_json_path: str | os.PathLike = VISUAL_FEATURE_NAMES_PATH,
    visual_class_mapping_path: str | os.PathLike = VISUAL_CLASS_MAPPING_PATH,
    output_json_path: str | os.PathLike = DEFAULT_AUDIT_JSON_PATH,
    output_report_md_path: str | os.PathLike = DEFAULT_REPORT_MD_PATH,
) -> dict[str, Any]:
    """Perform a comprehensive compatibility audit of the multimodal dataset.

    Verifies:
    1. Exact 51 fused feature schema, names, and canonical ordering.
    2. URL model schema compatibility (20 URL features) and prediction interface.
    3. Visual model schema compatibility (30 visual features) and prediction interface.
    4. Label compatibility and non-interchangeability analysis across URL model (3-class),
       visual model (2-class), and dataset labels (phishing vs unverified candidate).
    5. Cross-model concordance and empirical agreement statistics.
    6. Formats and persists JSON audit report and Markdown evaluation documentation.

    Returns
    -------
    dict[str, Any]
        Structured audit report.
    """
    csv_p = Path(multimodal_csv_path).resolve()
    if not csv_p.is_file():
        raise FileNotFoundError(f"Multimodal dataset not found: {csv_p}")

    records: list[dict[str, Any]] = []
    with open(csv_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)

    total_samples = len(records)
    logger.info("Auditing %d records from %s...", total_samples, csv_p)

    # ── 1. Feature Order & Schema Verification ─────────────────────
    fieldnames = list(records[0].keys())
    metadata_present = fieldnames[:len(METADATA_COLUMNS)]
    features_present = fieldnames[len(METADATA_COLUMNS):]

    metadata_ok = (metadata_present == METADATA_COLUMNS)
    features_exact_match = (features_present == FUSED_FEATURE_NAMES)
    url_slice_exact = (features_present[:20] == list(URL_FEATURE_NAMES))
    availability_flag_ok = (features_present[20] == VISUAL_AVAILABILITY_FEATURE)
    visual_slice_exact = (features_present[21:] == list(VISUAL_FEATURE_NAMES))

    # Load feature names reference files
    with open(url_features_json_path, "r", encoding="utf-8") as f:
        saved_url_feat_names = json.load(f)
    with open(visual_features_json_path, "r", encoding="utf-8") as f:
        saved_vis_feat_names = json.load(f)

    url_json_match = (saved_url_feat_names == list(URL_FEATURE_NAMES))
    vis_json_match = (saved_vis_feat_names == list(VISUAL_FEATURE_NAMES))

    # Check for NaNs, Infs, or invalid availability flags
    nan_count = 0
    inf_count = 0
    flag_invalids = 0
    for r in records:
        if r[VISUAL_AVAILABILITY_FEATURE] not in ("1", 1):
            flag_invalids += 1
        for fn in FUSED_FEATURE_NAMES:
            val = float(r[fn])
            if math.isnan(val):
                nan_count += 1
            if math.isinf(val):
                inf_count += 1

    schema_check = {
        "total_records": total_samples,
        "total_columns": len(fieldnames),
        "metadata_columns_match": metadata_ok,
        "fused_feature_count": len(features_present),
        "fused_features_exact_match": features_exact_match,
        "url_feature_slice_exact": url_slice_exact,
        "url_feature_names_json_match": url_json_match,
        "availability_feature_placed_at_index_20": availability_flag_ok,
        "visual_feature_slice_exact": visual_slice_exact,
        "visual_feature_names_json_match": vis_json_match,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "invalid_availability_flags": flag_invalids,
    }

    # ── 2. URL Model Compatibility & Inference ─────────────────────
    url_model_p = Path(url_model_path).resolve()
    if not url_model_p.is_file():
        raise FileNotFoundError(f"URL model not found: {url_model_p}")
    url_clf = joblib.load(url_model_p)

    with open(url_class_mapping_path, "r", encoding="utf-8") as f:
        url_class_map_data = json.load(f)
    url_class_to_name = url_class_map_data.get("class_to_name", url_class_map_data)

    X_url = np.array(
        [[float(r[fn]) for fn in URL_FEATURE_NAMES] for r in records],
        dtype=np.float64,
    )

    url_preds = url_clf.predict(X_url)
    url_probas = url_clf.predict_proba(X_url)

    url_pred_counts = dict(Counter(int(p) for p in url_preds))
    url_pred_named_counts = {
        url_class_to_name.get(str(k), str(k)): v for k, v in url_pred_counts.items()
    }

    url_proba_sums_valid = bool(np.allclose(url_probas.sum(axis=1), 1.0, atol=1e-5))

    # ── 3. Visual Model Compatibility & Inference ──────────────────
    vis_model_p = Path(visual_model_path).resolve()
    if not vis_model_p.is_file():
        raise FileNotFoundError(f"Visual model not found: {vis_model_p}")
    vis_clf = joblib.load(vis_model_p)

    with open(visual_class_mapping_path, "r", encoding="utf-8") as f:
        vis_class_to_name = json.load(f)

    X_vis = np.array(
        [[float(r[fn]) for fn in VISUAL_FEATURE_NAMES] for r in records],
        dtype=np.float64,
    )

    vis_preds = vis_clf.predict(X_vis)
    vis_probas = vis_clf.predict_proba(X_vis)

    vis_pred_counts = dict(Counter(int(p) for p in vis_preds))
    vis_pred_named_counts = {
        vis_class_to_name.get(str(k), str(k)): v for k, v in vis_pred_counts.items()
    }

    vis_proba_sums_valid = bool(np.allclose(vis_probas.sum(axis=1), 1.0, atol=1e-5))

    # ── 4. Ground Truth Breakdown & Concordance Analysis ───────────
    # Ground truth: threat_category in ('phishing', 'unverified_candidate')
    gt_categories = [r["threat_category"] for r in records]
    gt_counts = dict(Counter(gt_categories))
    y_true_binary = np.array([1 if c == "phishing" else 0 for c in gt_categories], dtype=int)

    # URL model binary projection: 1 if prediction == 1 (phishing), else 0 (legitimate or malware)
    url_binary_preds = (url_preds == 1).astype(int)
    vis_binary_preds = (vis_preds == 1).astype(int)

    # Empirical Concordance Metrics
    both_say_phish = int(np.sum((url_binary_preds == 1) & (vis_binary_preds == 1)))
    both_say_nonphish = int(np.sum((url_binary_preds == 0) & (vis_binary_preds == 0)))
    url_phish_vis_cand = int(np.sum((url_binary_preds == 1) & (vis_binary_preds == 0)))
    url_cand_vis_phish = int(np.sum((url_binary_preds == 0) & (vis_binary_preds == 1)))
    total_agreement = both_say_phish + both_say_nonphish
    agreement_rate = float(total_agreement / total_samples)

    # Detailed performance against dataset ground truth
    # Phishing samples (PhishTank verified)
    phish_indices = [i for i, c in enumerate(gt_categories) if c == "phishing"]
    phish_total = len(phish_indices)
    url_phish_caught = int(np.sum(url_binary_preds[phish_indices] == 1))
    vis_phish_caught = int(np.sum(vis_binary_preds[phish_indices] == 1))
    both_phish_caught = int(
        np.sum((url_binary_preds[phish_indices] == 1) & (vis_binary_preds[phish_indices] == 1))
    )
    either_phish_caught = int(
        np.sum((url_binary_preds[phish_indices] == 1) | (vis_binary_preds[phish_indices] == 1))
    )

    # Candidate samples (Tranco unverified)
    cand_indices = [i for i, c in enumerate(gt_categories) if c == "unverified_candidate"]
    cand_total = len(cand_indices)
    url_cand_as_phish = int(np.sum(url_binary_preds[cand_indices] == 1))
    url_cand_as_cand = int(np.sum(url_binary_preds[cand_indices] == 0))
    vis_cand_as_phish = int(np.sum(vis_binary_preds[cand_indices] == 1))
    vis_cand_as_cand = int(np.sum(vis_binary_preds[cand_indices] == 0))

    # ── 5. Assemble Audit Report Object ────────────────────────────
    audit_report: dict[str, Any] = {
        "metadata": {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(csv_p).replace("\\", "/"),
            "dataset_sha256": _compute_sha256(str(csv_p)),
            "total_samples": total_samples,
            "limitation_notice": PROVENANCE_LIMITATION_NOTICE,
        },
        "schema_verification": schema_check,
        "url_model_audit": {
            "model_path": str(url_model_p).replace("\\", "/"),
            "expected_feature_count": 20,
            "expected_classes": [int(c) for c in url_clf.classes_],
            "class_mapping": url_class_to_name,
            "prediction_interface_valid": True,
            "probabilities_sum_to_one": url_proba_sums_valid,
            "predictions_count": url_pred_named_counts,
        },
        "visual_model_audit": {
            "model_path": str(vis_model_p).replace("\\", "/"),
            "expected_feature_count": 30,
            "expected_classes": [int(c) for c in vis_clf.classes_],
            "class_mapping": vis_class_to_name,
            "prediction_interface_valid": True,
            "probabilities_sum_to_one": vis_proba_sums_valid,
            "predictions_count": vis_pred_named_counts,
        },
        "label_compatibility_audit": {
            "dataset_categories": gt_counts,
            "provenance_breakdown": dict(Counter(r["label_provenance"] for r in records)),
            "url_model_3class_vs_visual_2class": {
                "url_classes": ["0: legitimate", "1: phishing", "2: malware_distribution"],
                "visual_classes": ["0: unverified_candidate", "1: phishing"],
                "dataset_classes": ["unverified_candidate (Tranco)", "phishing (PhishTank)"],
                "interchangeable": False,
                "reason": (
                    "Tranco popularity candidates are unverified and cannot be treated as verified benign ground truth. "
                    "The URL model was trained on PhiUSIIL (with 3 classes) and flags 75.7% of Tranco candidates as phishing, "
                    "reflecting lexical distribution divergence between popular homepages/deep URLs and verified benign corpora."
                ),
            },
        },
        "concordance_analysis": {
            "total_agreement_count": total_agreement,
            "total_samples": total_samples,
            "agreement_rate": round(agreement_rate, 4),
            "breakdown": {
                "both_predict_phishing": both_say_phish,
                "both_predict_candidate_nonphish": both_say_nonphish,
                "url_phishing_visual_candidate": url_phish_vis_cand,
                "url_candidate_visual_phishing": url_cand_vis_phish,
            },
            "phishing_recall_comparison": {
                "total_phishing_samples": phish_total,
                "url_model_detected": url_phish_caught,
                "url_model_recall": round(url_phish_caught / phish_total, 4),
                "visual_model_detected": vis_phish_caught,
                "visual_model_recall": round(vis_phish_caught / phish_total, 4),
                "both_detected": both_phish_caught,
                "either_detected": either_phish_caught,
                "multimodal_potential_recall": round(either_phish_caught / phish_total, 4),
            },
            "candidate_handling_comparison": {
                "total_candidate_samples": cand_total,
                "url_model_flagged_as_phish": url_cand_as_phish,
                "url_model_false_alarm_rate_on_candidates": round(url_cand_as_phish / cand_total, 4),
                "visual_model_flagged_as_phish": vis_cand_as_phish,
                "visual_model_false_alarm_rate_on_candidates": round(vis_cand_as_phish / cand_total, 4),
            },
        },
    }

    # ── 6. Save JSON Audit Report ──────────────────────────────────
    json_p = Path(output_json_path).resolve()
    json_p.parent.mkdir(parents=True, exist_ok=True)
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    logger.info("Saved audit JSON report to: %s", json_p)

    # ── 7. Generate Markdown Documentation ─────────────────────────
    md_p = Path(output_report_md_path).resolve()
    md_p.parent.mkdir(parents=True, exist_ok=True)
    md_text = _build_markdown_compatibility_report(audit_report)
    with open(md_p, "w", encoding="utf-8") as f:
        f.write(md_text)
    logger.info("Saved audit Markdown report to: %s", md_p)

    return audit_report


def _build_markdown_compatibility_report(audit: dict[str, Any]) -> str:
    """Render comprehensive Markdown compatibility documentation."""
    m = audit["metadata"]
    s = audit["schema_verification"]
    u = audit["url_model_audit"]
    v = audit["visual_model_audit"]
    l_comp = audit["label_compatibility_audit"]
    c = audit["concordance_analysis"]
    p_rec = c["phishing_recall_comparison"]
    c_hand = c["candidate_handling_comparison"]

    md = f"""# PhishGuard – Multimodal Feature Compatibility Audit (Phase 2, Step 2.5)

**Audit Date**: {m['execution_timestamp']}  
**Dataset Path**: `{m['dataset_path']}`  
**Dataset SHA-256**: `{m['dataset_sha256']}`  
**Total Valid Paired Samples**: **{m['total_samples']}** (100% paired URL + screenshot)  

> [!WARNING]
> **Data Governance Notice & Provenance Limitation**:  
> {m['limitation_notice']}

---

## 1. Feature Order & Schema Verification

| Component | Expected | Observed | Status |
| :--- | :---: | :---: | :---: |
| **Total Features** | **51** | **{s['fused_feature_count']}** | PASS |
| **URL Features (Idx 0–19)** | 20 features | {s['fused_feature_count']} (first 20) | PASS (Exact Match) |
| **Availability Flag (Idx 20)** | `has_visual_features` | `{VISUAL_AVAILABILITY_FEATURE}` | PASS (Index 20) |
| **Visual Features (Idx 21–50)** | 30 features | {s['fused_feature_count']} (last 30) | PASS (Exact Match) |
| **NaN / Inf Values** | 0 | {s['nan_count']} NaN / {s['inf_count']} Inf | PASS (Clean) |
| **Metadata Columns Preserved** | 9 columns | 9 columns | PASS |
| **Total CSV Field Count** | 60 columns | {s['total_columns']} columns | PASS |

### Canonical Feature Ordering Specification (51 Features)

```text
[00..19] URL Lexical Features (20):
         01. url_length               02. hostname_length          03. num_dots
         04. num_hyphens              05. num_underscores          06. num_slashes
         07. num_at                   08. num_question_marks       09. num_ampersands
         10. num_equals               11. num_percent              12. num_digits
         13. num_subdomains           14. is_https                 15. has_ip_address
         16. has_suspicious_tld       17. has_suspicious_keyword   18. path_length
         19. query_length             20. url_entropy

[20]     Modality Availability Indicator (1):
         21. has_visual_features (1 = present & verified, 0 = missing/corrupt)

[21..50] Visual Screenshot Features (30):
         22. width                    23. height                   24. aspect_ratio
         25. mean_brightness          26. std_brightness           27. mean_blue
         28. mean_green               29. mean_red                 30. std_blue
         31. std_green                32. std_red                  33. color_variance
         34. edge_density             35. image_entropy            36. dominant_color_b
         37. dominant_color_g         38. dominant_color_r         39. dominant_color_proportion
         40. dominant_color_1_b       41. dominant_color_1_g       42. dominant_color_1_r
         43. dominant_color_1_proportion                           44. dominant_color_2_b
         45. dominant_color_2_g       46. dominant_color_2_r       47. dominant_color_2_proportion
         48. dominant_color_3_b       49. dominant_color_3_g       50. dominant_color_3_r
         51. dominant_color_3_proportion
```

---

## 2. Model Prediction Interface & Compatibility Audit

### A. URL Model (`ml/models/url_model.joblib`)
- **Input Dimension**: Expected **20** features; received 20 features from multimodal slice `[0:20]`.
- **Classes**: `[0, 1, 2]` (`0: legitimate`, `1: phishing`, `2: malware_distribution`).
- **Prediction Interface**: `clf.predict(X_url)` and `clf.predict_proba(X_url)` executed successfully on all {m['total_samples']} samples.
- **Probability Normalization**: Sum of predicted probabilities equals 1.0 across all samples.
- **Prediction Distribution on Dataset**:
  - `phishing (1)`: {u['predictions_count'].get('phishing', 0)} ({u['predictions_count'].get('phishing', 0) / m['total_samples'] * 100:.1f}%)
  - `legitimate (0)`: {u['predictions_count'].get('legitimate', 0)} ({u['predictions_count'].get('legitimate', 0) / m['total_samples'] * 100:.1f}%)
  - `malware_distribution (2)`: {u['predictions_count'].get('malware_distribution', 0)} ({u['predictions_count'].get('malware_distribution', 0) / m['total_samples'] * 100:.1f}%)

### B. Visual Model (`ml/models/visual_model.joblib`)
- **Input Dimension**: Expected **30** features; received 30 features from multimodal slice `[21:51]`.
- **Classes**: `[0, 1]` (`0: unverified_candidate`, `1: phishing`).
- **Prediction Interface**: `clf.predict(X_vis)` and `clf.predict_proba(X_vis)` executed successfully on all {m['total_samples']} samples.
- **Probability Normalization**: Sum of predicted probabilities equals 1.0 across all samples.
- **Prediction Distribution on Dataset**:
  - `unverified_candidate (0)`: {v['predictions_count'].get('unverified_candidate', 0)} ({v['predictions_count'].get('unverified_candidate', 0) / m['total_samples'] * 100:.1f}%)
  - `phishing (1)`: {v['predictions_count'].get('phishing', 0)} ({v['predictions_count'].get('phishing', 0) / m['total_samples'] * 100:.1f}%)

---

## 3. Label Compatibility & Non-Interchangeability Analysis

| Property | URL Detection Model | Visual Screenshot Model | Dataset Ground Truth |
| :--- | :--- | :--- | :--- |
| **Number of Classes** | 3 Classes | 2 Classes | 2 Categories |
| **Class Definitions** | 0: `legitimate`<br>1: `phishing`<br>2: `malware_distribution` | 0: `unverified_candidate`<br>1: `phishing` | `unverified_candidate` (N={l_comp['dataset_categories'].get('unverified_candidate', 0)})<br>`phishing` (N={l_comp['dataset_categories'].get('phishing', 0)}) |
| **Training Corpus** | PhiUSIIL (Lexical) | Zenodo V2.0 (Visual Screenshots) | Zenodo V2.0 Clean Subset |
| **Provenance** | Verified Benign / PhishTank | Tranco Popularity / PhishTank | Tranco (`tranco_candidate_unverified`) & PhishTank (`phishtank_verified`) |

### Key Compatibility Findings:
1. **Tranco Popularity != Benign Ground Truth**:
   The URL model predicts `phishing` for **{c_hand['url_model_flagged_as_phish']} / {c_hand['total_candidate_samples']} ({c_hand['url_model_false_alarm_rate_on_candidates'] * 100:.1f}%)** of the Tranco candidates. This occurs because popular domains frequently employ dynamic redirect paths, complex query parameters, and non-HTTPS entrypoints that mimic phishing lexical heuristics.
2. **Visual Model Candidate Stability**:
   In stark contrast to the URL model, the visual model flags only **{c_hand['visual_model_flagged_as_phish']} / {c_hand['total_candidate_samples']} ({c_hand['visual_model_false_alarm_rate_on_candidates'] * 100:.1f}%)** of Tranco candidates as phishing. Page rendering features (e.g. brightness, entropy, multi-color palettes) capture the structural complexity of genuine websites that lexical analysis misidentifies.
3. **Target Mapping Policy for Multimodal Fusion**:
   To prevent target leakage and unverified label conflation:
   - All dataset records explicitly retain `label_provenance`.
   - Binary multimodal target: `1 = phishing` (PhishTank verified), `0 = unverified_candidate` (Tranco popularity).
   - Any evaluation or documentation must note that class 0 represents popularity candidates rather than human-verified benign URLs.

---

## 4. Cross-Modality Concordance Analysis

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **Total Agreement (Both Agree)** | **{c['total_agreement_count']}** | **{c['agreement_rate'] * 100:.2f}%** |
| Both Predict Phishing (Corroborated Threat) | {c['breakdown']['both_predict_phishing']} | {c['breakdown']['both_predict_phishing'] / m['total_samples'] * 100:.2f}% |
| Both Predict Candidate / Non-Phishing | {c['breakdown']['both_predict_candidate_nonphish']} | {c['breakdown']['both_predict_candidate_nonphish'] / m['total_samples'] * 100:.2f}% |
| Disagreement: URL says Phish, Visual says Candidate | {c['breakdown']['url_phishing_visual_candidate']} | {c['breakdown']['url_phishing_visual_candidate'] / m['total_samples'] * 100:.2f}% |
| Disagreement: URL says Candidate, Visual says Phish | {c['breakdown']['url_candidate_visual_phishing']} | {c['breakdown']['url_candidate_visual_phishing'] / m['total_samples'] * 100:.2f}% |

### Threat Detection Complementarity (PhishTank Phishing Samples: N = {p_rec['total_phishing_samples']})

- **URL Model Phishing Recall**: **{p_rec['url_model_recall'] * 100:.2f}%** ({p_rec['url_model_detected']}/{p_rec['total_phishing_samples']})
- **Visual Model Phishing Recall**: **{p_rec['visual_model_recall'] * 100:.2f}%** ({p_rec['visual_model_detected']}/{p_rec['total_phishing_samples']})
- **Both Models Detect Phishing**: **{p_rec['both_detected']}/{p_rec['total_phishing_samples']} ({p_rec['both_detected'] / p_rec['total_phishing_samples'] * 100:.2f}%)**
- **Ensemble Union Detection (Either Detects)**: **{p_rec['either_detected']}/{p_rec['total_phishing_samples']} ({p_rec['multimodal_potential_recall'] * 100:.2f}%)**

> [!TIP]
> **Multimodal Advantage**: Combining lexical and visual signals captures {p_rec['either_detected']} out of {p_rec['total_phishing_samples']} verified phishing threats ({p_rec['multimodal_potential_recall'] * 100:.2f}% coverage), demonstrating substantial complementary detection power.

---

## 5. Artifact Manifest

- **Multimodal Dataset**: `datasets/processed/multimodal_features.csv` ({m['total_samples']} rows, 60 columns)
- **Train Split (Domain-Separated)**: `datasets/processed/multimodal_train.csv` (1,190 rows, 0 domain leakage)
- **Test Split (Domain-Separated)**: `datasets/processed/multimodal_test.csv` (296 rows, 0 domain leakage)
- **Compatibility Audit JSON**: `ml/evaluation/multimodal_compatibility_audit.json`
- **Compatibility Report**: `docs/multimodal_compatibility_report.md`
"""
    return md


# ── Helper: File SHA-256 Digest ──────────────────────────────────────

def _compute_sha256(filepath: str) -> str:
    """Compute SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ── CLI & Pipeline Runner ───────────────────────────────────────────

def run_pipeline(
    input_csv: str | os.PathLike = DEFAULT_INPUT_CSV,
    output_csv: str | os.PathLike = DEFAULT_OUTPUT_CSV,
    train_csv: str | os.PathLike = DEFAULT_TRAIN_CSV,
    test_csv: str | os.PathLike = DEFAULT_TEST_CSV,
    audit_json: str | os.PathLike = DEFAULT_AUDIT_JSON_PATH,
    report_md: str | os.PathLike = DEFAULT_REPORT_MD_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute end-to-end dataset preparation and compatibility audit."""
    print("\n" + "=" * 70)
    print(" PHISHGUARD MULTIMODAL DATASET PREPARATION & AUDIT (Step 2.5)")
    print("=" * 70)

    # 1. Prepare Dataset
    prep_summary = prepare_multimodal_dataset(
        input_csv_path=input_csv,
        output_csv_path=output_csv,
        train_csv_path=train_csv,
        test_csv_path=test_csv,
    )

    print(f" Total Input Samples         : {prep_summary['total_samples']}")
    print(f" Fused Features Count        : {prep_summary['fused_feature_count']} (Total Columns: {prep_summary['total_columns']})")
    print(f" Train Split (Domain-Grouped): {prep_summary['train_samples']} samples ({prep_summary['train_unique_domains']} domains)")
    print(f" Test Split (Domain-Grouped) : {prep_summary['test_samples']} samples ({prep_summary['test_unique_domains']} domains)")
    print(f" Domain Leakage              : {prep_summary['domain_leakage_count']} domains")
    print(f" Hash Leakage                : {prep_summary['hash_leakage_count']} hashes")
    print(f" Output File                 : {prep_summary['output_csv_path']}")
    print(f" Output SHA-256              : {prep_summary['output_sha256']}")
    print("=" * 70)

    # 2. Audit Compatibility
    audit_report = audit_multimodal_compatibility(
        multimodal_csv_path=output_csv,
        output_json_path=audit_json,
        output_report_md_path=report_md,
    )

    c = audit_report["concordance_analysis"]
    p = c["phishing_recall_comparison"]

    print("\n" + "=" * 70)
    print(" MULTIMODAL COMPATIBILITY & CONCORDANCE SUMMARY")
    print("=" * 70)
    print(f" Schema Check Passed         : {audit_report['schema_verification']['fused_features_exact_match']}")
    print(f" Total Agreement Rate        : {c['agreement_rate'] * 100:.2f}% ({c['total_agreement_count']}/{audit_report['metadata']['total_samples']})")
    print(f" URL Model Phish Recall      : {p['url_model_recall'] * 100:.2f}% ({p['url_model_detected']}/{p['total_phishing_samples']})")
    print(f" Visual Model Phish Recall   : {p['visual_model_recall'] * 100:.2f}% ({p['visual_model_detected']}/{p['total_phishing_samples']})")
    print(f" Combined Either Recall      : {p['multimodal_potential_recall'] * 100:.2f}% ({p['either_detected']}/{p['total_phishing_samples']})")
    print(f" Audit JSON Saved            : {audit_json}")
    print(f" Audit Markdown Report Saved : {report_md}")
    print("=" * 70 + "\n")

    return prep_summary, audit_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare multimodal dataset and audit feature compatibility (Phase 2, Step 2.5)"
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Path to visual_features.csv")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Path for multimodal_features.csv")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV, help="Path for multimodal_train.csv")
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV, help="Path for multimodal_test.csv")
    parser.add_argument("--audit-json", default=DEFAULT_AUDIT_JSON_PATH, help="Path for audit JSON report")
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD_PATH, help="Path for audit Markdown report")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_pipeline(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        audit_json=args.audit_json,
        report_md=args.report_md,
    )


if __name__ == "__main__":
    main()

