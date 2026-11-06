"""
Unit Tests for PhishGuard Dataset Quality Cleaning and Batch Visual Feature Extraction
(datasets/scripts/clean_and_extract_visual_features.py)
"""

import csv
import json
import math
import os
import cv2
import numpy as np
import pytest

from datasets.scripts.clean_and_extract_visual_features import (
    CSV_FIELDNAMES,
    METADATA_COLUMNS,
    clean_visual_dataset,
    extract_features_batch,
    identify_cross_class_domains,
    identify_cross_class_screenshot_hashes,
    is_blank_or_nearly_blank,
    run_cleaning_and_feature_extraction,
    validate_processed_visual_dataset,
)
from ml.features.visual_features import VISUAL_FEATURE_NAMES
from ml.training.dataset_pipeline import (
    KNOWN_CONFLICTING_DOMAINS,
    THREAT_CATEGORY_PHISHING,
    THREAT_CATEGORY_UNVERIFIED,
)


@pytest.fixture
def synthetic_images(tmp_path):
    """Generate synthetic images covering diverse visual characteristics."""
    imgs_dir = tmp_path / "images"
    imgs_dir.mkdir(parents=True)

    # 1. Normal rich image (100x100 BGR, green with shapes)
    normal_img = np.zeros((100, 100, 3), dtype=np.uint8)
    normal_img[:, :] = [50, 180, 50]
    cv2.circle(normal_img, (50, 50), 20, (0, 0, 255), -1)
    normal_path = imgs_dir / "normal.png"
    cv2.imwrite(str(normal_path), normal_img)

    # 2. Pure blank white image (std == 0)
    pure_blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    pure_blank_path = imgs_dir / "pure_blank.png"
    cv2.imwrite(str(pure_blank_path), pure_blank)

    # 3. Nearly blank image with tiny single pixel/noise (std < 2.0)
    low_std = np.full((100, 100, 3), 255, dtype=np.uint8)
    low_std[10, 10] = [250, 250, 250]
    low_std_path = imgs_dir / "low_std.png"
    cv2.imwrite(str(low_std_path), low_std)

    # 4. Sparse edge image on extreme background
    sparse_edge = np.full((100, 100, 3), 254, dtype=np.uint8)
    sparse_edge[50:52, 50:52] = [200, 200, 200]
    sparse_path = imgs_dir / "sparse_edge.png"
    cv2.imwrite(str(sparse_path), sparse_edge)

    # 5. Corrupted image file
    corrupt_path = imgs_dir / "corrupted.png"
    corrupt_path.write_bytes(b"NOT_A_VALID_IMAGE_DATA")

    return {
        "normal": normal_path,
        "pure_blank": pure_blank_path,
        "low_std": low_std_path,
        "sparse_edge": sparse_path,
        "corrupted": corrupt_path,
    }


def test_is_blank_or_nearly_blank(synthetic_images):
    """Verify detection of pure blank, low std, and sparse-edge nearly blank screenshots."""
    # Pure blank
    is_blank, reason, metrics = is_blank_or_nearly_blank(synthetic_images["pure_blank"])
    assert is_blank is True
    assert reason == "pure_blank_image"
    assert metrics["std_brightness"] == 0.0

    # Low standard deviation
    is_blank, reason, metrics = is_blank_or_nearly_blank(synthetic_images["low_std"])
    assert is_blank is True
    assert reason == "nearly_blank_low_std"
    assert metrics["std_brightness"] < 2.0

    # Normal rich image
    is_blank, reason, metrics = is_blank_or_nearly_blank(synthetic_images["normal"])
    assert is_blank is False
    assert reason is None
    assert metrics["std_brightness"] > 10.0

    # Corrupted image file
    is_blank, reason, metrics = is_blank_or_nearly_blank(synthetic_images["corrupted"])
    assert is_blank is True
    assert reason == "corrupt_image"


def test_identify_cross_class_domains():
    """Verify detection of domains appearing in both phishing and legitimate classes."""
    entries = [
        {"registered_domain": "github.io", "threat_category": "legitimate"},
        {"registered_domain": "github.io", "threat_category": "phishing"},
        {"registered_domain": "paypal.com", "threat_category": "legitimate"},
        {"registered_domain": "evil-phish.com", "threat_category": "phishing"},
    ]
    conflicts = identify_cross_class_domains(entries)
    assert conflicts == {"github.io"}


def test_identify_cross_class_screenshot_hashes():
    """Verify detection of identical screenshot SHA-256 hashes used across conflicting classes."""
    entries = [
        {"screenshot_sha256": "hash_a", "threat_category": "phishing"},
        {"screenshot_sha256": "hash_a", "threat_category": "unverified_candidate"},
        {"screenshot_sha256": "hash_b", "threat_category": "phishing"},
        {"screenshot_sha256": "hash_b", "threat_category": "phishing"},
        {"screenshot_sha256": "hash_c", "threat_category": "unverified_candidate"},
    ]
    conflicts = identify_cross_class_screenshot_hashes(entries)
    assert conflicts == {"hash_a"}


def test_clean_visual_dataset_quarantine_rules(synthetic_images):
    """Verify clean_visual_dataset quarantines duplicates, conflicts, known domains, and blanks."""
    manifest_entries = [
        # 1. Valid rich phishing sample
        {
            "csv_line_num": 1,
            "original_url": "https://secure-login.org",
            "registered_domain": "secure-login.org",
            "original_label": "phishing",
            "threat_category": THREAT_CATEGORY_PHISHING,
            "label_provenance": "phishtank_verified",
            "screenshot_path": str(synthetic_images["normal"]),
            "screenshot_sha256": "sha_normal_1",
        },
        # 2. Duplicate hash of sample 1 (should be excluded)
        {
            "csv_line_num": 2,
            "original_url": "https://secure-login.org/dup",
            "registered_domain": "other-dup.org",
            "original_label": "phishing",
            "threat_category": THREAT_CATEGORY_PHISHING,
            "label_provenance": "phishtank_verified",
            "screenshot_path": str(synthetic_images["normal"]),
            "screenshot_sha256": "sha_normal_1",
        },
        # 3. Pure blank screenshot (should be excluded)
        {
            "csv_line_num": 3,
            "original_url": "https://blank-site.com",
            "registered_domain": "blank-site.com",
            "original_label": "legitimate",
            "threat_category": THREAT_CATEGORY_UNVERIFIED,
            "label_provenance": "tranco_candidate_unverified",
            "screenshot_path": str(synthetic_images["pure_blank"]),
            "screenshot_sha256": "sha_blank_1",
        },
        # 4. Known conflicting domain kakaku.com (should be excluded)
        {
            "csv_line_num": 4,
            "original_url": "https://kakaku.com/deal",
            "registered_domain": "kakaku.com",
            "original_label": "phishing",
            "threat_category": THREAT_CATEGORY_PHISHING,
            "label_provenance": "phishtank_verified",
            "screenshot_path": str(synthetic_images["normal"]),
            "screenshot_sha256": "sha_kakaku",
        },
        # 5 & 6: Cross-class domain conflict (github.io in both classes)
        {
            "csv_line_num": 5,
            "original_url": "https://github.io",
            "registered_domain": "github.io",
            "original_label": "legitimate",
            "threat_category": THREAT_CATEGORY_UNVERIFIED,
            "label_provenance": "tranco_candidate_unverified",
            "screenshot_path": str(synthetic_images["normal"]),
            "screenshot_sha256": "sha_gh_legit",
        },
        {
            "csv_line_num": 6,
            "original_url": "https://phish.github.io",
            "registered_domain": "github.io",
            "original_label": "phishing",
            "threat_category": THREAT_CATEGORY_PHISHING,
            "label_provenance": "phishtank_verified",
            "screenshot_path": str(synthetic_images["normal"]),
            "screenshot_sha256": "sha_gh_phish",
        },
    ]

    accepted, excluded, summary = clean_visual_dataset(manifest_entries)

    # Only sample 1 should be accepted
    assert len(accepted) == 1
    assert accepted[0]["csv_line_num"] == 1
    assert accepted[0]["threat_category"] == THREAT_CATEGORY_PHISHING

    # 5 samples excluded
    assert len(excluded) == 5
    excluded_reasons = {e["csv_line_num"]: e["exclusion_reason"] for e in excluded}

    assert excluded_reasons[2] == "duplicate_screenshot_hash"
    assert excluded_reasons[3] == "pure_blank_image"
    assert excluded_reasons[4] == "known_conflict_domain"
    assert excluded_reasons[5] == "cross_class_domain_conflict"
    assert excluded_reasons[6] == "cross_class_domain_conflict"

    # Verify Tranco provenance is preserved
    assert summary["total_manifest_records"] == 6
    assert summary["total_accepted_records"] == 1
    assert summary["total_excluded_records"] == 5


def test_extract_features_batch_and_validation(tmp_path, synthetic_images):
    """Verify batch extraction produces all 30 features, no NaN/inf, and correct CSV format."""
    samples = [
        {
            "csv_line_num": 10,
            "original_url": "https://legit-site.org",
            "normalized_url": "https://legit-site.org",
            "registered_domain": "legit-site.org",
            "original_label": "legitimate",
            "threat_category": THREAT_CATEGORY_UNVERIFIED,
            "label_provenance": "tranco_candidate_unverified",
            "screenshot_path": str(synthetic_images["normal"]),
            "screenshot_sha256": "hash_sample_10",
        }
    ]

    out_csv = tmp_path / "test_visual_features.csv"
    extracted, stats = extract_features_batch(samples, output_csv_path=out_csv)

    assert len(extracted) == 1
    rec = extracted[0]

    # Verify all 30 visual features are present
    for feat_name in VISUAL_FEATURE_NAMES:
        assert feat_name in rec
        assert isinstance(rec[feat_name], (int, float))
        assert not math.isnan(rec[feat_name])
        assert not math.isinf(rec[feat_name])

    # Verify metadata fields preserved
    assert rec["threat_category"] == THREAT_CATEGORY_UNVERIFIED
    assert rec["label_provenance"] == "tranco_candidate_unverified"
    assert rec["screenshot_sha256"] == "hash_sample_10"

    # Verify CSV file output
    assert os.path.isfile(out_csv)
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert len(row) == len(CSV_FIELDNAMES)
    assert row["width"] == "100"
    assert row["height"] == "100"

    # Test validation function
    val_result = validate_processed_visual_dataset(extracted, csv_path=out_csv)
    assert val_result["is_valid"] is True
    assert val_result["all_records_have_30_features"] is True
    assert val_result["nan_value_count"] == 0
    assert val_result["inf_value_count"] == 0
    assert val_result["duplicate_screenshot_hashes_count"] == 0
    assert val_result["unresolved_cross_class_domains_count"] == 0
    assert val_result["quarantined_samples_count"] == 0


def test_batch_extraction_failure_isolation(tmp_path, synthetic_images):
    """Verify that an individual corrupted image failure does not terminate the batch."""
    samples = [
        {
            "csv_line_num": 1,
            "original_url": "https://corrupted-page.com",
            "normalized_url": "https://corrupted-page.com",
            "registered_domain": "corrupted-page.com",
            "original_label": "phishing",
            "threat_category": THREAT_CATEGORY_PHISHING,
            "label_provenance": "phishtank_verified",
            "screenshot_path": str(synthetic_images["corrupted"]),
            "screenshot_sha256": "hash_corrupt",
        },
        {
            "csv_line_num": 2,
            "original_url": "https://valid-page.com",
            "normalized_url": "https://valid-page.com",
            "registered_domain": "valid-page.com",
            "original_label": "phishing",
            "threat_category": THREAT_CATEGORY_PHISHING,
            "label_provenance": "phishtank_verified",
            "screenshot_path": str(synthetic_images["normal"]),
            "screenshot_sha256": "hash_valid",
        },
    ]

    out_csv = tmp_path / "isolation_test.csv"
    extracted, stats = extract_features_batch(samples, output_csv_path=out_csv)

    # Corrupt sample failed, valid sample succeeded
    assert len(extracted) == 1
    assert extracted[0]["csv_line_num"] == 2
    assert stats["failed_extractions_count"] == 1
    assert stats["successfully_extracted"] == 1


def test_end_to_end_cleaning_pipeline_dry_run(tmp_path, synthetic_images):
    """Verify end-to-end run_cleaning_and_feature_extraction dry run."""
    manifest_path = tmp_path / "test_manifest.jsonl"
    entries = [
        {
            "csv_line_num": 1,
            "original_url": "https://sample1.com",
            "normalized_url": "https://sample1.com",
            "registered_domain": "sample1.com",
            "original_label": "phishing",
            "threat_category": THREAT_CATEGORY_PHISHING,
            "label_provenance": "phishtank_verified",
            "screenshot_path": str(synthetic_images["normal"]),
            "screenshot_sha256": "sha_1",
        }
    ]
    with open(manifest_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    out_csv = tmp_path / "out.csv"
    report_json = tmp_path / "report.json"

    accepted_cnt, excluded_cnt, summary = run_cleaning_and_feature_extraction(
        manifest_path=manifest_path,
        output_csv_path=out_csv,
        report_path=report_json,
        dry_run=True,
    )

    assert accepted_cnt == 1
    assert excluded_cnt == 0
    # In dry-run mode, no output files written
    assert not os.path.exists(out_csv)
    assert not os.path.exists(report_json)

