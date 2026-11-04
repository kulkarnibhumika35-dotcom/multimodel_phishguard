"""
Unit tests for PhishGuard Dataset Validation & Provenance (ml.training.validate_dataset).

Validates:
- Missing value detection (URLs, labels)
- Duplicate URL detection and contradictory label identification
- Invalid label identification
- Class distribution calculation and imbalance warnings
- Training readiness thresholds
- Provenance manifest generation, loading, and SHA-256 computation
- Error handling on missing files or invalid headers
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import pytest

from ml.training.validate_dataset import (
    compute_file_sha256,
    load_dataset_manifest,
    save_dataset_manifest,
    validate_dataset,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _write_csv(path: str, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def clean_dataset_path(tmp_path) -> str:
    """Create a balanced, clean dataset meeting all training requirements."""
    csv_file = str(tmp_path / "clean_dataset.csv")
    rows = []
    # 15 legitimate
    for i in range(15):
        rows.append([f"https://legit-{i}.com/page", "0"])
    # 15 phishing
    for i in range(15):
        rows.append([f"http://phish-{i}.xyz/login", "1"])
    _write_csv(csv_file, ["url", "label"], rows)
    return csv_file


# ── Tests: Basic Validation & Happy Path ────────────────────────────

def test_validate_clean_dataset(clean_dataset_path: str):
    """Test validation of a fully compliant dataset."""
    report = validate_dataset(clean_dataset_path)

    assert report["total_rows"] == 30
    assert report["valid_rows"] == 30
    assert report["unique_valid_urls"] == 30
    assert report["missing_urls"] == 0
    assert report["missing_labels"] == 0
    assert report["invalid_labels"] == 0
    assert report["duplicate_urls"] == 0
    assert report["conflicting_labels"] == 0

    cd = report["class_distribution"]
    assert cd["legitimate"] == 15
    assert cd["phishing"] == 15
    assert cd["legitimate_pct"] == 50.0
    assert cd["phishing_pct"] == 50.0

    assert report["is_valid_for_training"] is True
    assert len(report["warnings"]) == 0


def test_custom_column_names(tmp_path):
    """Test validation with custom column headers."""
    csv_file = str(tmp_path / "custom_cols.csv")
    rows = [
        ["https://google.com", "legitimate"],
        ["http://bad.tk", "phishing"],
    ]
    _write_csv(csv_file, ["website_address", "threat_tag"], rows)

    report = validate_dataset(
        csv_file,
        url_column="website_address",
        label_column="threat_tag",
    )
    assert report["total_rows"] == 2
    assert report["valid_rows"] == 2


# ── Tests: Missing Values ───────────────────────────────────────────

def test_detects_missing_urls(tmp_path):
    """Test detection of blank or whitespace URLs."""
    csv_file = str(tmp_path / "missing_urls.csv")
    rows = [
        ["https://valid.com", "0"],
        ["", "1"],
        ["   ", "0"],
        ["https://another.com", "1"],
    ]
    _write_csv(csv_file, ["url", "label"], rows)

    report = validate_dataset(csv_file)
    assert report["total_rows"] == 4
    assert report["missing_urls"] == 2
    assert report["valid_rows"] == 2


def test_detects_missing_labels(tmp_path):
    """Test detection of blank or missing labels."""
    csv_file = str(tmp_path / "missing_labels.csv")
    rows = [
        ["https://valid.com", "0"],
        ["https://no-label.com", ""],
        ["https://spaces.com", "   "],
    ]
    _write_csv(csv_file, ["url", "label"], rows)

    report = validate_dataset(csv_file)
    assert report["total_rows"] == 3
    assert report["missing_labels"] == 2
    assert report["valid_rows"] == 1


# ── Tests: Invalid Labels ───────────────────────────────────────────

def test_detects_invalid_labels(tmp_path):
    """Test detection of unsupported or arbitrary label text."""
    csv_file = str(tmp_path / "invalid_labels.csv")
    rows = [
        ["https://a.com", "0"],
        ["https://b.com", "legitimate"],
        ["https://c.com", "unknown_category"],
        ["https://d.com", "2"],
        ["https://e.com", "99"],
    ]
    _write_csv(csv_file, ["url", "label"], rows)

    report = validate_dataset(csv_file)
    assert report["total_rows"] == 5
    assert report["invalid_labels"] == 3
    assert len(report["invalid_label_samples"]) == 3
    assert "unknown_category" in report["invalid_label_samples"]


# ── Tests: Duplicates & Conflicting Labels ───────────────────────────

def test_detects_duplicate_urls(tmp_path):
    """Test detection and counting of duplicate URLs."""
    csv_file = str(tmp_path / "duplicates.csv")
    rows = [
        ["https://repeat.com", "0"],
        ["https://repeat.com", "0"],
        ["https://repeat.com", "0"],
        ["https://unique.com", "1"],
    ]
    _write_csv(csv_file, ["url", "label"], rows)

    report = validate_dataset(csv_file)
    assert report["total_rows"] == 4
    assert report["valid_rows"] == 4
    assert report["unique_valid_urls"] == 2
    assert report["duplicate_urls"] == 2
    assert any("duplicate" in w.lower() for w in report["warnings"])


def test_detects_conflicting_labels(tmp_path):
    """Test critical integrity detection for contradictory labels."""
    csv_file = str(tmp_path / "conflict.csv")
    rows = [
        ["https://contradictory.com", "0"],
        ["https://contradictory.com", "1"],  # Same URL labeled as phishing!
        ["https://safe.com", "0"],
    ]
    _write_csv(csv_file, ["url", "label"], rows)

    report = validate_dataset(csv_file)
    assert report["conflicting_labels"] == 1
    assert report["is_valid_for_training"] is False
    assert any("conflicting" in w.lower() for w in report["warnings"])


# ── Tests: Class Distribution & Warnings ────────────────────────────

def test_severe_class_imbalance_warning(tmp_path):
    """Test warning when one class exceeds the imbalance threshold."""
    csv_file = str(tmp_path / "imbalance.csv")
    rows = []
    # 28 legitimate
    for i in range(28):
        rows.append([f"https://legit-{i}.com", "0"])
    # 2 phishing (less than 10%)
    for i in range(2):
        rows.append([f"http://phish-{i}.com", "1"])
    _write_csv(csv_file, ["url", "label"], rows)

    report = validate_dataset(csv_file)
    assert any("imbalance" in w.lower() for w in report["warnings"])
    assert report["class_distribution"]["legitimate_pct"] > 90.0


def test_minimum_sample_size_requirement(tmp_path):
    """Test that fewer than 20 samples flags is_valid_for_training as False."""
    csv_file = str(tmp_path / "too_few.csv")
    rows = [
        ["https://a.com", "0"],
        ["https://b.com", "1"],
    ]
    _write_csv(csv_file, ["url", "label"], rows)

    report = validate_dataset(csv_file)
    assert report["is_valid_for_training"] is False
    assert any("minimum 20" in w.lower() for w in report["warnings"])


# ── Tests: Provenance Manifest Management ───────────────────────────

def test_provenance_manifest_lifecycle(clean_dataset_path: str, tmp_path):
    """Test creation, loading, and verification of dataset provenance metadata."""
    manifest = save_dataset_manifest(
        csv_path=clean_dataset_path,
        source_name="PhishTank Community Feed",
        collection_date="2026-11-04",
        license_info="PhishTank Developer License",
        description="Verified online phishing URLs",
    )

    assert manifest["source_name"] == "PhishTank Community Feed"
    assert manifest["collection_date"] == "2026-11-04"
    assert len(manifest["sha256"]) == 64  # Valid SHA-256 hex string
    assert manifest["file_size_bytes"] > 0

    # Test loading
    loaded = load_dataset_manifest(clean_dataset_path)
    assert loaded is not None
    assert loaded["sha256"] == manifest["sha256"]

    # Validate dataset includes provenance
    report = validate_dataset(clean_dataset_path)
    assert report["provenance"] is not None
    assert report["provenance"]["source_name"] == "PhishTank Community Feed"


def test_compute_file_sha256(tmp_path):
    """Test deterministic file checksum."""
    test_file = str(tmp_path / "test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("PhishGuard Security Audit")

    sha1 = compute_file_sha256(test_file)
    sha2 = compute_file_sha256(test_file)
    assert sha1 == sha2
    assert len(sha1) == 64


# ── Tests: Error Handling ───────────────────────────────────────────

def test_file_not_found():
    """Test missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        validate_dataset("non_existent_file.csv")


def test_missing_column_raises(tmp_path):
    """Test missing required column raises ValueError."""
    csv_file1 = str(tmp_path / "missing_url_col.csv")
    _write_csv(csv_file1, ["label"], [["0"]])
    with pytest.raises(ValueError, match="URL column"):
        validate_dataset(csv_file1)

    csv_file2 = str(tmp_path / "missing_label_col.csv")
    _write_csv(csv_file2, ["url"], [["https://test.com"]])
    with pytest.raises(ValueError, match="Label column"):
        validate_dataset(csv_file2)


def test_empty_file_raises(tmp_path):
    """Test completely empty file raises ValueError."""
    empty_file = str(tmp_path / "empty.csv")
    with open(empty_file, "w") as f:
        f.write("")

    with pytest.raises(ValueError, match="empty or has no header"):
        validate_dataset(empty_file)
