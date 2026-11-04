"""
Unit tests for PhishGuard Raw Dataset Ingestion & Inspection (ml.training.raw_sources).

Covers:
- Reading headerless Tranco CSVs with rank and domain.
- Enforcing that Tranco domains remain UNLABELED candidates (label=None).
- Filtering PhishTank records to only verified == 'yes' and online == 'yes'.
- Extracting the url column as the phishing URL candidate.
- Normalizing URLs and removing duplicates safely.
- Preserving source provenance and licensing metadata.
- Error handling on missing files or invalid headers.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import pytest

from ml.training.raw_sources import (
    load_phishtank_records,
    load_tranco_records,
    load_kaggle_records,
    load_phiusiil_records,
    normalize_url,
    inspect_raw_dataset_sources,
    audit_dataset_quality,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _write_csv(path: str, header: list[str] | None, rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header is not None:
            w.writerow(header)
        w.writerows(rows)


# ── Tests: URL Normalization ────────────────────────────────────────

def test_normalize_url():
    """Test URL normalization utility."""
    assert normalize_url("https://example.com") == "https://example.com"
    assert normalize_url("  http://example.com/login  ") == "http://example.com/login"
    assert normalize_url("example.com/path") == "https://example.com/path"
    assert normalize_url("") is None
    assert normalize_url("   ") is None
    assert normalize_url(None) is None


# ── Tests: PhishTank Ingestion ──────────────────────────────────────

def test_phishtank_filters_verified_and_online(tmp_path):
    """Requirement 3: Retain only records where verified == yes and online == yes."""
    csv_file = str(tmp_path / "phishtank_sample.csv")
    header = [
        "phish_id", "url", "phish_detail_url", "submission_time",
        "verified", "verification_time", "online", "target"
    ]
    rows = [
        # 1. Valid: verified=yes, online=yes
        ["101", "https://phish1.com/login", "http://pt/101", "2026-11-04", "yes", "2026-11-04", "yes", "PayPal"],
        # 2. Invalid: verified=no, online=yes
        ["102", "https://phish2.com/account", "http://pt/102", "2026-11-04", "no", "2026-11-04", "yes", "eBay"],
        # 3. Invalid: verified=yes, online=no
        ["103", "https://phish3.com/verify", "http://pt/103", "2026-11-04", "yes", "2026-11-04", "no", "Apple"],
        # 4. Valid: verified=YES (case-insensitive), online=YES
        ["104", "http://phish4.com/secure", "http://pt/104", "2026-11-04", "YES", "2026-11-04", "YES", "Microsoft"],
        # 5. Invalid: missing url
        ["105", "", "http://pt/105", "2026-11-04", "yes", "2026-11-04", "yes", "Other"],
    ]
    _write_csv(csv_file, header, rows)

    candidates, stats = load_phishtank_records(csv_file)

    assert stats["total_rows"] == 5
    assert stats["unverified_dropped"] == 1
    assert stats["offline_dropped"] == 1
    assert stats["missing_urls"] == 1
    assert stats["retained_candidates"] == 2
    assert len(candidates) == 2

    assert candidates[0]["phish_id"] == "101"
    assert candidates[0]["url"] == "https://phish1.com/login"
    assert candidates[0]["label"] == 1
    assert candidates[0]["source"] == "PhishTank"
    assert "Developer License" in candidates[0]["license"]

    assert candidates[1]["phish_id"] == "104"
    assert candidates[1]["url"] == "http://phish4.com/secure"


def test_phishtank_deduplication(tmp_path):
    """Requirement 4: Remove duplicate URLs safely."""
    csv_file = str(tmp_path / "phishtank_dups.csv")
    header = [
        "phish_id", "url", "phish_detail_url", "submission_time",
        "verified", "verification_time", "online", "target"
    ]
    rows = [
        ["1", "https://duplicate-url.com/login", "http://pt/1", "2026-11-04", "yes", "2026-11-04", "yes", "Target"],
        ["2", "https://duplicate-url.com/login", "http://pt/2", "2026-11-04", "yes", "2026-11-04", "yes", "Target"],
        ["3", "https://unique-url.com/account", "http://pt/3", "2026-11-04", "yes", "2026-11-04", "yes", "Target"],
    ]
    _write_csv(csv_file, header, rows)

    candidates, stats = load_phishtank_records(csv_file)

    assert stats["total_rows"] == 3
    assert stats["duplicates_dropped"] == 1
    assert stats["retained_candidates"] == 2
    assert len(candidates) == 2


def test_phishtank_missing_columns_raises(tmp_path):
    """Test error handling when required columns are absent."""
    csv_file = str(tmp_path / "bad_phish.csv")
    _write_csv(csv_file, ["phish_id", "link"], [["1", "https://a.com"]])

    with pytest.raises(ValueError, match="Missing required PhishTank columns"):
        load_phishtank_records(csv_file)


# ── Tests: Tranco Ingestion ─────────────────────────────────────────

def test_tranco_headerless_parsing(tmp_path):
    """Requirement 2: Correctly read Tranco CSV as headerless rank,domain."""
    csv_file = str(tmp_path / "tranco_sample.csv")
    rows = [
        ["1", "google.com"],
        ["2", "cloudflare.com"],
        ["3", "facebook.com"],
    ]
    _write_csv(csv_file, None, rows)

    candidates, stats = load_tranco_records(csv_file)

    assert stats["total_rows"] == 3
    assert stats["missing_domains"] == 0
    assert stats["missing_ranks"] == 0
    assert stats["duplicates_dropped"] == 0
    assert stats["retained_candidates"] == 3
    assert len(candidates) == 3

    assert candidates[0]["rank"] == 1
    assert candidates[0]["domain"] == "google.com"
    assert candidates[0]["url"] == "https://google.com"


def test_tranco_never_automatically_labeled_as_legitimate(tmp_path):
    """Requirement 5: Do NOT automatically label Tranco domains as legitimate!

    They must be kept as unlabeled candidates (label=None).
    """
    csv_file = str(tmp_path / "tranco_unlabeled.csv")
    rows = [
        ["1", "google.com"],
        ["2", "example.org"],
    ]
    _write_csv(csv_file, None, rows)

    candidates, _ = load_tranco_records(csv_file)

    for rec in candidates:
        assert rec["label"] is None, (
            f"Tranco record {rec['domain']} must have label=None, got {rec['label']}"
        )
        assert rec["source"] == "Tranco"
        assert rec["license"] == "CC BY 4.0"


def test_tranco_deduplication(tmp_path):
    """Requirement 4: Normalize and remove duplicate domains safely."""
    csv_file = str(tmp_path / "tranco_dups.csv")
    rows = [
        ["1", "google.com"],
        ["10", "GOOGLE.COM"],      # Duplicate case variation
        ["50", "google.com/"],     # Duplicate trailing slash
        ["2", "amazon.com"],
    ]
    _write_csv(csv_file, None, rows)

    candidates, stats = load_tranco_records(csv_file)

    assert stats["total_rows"] == 4
    assert stats["duplicates_dropped"] == 2
    assert stats["retained_candidates"] == 2
    assert len(candidates) == 2


def test_tranco_missing_fields_handling(tmp_path):
    """Test handling of blank or malformed lines in Tranco."""
    csv_file = str(tmp_path / "tranco_malformed.csv")
    rows = [
        ["1", "valid.com"],
        ["2", ""],              # Missing domain
        ["not_a_num", "abc.com"], # Missing numeric rank
        ["", ""],               # Empty
    ]
    _write_csv(csv_file, None, rows)

    candidates, stats = load_tranco_records(csv_file)

    assert stats["total_rows"] == 4
    assert stats["missing_domains"] == 2
    assert stats["missing_ranks"] == 2
    assert stats["retained_candidates"] == 2


# ── Tests: Inspection Functionality ─────────────────────────────────

def test_inspect_raw_sources(tmp_path):
    """Requirement 7: Generate concise inspection report showing row counts, duplicates, etc."""
    phish_csv = str(tmp_path / "phish.csv")
    header = [
        "phish_id", "url", "phish_detail_url", "submission_time",
        "verified", "verification_time", "online", "target"
    ]
    rows_p = [
        ["1", "https://phish.com", "http://pt/1", "2026-11-04", "yes", "2026-11-04", "yes", "Target"],
    ]
    _write_csv(phish_csv, header, rows_p)

    tranco_csv = str(tmp_path / "tranco.csv")
    rows_t = [
        ["1", "example.com"],
    ]
    _write_csv(tranco_csv, None, rows_t)

    report = inspect_raw_dataset_sources(
        phish_path=phish_csv,
        tranco_path=tranco_csv,
        save_metadata=True,
    )

    assert "phishtank" in report["sources"]
    assert "tranco" in report["sources"]

    pt = report["sources"]["phishtank"]
    assert pt["total_rows"] == 1
    assert pt["retained_candidates"] == 1

    tr = report["sources"]["tranco"]
    assert tr["total_rows"] == 1
    assert tr["retained_candidates"] == 1

    # Verify metadata files were created without modifying original CSVs
    assert os.path.isfile(f"{phish_csv}.meta.json")
    assert os.path.isfile(f"{tranco_csv}.meta.json")


def test_inspect_raw_sources_missing_files():
    """Test handling of nonexistent files in inspect_raw_dataset_sources."""
    report = inspect_raw_dataset_sources(
        phish_path="nonexistent_phish.csv",
        tranco_path="nonexistent_tranco.csv",
        save_metadata=False,
    )
    assert "error" in report["sources"]["phishtank"]
    assert "error" in report["sources"]["tranco"]

# ── Tests: Kaggle Records & Disambiguation ──────────────────────────

def test_kaggle_records_parsing_and_threat_disambiguation(tmp_path):
    """Requirement 7 & 8: Keep URLhaus malware distinct from phishing and Tranco unverified."""
    csv_file = str(tmp_path / "kaggle_sample.csv")
    header = ["url", "label", "source"]
    rows = [
        ["http://192.168.1.1/malware.sh", "1", "urlhaus"],
        ["https://evil-phish.com/login", "1", "openphish"],
        ["https://popular-bank.com", "0", "tranco"],
    ]
    _write_csv(csv_file, header, rows)

    candidates, stats = load_kaggle_records(csv_file)

    assert stats["total_rows"] == 3
    assert stats["retained_candidates"] == 3
    assert stats["threat_type_breakdown"]["malware_distribution"] == 1
    assert stats["threat_type_breakdown"]["phishing"] == 1
    assert stats["threat_type_breakdown"]["unlabeled_candidate"] == 1

    by_src = {c["source"]: c for c in candidates}
    assert by_src["urlhaus"]["threat_type"] == "malware_distribution"
    assert by_src["urlhaus"]["is_verified_legitimate"] is False

    assert by_src["openphish"]["threat_type"] == "phishing"
    assert by_src["openphish"]["is_verified_legitimate"] is False

    assert by_src["tranco"]["threat_type"] == "unlabeled_candidate"
    assert by_src["tranco"]["is_verified_legitimate"] is False


def test_kaggle_deduplication(tmp_path):
    """Test deduplication in Kaggle dataset."""
    csv_file = str(tmp_path / "kaggle_dups.csv")
    header = ["url", "label", "source"]
    rows = [
        ["https://duplicate.com/page", "1", "openphish"],
        ["https://duplicate.com/page", "1", "openphish"],
        ["https://unique.com", "0", "tranco"],
    ]
    _write_csv(csv_file, header, rows)

    candidates, stats = load_kaggle_records(csv_file)

    assert stats["total_rows"] == 3
    assert stats["duplicates_dropped"] == 1
    assert stats["retained_candidates"] == 2


def test_kaggle_missing_columns_raises(tmp_path):
    """Test ValueError when required columns are missing in Kaggle CSV."""
    csv_file = str(tmp_path / "kaggle_invalid.csv")
    header = ["wrong_url", "wrong_label"]
    rows = [["https://test.com", "1"]]
    _write_csv(csv_file, header, rows)

    with pytest.raises(ValueError, match="(?i)missing required.*columns"):
        load_kaggle_records(csv_file)


# ── Tests: Quality & Provenance Audit ────────────────────────────────

def test_audit_dataset_quality_detects_conflicts_and_leakage(tmp_path):
    """Requirement 1, 2, 5, 6: Cross-dataset audit detects conflicts, domain leakage and validates 20 features."""
    phish_csv = str(tmp_path / "phish.csv")
    header_p = [
        "phish_id", "url", "phish_detail_url", "submission_time",
        "verified", "verification_time", "online", "target"
    ]
    rows_p = [
        ["1", "https://conflict-site.com/login", "http://pt/1", "2026-11-04", "yes", "2026-11-04", "yes", "Target"],
        ["2", "https://shared-host.com/account", "http://pt/2", "2026-11-04", "yes", "2026-11-04", "yes", "Target"],
    ]
    _write_csv(phish_csv, header_p, rows_p)

    tranco_csv = str(tmp_path / "tranco.csv")
    rows_t = [
        ["1", "shared-host.com"],
        ["2", "benign-service.org"],
    ]
    _write_csv(tranco_csv, None, rows_t)

    kaggle_csv = str(tmp_path / "kaggle.csv")
    header_k = ["url", "label", "source"]
    rows_k = [
        ["https://conflict-site.com/login", "0", "tranco"],
        ["http://10.0.0.1/malware.bin", "1", "urlhaus"],
    ]
    _write_csv(kaggle_csv, header_k, rows_k)

    audit = audit_dataset_quality(
        phish_path=phish_csv,
        tranco_path=tranco_csv,
        kaggle_path=kaggle_csv,
        sample_feature_test_size=5,
    )

    cda = audit["cross_dataset_audit"]
    assert cda["exact_url_overlap_pt_kaggle"] == 1
    assert cda["conflicting_labels_count"] == 1
    assert cda["conflicting_labels"][0]["phishtank_url"] == "https://conflict-site.com/login"

    # Domain leakage: conflict-site.com is in both PhishTank and Kaggle Tranco
    leakage = cda["domain_leakage"]
    assert leakage["shared_hostnames_phishtank_and_tranco_candidates"] >= 1
    assert "conflict-site.com" in leakage["sample_shared_hostnames_phish_tranco"]

    # 20 URL feature extraction verification test
    fe = cda["feature_extraction_verification"]
    assert fe["all_features_compatible"] is True
    assert sum(fe["extraction_errors"].values()) == 0

# ── Tests: PhiUSIIL Records & Label Mapping ─────────────────────────

def test_phiusiil_records_parsing_and_label_mapping(tmp_path):
    """Requirements 1, 2, 3, 5: Read only URL/label, map 1->0, 0->1, attribute as phiusiil."""
    csv_file = str(tmp_path / "phi_sample.csv")
    # Include extra dummy columns out of 56
    header = ["FILENAME", "URL", "URLLength", "Domain", "ExtraFeature54", "label"]
    rows = [
        ["f1.txt", "https://legit-site.org/index", "25", "legit-site.org", "99", "1"],  # raw 1 -> legit (0)
        ["f2.txt", "http://evil-phish.net/login", "24", "evil-phish.net", "42", "0"],   # raw 0 -> phish (1)
    ]
    _write_csv(csv_file, header, rows)

    candidates, stats = load_phiusiil_records(csv_file)

    assert stats["total_rows"] == 2
    assert stats["retained_candidates"] == 2
    assert stats["threat_type_breakdown"]["legitimate"] == 1
    assert stats["threat_type_breakdown"]["phishing"] == 1

    by_cat = {c["threat_type"]: c for c in candidates}
    # Raw label 1 -> PhishGuard 0 (legitimate)
    assert by_cat["legitimate"]["label"] == 0
    assert by_cat["legitimate"]["source"] == "phiusiil"
    assert by_cat["legitimate"]["is_verified_legitimate"] is True

    # Raw label 0 -> PhishGuard 1 (phishing)
    assert by_cat["phishing"]["label"] == 1
    assert by_cat["phishing"]["source"] == "phiusiil"
    assert by_cat["phishing"]["is_verified_legitimate"] is False


def test_phiusiil_deduplication(tmp_path):
    """Test deduplication within PhiUSIIL dataset."""
    csv_file = str(tmp_path / "phi_dups.csv")
    header = ["URL", "label"]
    rows = [
        ["https://repeat-url.com/path", "1"],
        ["https://repeat-url.com/path", "1"],
        ["https://unique-site.edu", "1"],
    ]
    _write_csv(csv_file, header, rows)

    candidates, stats = load_phiusiil_records(csv_file)

    assert stats["total_rows"] == 3
    assert stats["duplicates_dropped"] == 1
    assert stats["retained_candidates"] == 2


def test_phiusiil_missing_columns_raises(tmp_path):
    """Test ValueError when required columns (URL, label) are missing."""
    csv_file = str(tmp_path / "phi_invalid.csv")
    header = ["WrongCol1", "WrongCol2"]
    rows = [["https://test.com", "1"]]
    _write_csv(csv_file, header, rows)

    with pytest.raises(ValueError, match="(?i)missing required.*columns"):
        load_phiusiil_records(csv_file)
