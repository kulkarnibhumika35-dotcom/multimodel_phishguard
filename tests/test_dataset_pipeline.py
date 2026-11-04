"""
Unit tests for PhishGuard Reproducible Dataset Preparation Pipeline (ml.training.dataset_pipeline).

Validates:
1. Conflict quarantine for conflicting URLs (kakaku.com, pokeapi.co).
2. Deduplication via consistent canonical normalization.
3. Preservation of 3 threat categories: legitimate, phishing, and malware distribution.
4. Tranco unverified candidates never labeled as legitimate (label=0).
5. Explicit limitation reported when verified legitimate samples are insufficient.
6. Zero hostname leakage using registered-domain grouping.
7. Creation of summary JSON/MD and processed CSV outputs.
"""

from __future__ import annotations

import csv
import json
import os
import pytest

from ml.features.url_features import FEATURE_NAMES
from ml.training.dataset_pipeline import (
    THREAT_CATEGORY_LEGITIMATE,
    THREAT_CATEGORY_MALWARE,
    THREAT_CATEGORY_PHISHING,
    THREAT_CATEGORY_UNVERIFIED,
    canonical_normalize_url,
    extract_registered_domain,
    run_dataset_pipeline,
    split_by_registered_domain,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _write_csv(path: str, header: list[str] | None, rows: list[list[str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header is not None:
            w.writerow(header)
        w.writerows(rows)


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── 1. Domain Extraction & Normalization ────────────────────────────

class TestDomainAndNormalization:
    def test_extract_registered_domain_subdomains(self):
        assert extract_registered_domain("https://sub.example.com/page") == "example.com"
        assert extract_registered_domain("http://a.b.c.google.com") == "google.com"

    def test_extract_registered_domain_two_part_tlds(self):
        assert extract_registered_domain("https://banking.service.co.uk/login") == "service.co.uk"
        assert extract_registered_domain("http://portal.gov.au/home") == "portal.gov.au"
        assert extract_registered_domain("https://shop.co.jp") == "shop.co.jp"

    def test_extract_registered_domain_ip_and_ports(self):
        assert extract_registered_domain("http://192.168.1.1:8080/test") == "192.168.1.1"
        assert extract_registered_domain("http://[2001:db8::1]/path") == "2001:db8::1"

    def test_canonical_normalize_url_strips_trailing_slash_and_fragments(self):
        assert canonical_normalize_url("https://example.com/") == "https://example.com"
        assert canonical_normalize_url("http://example.com:80/") == "http://example.com"
        assert canonical_normalize_url("https://example.com:443/login") == "https://example.com/login"
        assert canonical_normalize_url("https://example.com/page#section") == "https://example.com/page"
        assert canonical_normalize_url("   https://example.com/Path?q=1   ") == "https://example.com/Path?q=1"

    def test_canonical_normalize_url_preserves_hostname_without_mutation(self):
        """Verify normalization preserves the exact authority/host without prepending www."""
        assert canonical_normalize_url("https://tbcpl.lol/") == "https://tbcpl.lol"
        assert canonical_normalize_url("https://github.com/") == "https://github.com"
        assert canonical_normalize_url("https://www.github.com/") == "https://www.github.com"

    def test_canonical_normalize_url_with_ensure_apex_www(self):
        assert canonical_normalize_url("https://github.com/", ensure_apex_www=True) == "https://www.github.com"
        assert canonical_normalize_url("https://www.github.com/", ensure_apex_www=True) == "https://www.github.com"
        assert canonical_normalize_url("https://google.com/", ensure_apex_www=True) == "https://www.google.com"
        assert canonical_normalize_url("https://sub.domain.com/path", ensure_apex_www=True) == "https://sub.domain.com/path"
        assert canonical_normalize_url("http://192.168.1.1:8080/setup.exe", ensure_apex_www=True) == "http://192.168.1.1:8080/setup.exe"


# ── 2. Leakage Prevention (Registered-Domain Grouping) ───────────────

class TestLeakagePrevention:
    def test_split_by_registered_domain_guarantees_zero_leakage(self):
        records = [
            {"url": "https://a.google.com/1", "registered_domain": "google.com", "threat_category": "phishing"},
            {"url": "https://b.google.com/2", "registered_domain": "google.com", "threat_category": "phishing"},
            {"url": "https://c.google.com/3", "registered_domain": "google.com", "threat_category": "phishing"},
            {"url": "https://evil.xyz/1", "registered_domain": "evil.xyz", "threat_category": "phishing"},
            {"url": "https://sub.evil.xyz/2", "registered_domain": "evil.xyz", "threat_category": "phishing"},
            {"url": "http://10.0.0.1/mal.sh", "registered_domain": "10.0.0.1", "threat_category": "malware_distribution"},
            {"url": "http://10.0.0.2/mal.sh", "registered_domain": "10.0.0.2", "threat_category": "malware_distribution"},
            {"url": "http://10.0.0.3/mal.sh", "registered_domain": "10.0.0.3", "threat_category": "malware_distribution"},
            {"url": "http://10.0.0.4/mal.sh", "registered_domain": "10.0.0.4", "threat_category": "malware_distribution"},
            {"url": "http://10.0.0.5/mal.sh", "registered_domain": "10.0.0.5", "threat_category": "malware_distribution"},
        ]

        train_recs, test_recs = split_by_registered_domain(
            records, test_size=0.3, random_state=42, stratify_by_category=True
        )

        assert len(train_recs) > 0
        assert len(test_recs) > 0
        assert len(train_recs) + len(test_recs) == len(records)

        train_domains = {r["registered_domain"] for r in train_recs}
        test_domains = {r["registered_domain"] for r in test_recs}

        # Requirement 6: Strict zero domain leakage assertion
        leakage = train_domains & test_domains
        assert len(leakage) == 0, f"Domain leakage detected between train and test: {leakage}"


# ── 3. Conflict Quarantine & Threat Categories ──────────────────────

class TestPipelineQuarantineAndCategories:
    def test_quarantines_audit_conflicts_kakaku_and_pokeapi(self, tmp_path):
        """Requirement 4: Quarantine conflicting URLs identified in the audit."""
        phish_csv = str(tmp_path / "phish.csv")
        header_p = [
            "phish_id", "url", "phish_detail_url", "submission_time",
            "verified", "verification_time", "online", "target"
        ]
        rows_p = [
            ["1", "https://kakaku.com", "http://pt/1", "2026-11-04", "yes", "2026-11-04", "yes", "Target"],
            ["2", "https://pokeapi.co", "http://pt/2", "2026-11-04", "yes", "2026-11-04", "yes", "Target"],
            ["3", "https://real-phish.com/login", "http://pt/3", "2026-11-04", "yes", "2026-11-04", "yes", "Target"],
        ]
        _write_csv(phish_csv, header_p, rows_p)

        kaggle_csv = str(tmp_path / "kaggle.csv")
        header_k = ["url", "label", "source"]
        rows_k = [
            ["https://kakaku.com", "0", "tranco"],
            ["https://pokeapi.co", "0", "tranco"],
            ["http://192.168.1.100/dropper.sh", "1", "urlhaus"],
            ["https://openphish-sample.com/auth", "1", "openphish"],
        ]
        _write_csv(kaggle_csv, header_k, rows_k)

        out_dir = str(tmp_path / "processed")
        summary = run_dataset_pipeline(
            phish_path=phish_csv,
            kaggle_path=kaggle_csv,
            tranco_path=str(tmp_path / "dummy_tranco.csv"),
            phiusiil_path=None,
            processed_dir=out_dir,
            extract_features_flag=True,
        )

        # Verify quarantine output
        assert summary["quarantined_conflicts_count"] == 2
        q_records = summary["quarantined_conflicts"]
        q_domains = {r["registered_domain"] for r in q_records}
        assert "kakaku.com" in q_domains
        assert "pokeapi.co" in q_domains

        # Verify quarantined records are excluded from processed_threats.csv
        threats_csv = os.path.join(out_dir, "processed_threats.csv")
        assert os.path.isfile(threats_csv)
        threat_rows = _read_csv(threats_csv)
        threat_domains = {r["registered_domain"] for r in threat_rows}
        assert "kakaku.com" not in threat_domains
        assert "pokeapi.co" not in threat_domains

        # Verify 3 categories handled properly
        cat_counts = summary["category_counts"]
        assert cat_counts[THREAT_CATEGORY_PHISHING] == 2  # real-phish + openphish
        assert cat_counts[THREAT_CATEGORY_MALWARE] == 1   # urlhaus
        assert cat_counts[THREAT_CATEGORY_LEGITIMATE] == 0  # zero verified legitimate

        # Verify 20 features were extracted
        assert len(threat_rows) == 3
        for feat in FEATURE_NAMES:
            assert feat in threat_rows[0]

    def test_unverified_tranco_never_labeled_legitimate(self, tmp_path):
        """Requirement 3: Never assign legitimate labels to unverified Tranco candidates."""
        phish_csv = str(tmp_path / "phish.csv")
        header_p = ["phish_id", "url", "phish_detail_url", "submission_time", "verified", "verification_time", "online", "target"]
        _write_csv(phish_csv, header_p, [["1", "https://phish-site.com", "", "", "yes", "", "yes", ""]])

        kaggle_csv = str(tmp_path / "kaggle.csv")
        header_k = ["url", "label", "source"]
        rows_k = [
            ["https://google.com", "0", "tranco"],
            ["https://microsoft.com", "0", "tranco"],
        ]
        _write_csv(kaggle_csv, header_k, rows_k)

        out_dir = str(tmp_path / "processed")
        summary = run_dataset_pipeline(
            phish_path=phish_csv,
            kaggle_path=kaggle_csv,
            phiusiil_path=None,
            processed_dir=out_dir,
            extract_features_flag=False,
        )

        # Unverified candidates must not be in processed_threats.csv as legitimate
        unverified_csv = os.path.join(out_dir, "unverified_candidates.csv")
        assert os.path.isfile(unverified_csv)
        unverified_rows = _read_csv(unverified_csv)
        assert len(unverified_rows) == 2
        for r in unverified_rows:
            assert r["threat_category"] == THREAT_CATEGORY_UNVERIFIED
            assert r["is_verified"] == "False"
            assert r["binary_label"] == ""  # Never assigned 0

        # Limitation must be recorded
        assert summary["limitation"] is not None
        assert "Zero verified legitimate samples available" in summary["limitation"]

    def test_deduplication_removes_identical_and_slash_duplicates(self, tmp_path):
        """Requirement 5: Remove duplicate URLs using consistent normalization."""
        phish_csv = str(tmp_path / "phish.csv")
        header_p = ["phish_id", "url", "phish_detail_url", "submission_time", "verified", "verification_time", "online", "target"]
        rows_p = [
            ["1", "https://duplicate-phish.com/", "", "", "yes", "", "yes", ""],
            ["2", "https://duplicate-phish.com", "", "", "yes", "", "yes", ""],
            ["3", "https://unique-phish.com/path", "", "", "yes", "", "yes", ""],
        ]
        _write_csv(phish_csv, header_p, rows_p)

        kaggle_csv = str(tmp_path / "kaggle.csv")
        header_k = ["url", "label", "source"]
        rows_k = [
            ["https://unique-phish.com/path", "1", "openphish"],  # Cross-feed duplicate
        ]
        _write_csv(kaggle_csv, header_k, rows_k)

        out_dir = str(tmp_path / "processed")
        summary = run_dataset_pipeline(
            phish_path=phish_csv,
            kaggle_path=kaggle_csv,
            phiusiil_path=None,
            processed_dir=out_dir,
            extract_features_flag=False,
        )

        assert summary["duplicates_removed"] == 2
        assert summary["category_counts"][THREAT_CATEGORY_PHISHING] == 2

    def test_verified_legitimate_source_ingestion_when_provided(self, tmp_path):
        """Test pipeline correctly preserves 'legitimate' threat category when verified data is provided."""
        phish_csv = str(tmp_path / "phish.csv")
        header_p = ["phish_id", "url", "phish_detail_url", "submission_time", "verified", "verification_time", "online", "target"]
        _write_csv(phish_csv, header_p, [["1", "https://phish.com/login", "", "", "yes", "", "yes", ""]])

        kaggle_csv = str(tmp_path / "kaggle.csv")
        header_k = ["url", "label", "source"]
        rows_k = [["http://192.168.1.1/mal.bin", "1", "urlhaus"]]
        _write_csv(kaggle_csv, header_k, rows_k)

        # External verified benign feed
        legit_csv = str(tmp_path / "verified_legit.csv")
        _write_csv(legit_csv, ["url"], [["https://verified-edu.org"]])

        out_dir = str(tmp_path / "processed")
        summary = run_dataset_pipeline(
            phish_path=phish_csv,
            kaggle_path=kaggle_csv,
            phiusiil_path=None,
            processed_dir=out_dir,
            verified_legitimate_path=legit_csv,
            extract_features_flag=False,
        )

        assert summary["category_counts"][THREAT_CATEGORY_LEGITIMATE] == 1
        assert summary["category_counts"][THREAT_CATEGORY_PHISHING] == 1
        assert summary["category_counts"][THREAT_CATEGORY_MALWARE] == 1
        assert summary["limitation"] is None  # Verified legitimate samples exist
# ── 4. PhiUSIIL Integration, Label Mapping & Cross-Feed Handling ─────

class TestPhiUSIILIntegration:
    def test_phiusiil_integration_label_mapping_and_features(self, tmp_path):
        """Requirements 1, 2, 3, 4, 5: Read only URL/label, map 1->0 and 0->1, extract 20 features."""
        phi_csv = str(tmp_path / "phiusiil_sample.csv")
        header_phi = ["FILENAME", "URL", "URLLength", "Domain", "ExtraCol53", "ExtraCol54", "label"]
        rows_phi = [
            ["f1.txt", "https://real-university.edu/admissions", "38", "real-university.edu", "val1", "val2", "1"],  # raw 1 -> legit (0)
            ["f2.txt", "http://credential-phish.net/login.php", "37", "credential-phish.net", "val3", "val4", "0"],   # raw 0 -> phish (1)
        ]
        _write_csv(phi_csv, header_phi, rows_phi)

        out_dir = str(tmp_path / "processed")
        summary = run_dataset_pipeline(
            phish_path=str(tmp_path / "empty_pt.csv"),
            kaggle_path=str(tmp_path / "empty_kg.csv"),
            phiusiil_path=phi_csv,
            processed_dir=out_dir,
            extract_features_flag=True,
        )

        surv = summary["surviving_records"]
        assert surv["legitimate"] == 1
        assert surv["phishing"] == 1
        assert surv["total_verified_labeled"] == 2

        # Check processed_urls.csv contents
        urls_csv = os.path.join(out_dir, "processed_urls.csv")
        assert os.path.isfile(urls_csv)
        rows = _read_csv(urls_csv)
        assert len(rows) == 2

        by_cat = {r["threat_category"]: r for r in rows}
        # Legitimate verification
        legit = by_cat[THREAT_CATEGORY_LEGITIMATE]
        assert legit["url"] == "https://real-university.edu/admissions"
        assert legit["binary_label"] == "0"
        assert legit["source"] == "phiusiil"
        assert legit["registered_domain"] == "real-university.edu"

        # Phishing verification
        phish = by_cat[THREAT_CATEGORY_PHISHING]
        assert phish["url"] == "http://credential-phish.net/login.php"
        assert phish["binary_label"] == "1"
        assert phish["source"] == "phiusiil"
        assert phish["registered_domain"] == "credential-phish.net"

        # Verify all 20 canonical features are present in both
        for feat in FEATURE_NAMES:
            assert feat in legit
            assert feat in phish
            assert legit[feat] != ""
            assert phish[feat] != ""

    def test_phiusiil_cross_feed_conflict_quarantine(self, tmp_path):
        """Requirement 7, 8: Detect cross-feed conflict and quarantine."""
        phish_csv = str(tmp_path / "phish.csv")
        header_p = ["phish_id", "url", "phish_detail_url", "submission_time", "verified", "verification_time", "online", "target"]
        rows_p = [
            ["1", "https://setupyouraccounts.com/login", "", "", "yes", "", "yes", ""],  # PhishTank = 1
        ]
        _write_csv(phish_csv, header_p, rows_p)

        phi_csv = str(tmp_path / "phiusiil.csv")
        header_phi = ["URL", "label"]
        rows_phi = [
            ["https://setupyouraccounts.com/login", "1"],  # PhiUSIIL raw 1 -> mapped to 0 (legitimate) -> CONFLICT!
            ["https://peaceful-site.org", "1"],           # Valid legitimate
        ]
        _write_csv(phi_csv, header_phi, rows_phi)

        out_dir = str(tmp_path / "processed")
        summary = run_dataset_pipeline(
            phish_path=phish_csv,
            kaggle_path=str(tmp_path / "empty_kg.csv"),
            phiusiil_path=phi_csv,
            processed_dir=out_dir,
            extract_features_flag=False,
        )

        assert summary["quarantined_conflicts_count"] == 1
        q_records = summary["quarantined_conflicts"]
        assert q_records[0]["registered_domain"] == "setupyouraccounts.com"

        # Quarantined URL must not appear in processed_urls.csv or processed_threats.csv
        urls_csv = os.path.join(out_dir, "processed_urls.csv")
        rows = _read_csv(urls_csv)
        assert len(rows) == 1
        assert rows[0]["url"] == "https://peaceful-site.org"

    def test_phiusiil_cross_feed_deduplication(self, tmp_path):
        """Requirement 7: Remove duplicate URLs across PhishTank and PhiUSIIL."""
        phish_csv = str(tmp_path / "phish.csv")
        header_p = ["phish_id", "url", "phish_detail_url", "submission_time", "verified", "verification_time", "online", "target"]
        rows_p = [
            ["1", "https://shared-evil.com/page", "", "", "yes", "", "yes", ""],
        ]
        _write_csv(phish_csv, header_p, rows_p)

        phi_csv = str(tmp_path / "phiusiil.csv")
        header_phi = ["URL", "label"]
        rows_phi = [
            ["https://shared-evil.com/page", "0"],   # PhiUSIIL raw 0 -> mapped to 1 (phishing) -> duplicate agreement!
            ["https://another-site.com", "1"],       # raw 1 -> legit (0)
        ]
        _write_csv(phi_csv, header_phi, rows_phi)

        out_dir = str(tmp_path / "processed")
        summary = run_dataset_pipeline(
            phish_path=phish_csv,
            kaggle_path=str(tmp_path / "empty_kg.csv"),
            phiusiil_path=phi_csv,
            processed_dir=out_dir,
            extract_features_flag=False,
        )

        assert summary["duplicates_removed"] == 1
        surv = summary["surviving_records"]
        assert surv["phishing"] == 1
        assert surv["legitimate"] == 1

        # Priority check: PhishTank has higher priority than PhiUSIIL for the same phishing URL
        threats_csv = os.path.join(out_dir, "processed_threats.csv")
        t_rows = _read_csv(threats_csv)
        assert len(t_rows) == 1
        assert t_rows[0]["source"] == "phishtank"
    def test_merged_dataset_domain_stratified_split(self, tmp_path):
        """Requirement 10: Apply registered-domain grouping to prevent train/test leakage."""
        phi_csv = str(tmp_path / "phiusiil.csv")
        header_phi = ["URL", "label"]
        rows_phi = [
            ["https://legit-domain-1.com/a", "1"],
            ["https://legit-domain-1.com/b", "1"],
            ["https://legit-domain-2.org/page", "1"],
            ["https://phish-domain-1.net/bad", "0"],
            ["https://phish-domain-1.net/bad2", "0"],
            ["https://phish-domain-2.info/login", "0"],
        ]
        _write_csv(phi_csv, header_phi, rows_phi)

        out_dir = str(tmp_path / "processed")
        summary = run_dataset_pipeline(
            phish_path=str(tmp_path / "empty_pt.csv"),
            kaggle_path=str(tmp_path / "empty_kg.csv"),
            phiusiil_path=phi_csv,
            processed_dir=out_dir,
            extract_features_flag=False,
        )

        leak_info = summary["domain_leakage_protection"]
        assert leak_info["domain_overlap_count"] == 0
        assert leak_info["leakage_rate_percent"] == "0.00%"
        assert leak_info["status"] == "ZERO_LEAKAGE_VERIFIED"
