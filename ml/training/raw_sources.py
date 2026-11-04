"""
PhishGuard – Raw Dataset Source Ingestor & Inspector

Specialized loaders for raw intelligence feeds:
1. PhishTank CSV:
   - Header: phish_id, url, phish_detail_url, submission_time, verified, verification_time, online, target
   - Filter criteria: verified == 'yes' AND online == 'yes'
   - Phishing URL candidate: url column
   - Normalization: whitespace stripping, URL syntax validation
   - Deduplication: safe hash-set deduplication
   - Provenance: PhishTank Developer License
2. Tranco CSV:
   - Headerless CSV: rank, domain
   - Domain candidate: domain column normalized into HTTPS candidate URL
   - CRITICAL: Never automatically labeled as legitimate. Retained as unlabeled candidates (label=None)
   - Deduplication: unique domain deduplication
   - Provenance: CC BY 4.0 (Victor Le Pochat et al.)
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.training.validate_dataset import compute_file_sha256, save_dataset_manifest

logger = logging.getLogger(__name__)

DEFAULT_PHISHTANK_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "phishing.csv")
DEFAULT_TRANCO_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "tranco.csv")
DEFAULT_KAGGLE_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "kaggle_urls.csv")
DEFAULT_PHIUSIIL_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "phiusiil.csv")


# ── URL Normalization Helper ────────────────────────────────────────

def normalize_url(raw_url: str, default_scheme: str = "https") -> str | None:
    """Safely clean and normalize a raw URL candidate.

    Returns normalized URL string, or None if raw_url is empty/invalid.
    """
    if not raw_url or not isinstance(raw_url, str):
        return None

    cleaned = raw_url.strip()
    if not cleaned:
        return None

    parsed = urlparse(cleaned)
    if not parsed.scheme:
        cleaned = f"{default_scheme}://{cleaned}"
        parsed = urlparse(cleaned)

    # Valid candidate must have a netloc or path
    if not parsed.netloc and not parsed.path:
        return None

    return cleaned


# ── PhishTank Ingestion ─────────────────────────────────────────────

def load_phishtank_records(
    filepath: str = DEFAULT_PHISHTANK_PATH,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read, filter, normalize and deduplicate PhishTank raw records.

    Filters strictly for verified == 'yes' and online == 'yes'.
    Extracts the 'url' column as the candidate phishing URL.

    Parameters
    ----------
    filepath : str
        Path to the raw PhishTank CSV export.
    max_records : int | None
        Optional maximum number of records to return.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        List of candidate dictionaries and a summary statistics dictionary.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"PhishTank raw file not found: {filepath}")

    total_rows = 0
    missing_url_count = 0
    unverified_count = 0
    offline_count = 0
    duplicate_count = 0

    seen_urls: set[str] = set()
    candidates: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"PhishTank CSV is empty or has no header: {filepath}")

        # Case-insensitive column map
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        col_url = cols.get("url")
        col_verified = cols.get("verified")
        col_online = cols.get("online")
        col_id = cols.get("phish_id")
        col_target = cols.get("target")
        col_sub_time = cols.get("submission_time")
        col_ver_time = cols.get("verification_time")

        if not col_url or not col_verified or not col_online:
            raise ValueError(
                f"Missing required PhishTank columns in {filepath}. Found: {reader.fieldnames}"
            )

        for row in reader:
            total_rows += 1

            raw_url = row.get(col_url) or ""
            verified = (row.get(col_verified) or "").strip().lower()
            online = (row.get(col_online) or "").strip().lower()

            if not raw_url.strip():
                missing_url_count += 1
                continue

            # Requirement 3: retain only records where verified == yes and online == yes
            if verified != "yes":
                unverified_count += 1
                continue

            if online != "yes":
                offline_count += 1
                continue

            norm_url = normalize_url(raw_url)
            if not norm_url:
                missing_url_count += 1
                continue

            # Requirement 4: remove duplicates safely
            if norm_url in seen_urls:
                duplicate_count += 1
                continue

            seen_urls.add(norm_url)

            record = {
                "phish_id": row.get(col_id) if col_id else None,
                "url": norm_url,
                "label": 1,  # Phishing
                "target": row.get(col_target) if col_target else None,
                "submission_time": row.get(col_sub_time) if col_sub_time else None,
                "verification_time": row.get(col_ver_time) if col_ver_time else None,
                "source": "PhishTank",
                "license": "PhishTank Developer License",
            }

            candidates.append(record)
            if len(sample_records) < 3:
                sample_records.append(record)

            if max_records and len(candidates) >= max_records:
                break

    stats = {
        "source": "PhishTank",
        "filepath": filepath,
        "total_rows": total_rows,
        "missing_urls": missing_url_count,
        "unverified_dropped": unverified_count,
        "offline_dropped": offline_count,
        "duplicates_dropped": duplicate_count,
        "retained_candidates": len(candidates),
        "license": "PhishTank Developer License (Cisco/OpenDNS)",
        "sample_records": sample_records,
    }

    return candidates, stats


# ── Tranco Ingestion ────────────────────────────────────────────────

def load_tranco_records(
    filepath: str = DEFAULT_TRANCO_PATH,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read, parse, normalize and deduplicate Tranco raw records.

    Requirement 2: Correctly read as a headerless file with columns `rank` and `domain`.
    Requirement 5: Do NOT automatically label Tranco domains as legitimate.
                   Keep them as unlabeled candidates (label=None).

    Parameters
    ----------
    filepath : str
        Path to the raw Tranco CSV (headerless rank,domain).
    max_records : int | None
        Optional maximum number of records to return.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        List of candidate dictionaries and a summary statistics dictionary.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Tranco raw file not found: {filepath}")

    total_rows = 0
    missing_domain_count = 0
    missing_rank_count = 0
    duplicate_count = 0

    seen_domains: set[str] = set()
    candidates: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            total_rows += 1

            if not row:
                continue

            rank_val: int | None = None
            domain_val: str = ""

            if len(row) >= 2:
                rank_str = row[0].strip()
                domain_val = row[1].strip()
                if rank_str.isdigit():
                    rank_val = int(rank_str)
                else:
                    missing_rank_count += 1
            elif len(row) == 1:
                missing_rank_count += 1
                domain_val = row[0].strip()

            if not domain_val:
                missing_domain_count += 1
                continue

            clean_domain = domain_val.lower().rstrip("/")
            if not clean_domain:
                missing_domain_count += 1
                continue

            # Requirement 4: remove duplicates safely
            if clean_domain in seen_domains:
                duplicate_count += 1
                continue

            seen_domains.add(clean_domain)

            candidate_url = f"https://{clean_domain}"

            # Requirement 5: Do NOT automatically label Tranco domains as legitimate!
            # Keep them as unlabeled candidates (label: None)
            record = {
                "rank": rank_val,
                "domain": clean_domain,
                "url": candidate_url,
                "label": None,  # Explicitly unlabeled candidate
                "source": "Tranco",
                "license": "CC BY 4.0",
            }

            candidates.append(record)
            if len(sample_records) < 3:
                sample_records.append(record)

            if max_records and len(candidates) >= max_records:
                break

    stats = {
        "source": "Tranco",
        "filepath": filepath,
        "total_rows": total_rows,
        "missing_domains": missing_domain_count,
        "missing_ranks": missing_rank_count,
        "duplicates_dropped": duplicate_count,
        "retained_candidates": len(candidates),
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "sample_records": sample_records,
    }

    return candidates, stats


# ── Kaggle Ingestion ────────────────────────────────────────────────

def load_kaggle_records(
    filepath: str = DEFAULT_KAGGLE_PATH,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read, parse, normalize and deduplicate Kaggle multi-source URL records.

    Columns: url, label, source
    - urlhaus: malware distribution payloads (kept distinguishable from phishing)
    - openphish: zero-day phishing feeds
    - tranco: popular domains (kept as unverified candidates, never auto-treated as legitimate)

    Parameters
    ----------
    filepath : str
        Path to the Kaggle URLs CSV.
    max_records : int | None
        Optional maximum number of records to return.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        List of candidate dictionaries and a summary statistics dictionary.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Kaggle raw file not found: {filepath}")

    total_rows = 0
    missing_url_count = 0
    missing_label_count = 0
    missing_source_count = 0
    duplicate_count = 0

    seen_urls: set[str] = set()
    candidates: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []

    source_counts: dict[str, int] = {}
    threat_counts: dict[str, int] = {}

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Kaggle CSV is empty or has no header: {filepath}")

        cols = {c.strip().lower(): c for c in reader.fieldnames}
        col_url = cols.get("url")
        col_label = cols.get("label")
        col_source = cols.get("source")

        if not col_url or not col_label or not col_source:
            raise ValueError(
                f"Missing required Kaggle columns (url, label, source) in {filepath}. Found: {reader.fieldnames}"
            )

        for row in reader:
            total_rows += 1

            raw_url = (row.get(col_url) or "").strip()
            raw_label = (row.get(col_label) or "").strip()
            raw_source = (row.get(col_source) or "").strip().lower()

            if not raw_url:
                missing_url_count += 1
                continue
            if not raw_label:
                missing_label_count += 1
                continue
            if not raw_source:
                missing_source_count += 1
                continue

            norm_url = normalize_url(raw_url)
            if not norm_url:
                missing_url_count += 1
                continue

            if norm_url in seen_urls:
                duplicate_count += 1
                continue

            seen_urls.add(norm_url)

            # Classify threat type and provenance
            # Requirement 7: Keep URLhaus malware URLs distinguishable from confirmed phishing URLs
            # Requirement 8: Never automatically treat Tranco candidates as verified legitimate URLs
            if raw_source == "urlhaus":
                threat_type = "malware_distribution"
                license_info = "CC0 1.0 Universal"
                is_verified_legit = False
                parsed_label: int | None = int(raw_label) if raw_label.isdigit() else 1
            elif raw_source == "openphish":
                threat_type = "phishing"
                license_info = "OpenPhish Community Feed"
                is_verified_legit = False
                parsed_label = int(raw_label) if raw_label.isdigit() else 1
            elif raw_source == "tranco":
                threat_type = "unlabeled_candidate"
                license_info = "CC BY 4.0"
                is_verified_legit = False
                # Retained as unverified candidate (raw label is 0 in kaggle, but flagged unverified)
                parsed_label = int(raw_label) if raw_label.isdigit() else 0
            else:
                threat_type = "unknown"
                license_info = "Unspecified"
                is_verified_legit = False
                parsed_label = int(raw_label) if raw_label.isdigit() else None

            source_counts[raw_source] = source_counts.get(raw_source, 0) + 1
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1

            record = {
                "url": norm_url,
                "label": parsed_label,
                "source": raw_source,
                "threat_type": threat_type,
                "is_verified_legitimate": is_verified_legit,
                "license": license_info,
            }

            candidates.append(record)
            if len(sample_records) < 3:
                sample_records.append(record)

            if max_records and len(candidates) >= max_records:
                break

    stats = {
        "source": "Kaggle Multi-Source",
        "filepath": filepath,
        "total_rows": total_rows,
        "missing_urls": missing_url_count,
        "missing_labels": missing_label_count,
        "missing_sources": missing_source_count,
        "duplicates_dropped": duplicate_count,
        "retained_candidates": len(candidates),
        "source_breakdown": source_counts,
        "threat_type_breakdown": threat_counts,
        "sample_records": sample_records,
    }

    return candidates, stats


# ── PhiUSIIL Loader ──────────────────────────────────────────────────

def load_phiusiil_records(
    filepath: str = DEFAULT_PHIUSIIL_PATH,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load PhiUSIIL Phishing URL Dataset (Prasad & Chandra 2024 / UCI ID: 967).

    Requirements:
    1. Read only the original URL and label columns. Ignore all 54 other precomputed features.
    2. Map PhiUSIIL labels correctly:
       - Original label 1 -> PhishGuard label 0 (legitimate).
       - Original label 0 -> PhishGuard label 1 (phishing).
    3. Preserve source attribution as 'phiusiil'.
    4. Normalize URLs consistently and remove internal duplicates safely.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"PhiUSIIL dataset file not found: {filepath}")

    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    total_rows = 0
    missing_url_count = 0
    missing_label_count = 0
    duplicate_count = 0
    invalid_label_count = 0

    threat_counts: dict[str, int] = {"legitimate": 0, "phishing": 0}
    sample_records: list[dict[str, Any]] = []

    with open(filepath, mode="r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"PhiUSIIL CSV is empty or malformed: {filepath}")

        cols = {c.strip().lower(): c for c in reader.fieldnames}
        col_url = cols.get("url")
        col_label = cols.get("label")

        if not col_url or not col_label:
            raise ValueError(
                f"Missing required PhiUSIIL columns (URL, label) in {filepath}. Found: {reader.fieldnames[:10]}"
            )

        for row in reader:
            total_rows += 1

            raw_url = row.get(col_url) or ""
            raw_label = (row.get(col_label) or "").strip()

            if not raw_url.strip():
                missing_url_count += 1
                continue

            if not raw_label:
                missing_label_count += 1
                continue

            # Requirement 2: Map PhiUSIIL labels correctly:
            # Original label 1 -> PhishGuard label 0 (legitimate)
            # Original label 0 -> PhishGuard label 1 (phishing)
            if raw_label == "1":
                mapped_label = 0
                threat_type = "legitimate"
                is_legit = True
            elif raw_label == "0":
                mapped_label = 1
                threat_type = "phishing"
                is_legit = False
            else:
                invalid_label_count += 1
                continue

            norm_url = normalize_url(raw_url)
            if not norm_url:
                missing_url_count += 1
                continue

            if norm_url in seen_urls:
                duplicate_count += 1
                continue
            seen_urls.add(norm_url)

            threat_counts[threat_type] += 1

            record = {
                "url": norm_url,
                "label": mapped_label,
                "source": "phiusiil",
                "threat_type": threat_type,
                "is_verified_legitimate": is_legit,
                "license": "CC BY 4.0",
            }
            candidates.append(record)

            if len(sample_records) < 3:
                sample_records.append(record)

            if max_records and len(candidates) >= max_records:
                break

    stats = {
        "source": "PhiUSIIL (UCI ML Repository ID: 967)",
        "filepath": filepath,
        "total_rows": total_rows,
        "missing_urls": missing_url_count,
        "missing_labels": missing_label_count,
        "invalid_labels": invalid_label_count,
        "duplicates_dropped": duplicate_count,
        "retained_candidates": len(candidates),
        "threat_type_breakdown": threat_counts,
        "sample_records": sample_records,
    }

    return candidates, stats


# ── Cross-Dataset Quality & Provenance Audit ─────────────────────────

def audit_dataset_quality(
    phish_path: str = DEFAULT_PHISHTANK_PATH,
    tranco_path: str = DEFAULT_TRANCO_PATH,
    kaggle_path: str = DEFAULT_KAGGLE_PATH,
    sample_feature_test_size: int = 1000,
) -> dict[str, Any]:
    """Perform comprehensive quality, overlap, and leakage audit across all 3 raw sources.

    Requirements addressed:
    1. Compare overlapping URLs across PhishTank and Kaggle.
    2. Detect conflicting labels between sources.
    3. Check URL normalization consistency.
    4. Verify source-wise label assignments.
    5. Identify potential dataset leakage and duplicate domains across splits.
    6. Check whether 20 URL features can be extracted from all samples.
    7. Keep URLhaus malware URLs distinguishable from confirmed phishing URLs.
    8. Never automatically treat Tranco candidates as verified legitimate URLs.
    9. Preserve all original raw files.
    """
    from ml.features.url_features import extract_url_features

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sources": {},
        "cross_dataset_audit": {},
    }

    # 1. Load PhishTank
    pt_candidates, pt_stats = load_phishtank_records(phish_path)
    report["sources"]["phishtank"] = pt_stats

    # 2. Load Tranco
    tr_candidates, tr_stats = load_tranco_records(tranco_path)
    report["sources"]["tranco"] = tr_stats

    # 3. Load Kaggle
    kg_candidates, kg_stats = load_kaggle_records(kaggle_path)
    report["sources"]["kaggle"] = kg_stats

    # 4. Overlap & Conflict Analysis (PhishTank vs Kaggle)
    pt_url_map = {r["url"]: r for r in pt_candidates}
    kg_url_map = {r["url"]: r for r in kg_candidates}

    exact_overlaps = set(pt_url_map.keys()) & set(kg_url_map.keys())

    # Normalized overlap (lowercase, stripped trailing slash)
    def _norm_key(u: str) -> str:
        u_clean = u.strip().lower()
        if not u_clean.startswith(("http://", "https://")):
            u_clean = "http://" + u_clean
        return u_clean.rstrip("/")

    pt_norm_map = {_norm_key(r["url"]): r for r in pt_candidates}
    kg_norm_map = {_norm_key(r["url"]): r for r in kg_candidates}
    normalized_overlaps = set(pt_norm_map.keys()) & set(kg_norm_map.keys())

    conflicting_labels: list[dict[str, Any]] = []
    for nkey in normalized_overlaps:
        pt_rec = pt_norm_map[nkey]
        kg_rec = kg_norm_map[nkey]
        if pt_rec["label"] != kg_rec["label"]:
            conflicting_labels.append({
                "normalized_url": nkey,
                "phishtank_url": pt_rec["url"],
                "phishtank_label": pt_rec["label"],
                "kaggle_url": kg_rec["url"],
                "kaggle_label": kg_rec["label"],
                "kaggle_source": kg_rec["source"],
            })

    # 5. Domain / Hostname Leakage Analysis
    def _get_hostname(u: str) -> str:
        if not u.startswith(("http://", "https://")):
            u = "http://" + u
        try:
            netloc = urlparse(u).netloc
            if ":" in netloc:
                netloc = netloc.split(":")[0]
            return netloc.lower().strip()
        except Exception:
            return ""

    pt_hosts = {_get_hostname(r["url"]) for r in pt_candidates if _get_hostname(r["url"])}
    kg_tranco_hosts = {
        _get_hostname(r["url"]) for r in kg_candidates
        if r["source"] == "tranco" and _get_hostname(r["url"])
    }
    kg_urlhaus_hosts = {
        _get_hostname(r["url"]) for r in kg_candidates
        if r["source"] == "urlhaus" and _get_hostname(r["url"])
    }

    shared_pt_kg_tranco = pt_hosts & kg_tranco_hosts
    shared_urlhaus_kg_tranco = kg_urlhaus_hosts & kg_tranco_hosts

    # 6. Feature Extraction Validation on Sample Batches
    feature_extraction_errors: dict[str, int] = {"phishtank": 0, "tranco": 0, "kaggle": 0}
    test_batches = {
        "phishtank": [r["url"] for r in pt_candidates[:sample_feature_test_size]],
        "tranco": [r["url"] for r in tr_candidates[:sample_feature_test_size]],
        "kaggle": [r["url"] for r in kg_candidates[:sample_feature_test_size]],
    }

    for sname, url_list in test_batches.items():
        errs = 0
        for u in url_list:
            try:
                feats = extract_url_features(u)
                if len(feats) != 20:
                    errs += 1
                for v in feats.values():
                    if v is None or (isinstance(v, float) and (v != v)):
                        errs += 1
            except Exception:
                errs += 1
        feature_extraction_errors[sname] = errs

    report["cross_dataset_audit"] = {
        "exact_url_overlap_pt_kaggle": len(exact_overlaps),
        "normalized_url_overlap_pt_kaggle": len(normalized_overlaps),
        "conflicting_labels_count": len(conflicting_labels),
        "conflicting_labels": conflicting_labels,
        "domain_leakage": {
            "shared_hostnames_phishtank_and_tranco_candidates": len(shared_pt_kg_tranco),
            "sample_shared_hostnames_phish_tranco": sorted(list(shared_pt_kg_tranco))[:10],
            "shared_hostnames_urlhaus_and_tranco_candidates": len(shared_urlhaus_kg_tranco),
            "sample_shared_hostnames_urlhaus_tranco": sorted(list(shared_urlhaus_kg_tranco))[:10],
        },
        "feature_extraction_verification": {
            "samples_tested_per_source": sample_feature_test_size,
            "extraction_errors": feature_extraction_errors,
            "all_features_compatible": sum(feature_extraction_errors.values()) == 0,
        },
    }

    return report


# ── Full Inspection & Report Generator ──────────────────────────────

def inspect_raw_dataset_sources(
    phish_path: str = DEFAULT_PHISHTANK_PATH,
    tranco_path: str = DEFAULT_TRANCO_PATH,
    kaggle_path: str = DEFAULT_KAGGLE_PATH,
    save_metadata: bool = True,
) -> dict[str, Any]:
    """Inspect raw datasets, generate statistics and provenance manifests.

    Requirement 7: Generate a concise inspection report showing row counts,
                   duplicates, missing values, and sample records for sources.
    Requirement 9: Do not overwrite the original raw CSV files.

    Returns
    -------
    dict[str, Any]
        Consolidated inspection report.
    """
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }

    # 1. Inspect PhishTank
    if os.path.isfile(phish_path):
        phish_candidates, phish_stats = load_phishtank_records(phish_path)
        report["sources"]["phishtank"] = phish_stats

        if save_metadata:
            save_dataset_manifest(
                csv_path=phish_path,
                source_name="PhishTank",
                license_info="PhishTank Developer License (Cisco/OpenDNS)",
                description="Verified online phishing URL feed",
            )
    else:
        report["sources"]["phishtank"] = {"error": f"File not found: {phish_path}"}

    # 2. Inspect Tranco
    if os.path.isfile(tranco_path):
        tranco_candidates, tranco_stats = load_tranco_records(tranco_path)
        report["sources"]["tranco"] = tranco_stats

        if save_metadata:
            save_dataset_manifest(
                csv_path=tranco_path,
                source_name="Tranco",
                license_info="Creative Commons Attribution 4.0 International (CC BY 4.0)",
                description="Tranco research-oriented top 1M site ranking list",
            )
    else:
        report["sources"]["tranco"] = {"error": f"File not found: {tranco_path}"}

    # 3. Inspect Kaggle
    if os.path.isfile(kaggle_path):
        kaggle_candidates, kaggle_stats = load_kaggle_records(kaggle_path)
        report["sources"]["kaggle"] = kaggle_stats

        if save_metadata:
            save_dataset_manifest(
                csv_path=kaggle_path,
                source_name="Kaggle Multi-Source URL Feed (URLhaus + Tranco + OpenPhish)",
                license_info="Multi-License (URLhaus CC0, Tranco CC BY 4.0, OpenPhish Community)",
                description="Aggregated dataset containing URLhaus malware URLs, Tranco top domains, and OpenPhish URLs",
            )
    else:
        report["sources"]["kaggle"] = {"error": f"File not found: {kaggle_path}"}

    return report


# ── CLI Interface ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("\n" + "=" * 70)
    print("PhishGuard Raw Dataset Inspection")
    print("=" * 70)

    inspection = inspect_raw_dataset_sources(save_metadata=True)

    pt = inspection["sources"].get("phishtank", {})
    if "error" not in pt:
        print("\n--- 1. PhishTank Inspection ---")
        print(f"File:                 {pt['filepath']}")
        print(f"Total Rows:           {pt['total_rows']:,}")
        print(f"Missing URLs:         {pt['missing_urls']:,}")
        print(f"Unverified Dropped:   {pt['unverified_dropped']:,}")
        print(f"Offline Dropped:      {pt['offline_dropped']:,}")
        print(f"Duplicates Dropped:   {pt['duplicates_dropped']:,}")
        print(f"Retained Candidates:  {pt['retained_candidates']:,}")
        print(f"License:              {pt['license']}")
        print("Sample Records:")
        for idx, rec in enumerate(pt["sample_records"], 1):
            print(f"  [{idx}] URL:    {rec['url']}")
            print(f"      Target: {rec['target']} | ID: {rec['phish_id']} | Label: {rec['label']}")
    else:
        print(f"PhishTank Error: {pt['error']}")

    tr = inspection["sources"].get("tranco", {})
    if "error" not in tr:
        print("\n--- 2. Tranco Inspection ---")
        print(f"File:                 {tr['filepath']}")
        print(f"Total Rows:           {tr['total_rows']:,}")
        print(f"Missing Domains:      {tr['missing_domains']:,}")
        print(f"Missing Ranks:        {tr['missing_ranks']:,}")
        print(f"Duplicates Dropped:   {tr['duplicates_dropped']:,}")
        print(f"Retained Candidates:  {tr['retained_candidates']:,}")
        print(f"License:              {tr['license']}")
        print("Sample Records (Unlabeled Candidates):")
        for idx, rec in enumerate(tr["sample_records"], 1):
            print(f"  [{idx}] Rank: {rec['rank']} | Domain: {rec['domain']}")
            print(f"      Candidate URL: {rec['url']} | Label: {rec['label']} (unlabeled)")
    else:
        print(f"Tranco Error: {tr['error']}")

    print("\n" + "=" * 70 + "\n")

