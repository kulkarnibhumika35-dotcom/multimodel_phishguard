"""
PhishGuard – Dataset Validation & Provenance Suite

Validates real phishing and legitimate URL datasets before preprocessing
and model training.

Checks for:
- Missing values (blank URLs, blank labels)
- Duplicate URLs and conflicting labels
- Invalid or unrecognised labels
- Class distribution and severe imbalance
- URL protocol / scheme validity
- Minimum sample size requirements for model training

Preserves:
- Data source provenance
- Collection timestamps
- Dataset SHA-256 integrity checksums

Strictly adheres to:
- No automated downloading
- Never invents or generates fake training samples
- Read-only inspection of input files
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Minimum training thresholds required by ml.training.train_url_model
MIN_TRAINING_SAMPLES: int = 20
MIN_CLASS_SAMPLES: int = 5
MAX_IMBALANCE_RATIO: float = 0.90  # 90/10 split triggers an imbalance warning

# Accepted label representations
_VALID_LABEL_MAP: dict[str, int] = {
    # Legitimate (0)
    "0": 0,
    "legitimate": 0,
    "benign": 0,
    "safe": 0,
    "good": 0,
    # Phishing (1)
    "1": 1,
    "phishing": 1,
    "malicious": 1,
    "bad": 1,
    "spam": 1,
}


# ── Integrity & Hash Helpers ────────────────────────────────────────

def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file for integrity tracking."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ── Provenance Manifest Management ──────────────────────────────────

def save_dataset_manifest(
    csv_path: str,
    source_name: str,
    collection_date: str | None = None,
    license_info: str = "",
    description: str = "",
    metadata_path: str | None = None,
) -> dict[str, Any]:
    """Generate and save a provenance metadata manifest for a dataset file.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV file.
    source_name : str
        Name of the data provider (e.g., 'PhishTank', 'URLhaus', 'Tranco').
    collection_date : str | None
        Date of collection (ISO 8601 YYYY-MM-DD). Defaults to current UTC date.
    license_info : str
        Licensing terms (e.g., 'CC0 1.0', 'CC BY 4.0', 'PhishTank Developer License').
    description : str
        Brief description of the dataset contents.
    metadata_path : str | None
        Optional custom path for the manifest. Defaults to `<csv_path>.meta.json`.

    Returns
    -------
    dict[str, Any]
        The created provenance manifest dictionary.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    target_path = metadata_path or f"{csv_path}.meta.json"
    file_sha = compute_file_sha256(csv_path)
    file_size_bytes = os.path.getsize(csv_path)

    today_str = collection_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    manifest = {
        "dataset_filename": os.path.basename(csv_path),
        "source_name": source_name.strip(),
        "collection_date": today_str,
        "license": license_info.strip() or "Unspecified",
        "description": description.strip(),
        "sha256": file_sha,
        "file_size_bytes": file_size_bytes,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved provenance manifest to %s", target_path)
    return manifest


def load_dataset_manifest(
    csv_path: str,
    metadata_path: str | None = None,
) -> dict[str, Any] | None:
    """Load an existing provenance manifest for a dataset file if present."""
    target_path = metadata_path or f"{csv_path}.meta.json"
    if not os.path.isfile(target_path):
        return None
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read manifest at %s: %s", target_path, exc)
        return None


# ── Dataset Validation Engine ───────────────────────────────────────

def validate_dataset(
    csv_path: str,
    url_column: str = "url",
    label_column: str = "label",
    metadata_path: str | None = None,
) -> dict[str, Any]:
    """Validate a raw or curated dataset CSV.

    Performs comprehensive integrity validation without modifying the file.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV file.
    url_column : str
        Expected header name for URLs (case-insensitive).
    label_column : str
        Expected header name for labels (case-insensitive).
    metadata_path : str | None
        Optional path to companion `.meta.json` manifest.

    Returns
    -------
    dict[str, Any]
        Detailed validation report including counts, class distribution,
        integrity flags, warnings, and provenance info.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If the file is empty or missing required columns.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty or has no header: {csv_path}")

        # Column resolution (case-insensitive)
        col_map = {c.strip().lower(): c for c in reader.fieldnames}
        actual_url_col = col_map.get(url_column.strip().lower())
        actual_label_col = col_map.get(label_column.strip().lower())

        if actual_url_col is None:
            raise ValueError(
                f"URL column '{url_column}' not found in CSV. Available columns: {list(reader.fieldnames)}"
            )
        if actual_label_col is None:
            raise ValueError(
                f"Label column '{label_column}' not found in CSV. Available columns: {list(reader.fieldnames)}"
            )

        rows = list(reader)

    total_rows = len(rows)
    warnings: list[str] = []

    if total_rows == 0:
        return {
            "total_rows": 0,
            "valid_rows": 0,
            "missing_urls": 0,
            "missing_labels": 0,
            "invalid_labels": 0,
            "duplicate_urls": 0,
            "unique_urls": 0,
            "conflicting_labels": 0,
            "class_distribution": {
                "legitimate": 0,
                "phishing": 0,
                "legitimate_pct": 0.0,
                "phishing_pct": 0.0,
            },
            "scheme_distribution": {"https": 0, "http": 0, "other": 0, "missing": 0},
            "warnings": ["Dataset contains 0 rows."],
            "is_valid_for_training": False,
            "provenance": load_dataset_manifest(csv_path, metadata_path),
        }

    # Tracking metrics
    missing_urls = 0
    missing_labels = 0
    invalid_labels = 0
    invalid_label_samples: list[str] = []

    schemes: dict[str, int] = {"https": 0, "http": 0, "other": 0, "missing": 0}
    url_to_labels: dict[str, set[int]] = {}
    valid_records: list[tuple[str, int]] = []

    for idx, row in enumerate(rows, start=1):
        raw_url = row.get(actual_url_col)
        raw_label = row.get(actual_label_col)

        # 1. Missing URL check
        if raw_url is None or not str(raw_url).strip():
            missing_urls += 1
            continue

        clean_url = str(raw_url).strip()

        # Scheme analysis
        parsed = urlparse(clean_url)
        scheme = parsed.scheme.lower() if parsed.scheme else ""
        if scheme == "https":
            schemes["https"] += 1
        elif scheme == "http":
            schemes["http"] += 1
        elif scheme == "":
            schemes["missing"] += 1
        else:
            schemes["other"] += 1

        # 2. Missing Label check
        if raw_label is None or not str(raw_label).strip():
            missing_labels += 1
            continue

        clean_label_str = str(raw_label).strip().lower()

        # 3. Invalid Label check
        if clean_label_str not in _VALID_LABEL_MAP:
            invalid_labels += 1
            if len(invalid_label_samples) < 5:
                invalid_label_samples.append(clean_label_str)
            continue

        norm_label = _VALID_LABEL_MAP[clean_label_str]

        # Record for duplication & conflict checking
        if clean_url not in url_to_labels:
            url_to_labels[clean_url] = set()
        url_to_labels[clean_url].add(norm_label)

        valid_records.append((clean_url, norm_label))

    # 4. Duplicates & Conflicts
    unique_urls = len(url_to_labels)
    duplicate_urls = len(valid_records) - unique_urls
    conflicting_labels = sum(1 for labels in url_to_labels.values() if len(labels) > 1)

    if duplicate_urls > 0:
        dup_pct = (duplicate_urls / len(valid_records)) * 100 if valid_records else 0.0
        warnings.append(
            f"Dataset contains {duplicate_urls} duplicate URL instances ({dup_pct:.1f}% of valid rows)."
        )

    if conflicting_labels > 0:
        warnings.append(
            f"CRITICAL: {conflicting_labels} URL(s) have conflicting labels (assigned as both 0 and 1)."
        )

    # 5. Class Distribution (based on unique valid URLs, resolving duplicates)
    # Deduplicate keeping first occurrence
    seen_urls: set[str] = set()
    deduped_classes: list[int] = []
    for u, lbl in valid_records:
        if u not in seen_urls:
            seen_urls.add(u)
            deduped_classes.append(lbl)

    legit_count = sum(1 for c in deduped_classes if c == 0)
    phish_count = sum(1 for c in deduped_classes if c == 1)
    deduped_total = len(deduped_classes)

    legit_pct = round((legit_count / deduped_total * 100), 2) if deduped_total else 0.0
    phish_pct = round((phish_count / deduped_total * 100), 2) if deduped_total else 0.0

    # 6. Training Thresholds & Balance Warnings
    if deduped_total < MIN_TRAINING_SAMPLES:
        warnings.append(
            f"Dataset has {deduped_total} unique samples, below the minimum {MIN_TRAINING_SAMPLES} required for training."
        )

    if legit_count < MIN_CLASS_SAMPLES:
        warnings.append(
            f"Legitimate class count ({legit_count}) is below the required minimum of {MIN_CLASS_SAMPLES} samples."
        )

    if phish_count < MIN_CLASS_SAMPLES:
        warnings.append(
            f"Phishing class count ({phish_count}) is below the required minimum of {MIN_CLASS_SAMPLES} samples."
        )

    if deduped_total > 0:
        max_ratio = max(legit_count, phish_count) / deduped_total
        if max_ratio > MAX_IMBALANCE_RATIO:
            dominant_class = "legitimate" if legit_count > phish_count else "phishing"
            warnings.append(
                f"Severe class imbalance: {dominant_class} accounts for {max_ratio * 100:.1f}% of samples."
            )

    if schemes["missing"] > 0:
        warnings.append(
            f"{schemes['missing']} URL(s) lack an explicit protocol scheme (http/https)."
        )

    # Training readiness determination
    is_valid_for_training = (
        deduped_total >= MIN_TRAINING_SAMPLES
        and legit_count >= MIN_CLASS_SAMPLES
        and phish_count >= MIN_CLASS_SAMPLES
        and conflicting_labels == 0
    )

    provenance = load_dataset_manifest(csv_path, metadata_path)

    report = {
        "total_rows": total_rows,
        "valid_rows": len(valid_records),
        "unique_valid_urls": unique_urls,
        "missing_urls": missing_urls,
        "missing_labels": missing_labels,
        "invalid_labels": invalid_labels,
        "invalid_label_samples": invalid_label_samples,
        "duplicate_urls": duplicate_urls,
        "conflicting_labels": conflicting_labels,
        "class_distribution": {
            "legitimate": legit_count,
            "phishing": phish_count,
            "legitimate_pct": legit_pct,
            "phishing_pct": phish_pct,
        },
        "scheme_distribution": schemes,
        "warnings": warnings,
        "is_valid_for_training": is_valid_for_training,
        "provenance": provenance,
    }

    return report


# ── CLI Interface ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Validate a PhishGuard URL dataset.")
    parser.add_argument("csv_path", help="Path to dataset CSV to inspect")
    parser.add_argument("--url-col", default="url", help="URL column name (default: url)")
    parser.add_argument("--label-col", default="label", help="Label column name (default: label)")
    parser.add_argument("--meta", default=None, help="Optional custom metadata manifest path")
    args = parser.parse_args()

    try:
        results = validate_dataset(
            args.csv_path,
            url_column=args.url_col,
            label_column=args.label_col,
            metadata_path=args.meta,
        )
        print("\n" + "=" * 60)
        print(f"PhishGuard Dataset Validation Report: {args.csv_path}")
        print("=" * 60)
        print(f"Total Rows:           {results['total_rows']}")
        print(f"Valid Rows:           {results['valid_rows']}")
        print(f"Unique URLs:          {results['unique_valid_urls']}")
        print(f"Missing URLs:         {results['missing_urls']}")
        print(f"Missing Labels:       {results['missing_labels']}")
        print(f"Invalid Labels:       {results['invalid_labels']}")
        print(f"Duplicates:           {results['duplicate_urls']}")
        print(f"Conflicting Labels:   {results['conflicting_labels']}")
        print("-" * 60)
        cd = results["class_distribution"]
        print(f"Legitimate (0):       {cd['legitimate']} ({cd['legitimate_pct']}%)")
        print(f"Phishing (1):         {cd['phishing']} ({cd['phishing_pct']}%)")
        print("-" * 60)
        print(f"Ready For Training:   {results['is_valid_for_training']}")
        if results["warnings"]:
            print("\nWarnings:")
            for w in results["warnings"]:
                print(f"  ⚠️  {w}")
        if results["provenance"]:
            print("\nProvenance:")
            for k, v in results["provenance"].items():
                print(f"  • {k}: {v}")
        print("=" * 60 + "\n")
    except Exception as err:
        logger.error("Validation failed: %s", err)
        sys.exit(1)

