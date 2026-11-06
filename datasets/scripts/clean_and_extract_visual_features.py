"""
PhishGuard – Dataset Quality Cleaning and Batch Visual Feature Extraction (Phase 2, Step 2.2)

Processes the visual acquisition manifest (datasets/screenshots/manifest.jsonl),
applies strict multimodal cleaning (duplicate hash removal, blank/nearly-blank
screenshot detection, domain conflict isolation, and provenance tagging), and
extracts the 30 canonical visual features using ml/features/visual_features.py.

Outputs:
1. datasets/screenshots/cleaning_report.json: Audit of excluded samples, counts, and reasons.
2. datasets/processed/visual_features.csv: Clean, deduplicated dataset with 30 visual features and provenance metadata.

Guarantees:
- Immutability: Raw screenshots and manifest.jsonl remain strictly read-only.
- Provenance Integrity: Tranco candidates remain 'unverified_candidate' (never auto-promoted to verified benign).
- Zero Duplicates: 100% unique screenshot hashes in the final processed dataset.
- Zero Leakage: 100% unique registered apex domains in the final processed dataset.
- Zero Model Training: Does not train or modify any machine learning models.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features.visual_features import (
    VISUAL_FEATURE_NAMES,
    extract_visual_features,
)
from ml.training.dataset_pipeline import (
    KNOWN_CONFLICTING_DOMAINS,
    THREAT_CATEGORY_PHISHING,
    THREAT_CATEGORY_UNVERIFIED,
)

logger = logging.getLogger("phishguard.clean_and_extract_visual_features")

DEFAULT_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "datasets", "screenshots", "manifest.jsonl")
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "datasets", "screenshots", "cleaning_report.json")
DEFAULT_OUTPUT_CSV = os.path.join(PROJECT_ROOT, "datasets", "processed", "visual_features.csv")

# Metadata columns preserved alongside the 30 visual features
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

CSV_FIELDNAMES: list[str] = METADATA_COLUMNS + VISUAL_FEATURE_NAMES


# ── Blank and Nearly Blank Detection ──────────────────────────────────

def is_blank_or_nearly_blank(
    image: np.ndarray | str | os.PathLike,
    std_threshold: float = 2.0,
    edge_threshold: float = 0.0005,
    bright_cutoff: float = 245.0,
    dark_cutoff: float = 15.0,
) -> tuple[bool, str | None, dict[str, float]]:
    """Detect if an image is pure blank or nearly blank (solid color or sparse edges).

    Parameters
    ----------
    image : np.ndarray | str | os.PathLike
        OpenCV BGR image array or image file path.
    std_threshold : float
        Grayscale standard deviation threshold below which an image is considered nearly blank.
    edge_threshold : float
        Canny edge density threshold below which an extreme-luminance image is considered nearly blank.
    bright_cutoff : float
        Mean brightness threshold for nearly pure white images.
    dark_cutoff : float
        Mean brightness threshold for nearly pure dark/black images.

    Returns
    -------
    tuple[bool, str | None, dict[str, float]]
        (is_blank, reason, metrics_dict)
    """
    if isinstance(image, (str, os.PathLike)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            return True, "corrupt_image", {"std_brightness": 0.0, "mean_brightness": 0.0, "edge_density": 0.0}
    elif isinstance(image, np.ndarray):
        img = image
    else:
        return True, "corrupt_image", {"std_brightness": 0.0, "mean_brightness": 0.0, "edge_density": 0.0}

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return True, "corrupt_image", {"std_brightness": 0.0, "mean_brightness": 0.0, "edge_density": 0.0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std_b = float(np.std(gray))
    mean_b = float(np.mean(gray))

    # 1. Pure blank (completely uniform pixel values)
    if std_b == 0.0:
        return True, "pure_blank_image", {
            "std_brightness": 0.0,
            "mean_brightness": round(mean_b, 4),
            "edge_density": 0.0,
        }

    # 2. Extremely low variance (almost uniform color with minor noise)
    if std_b < std_threshold:
        return True, "nearly_blank_low_std", {
            "std_brightness": round(std_b, 4),
            "mean_brightness": round(mean_b, 4),
            "edge_density": 0.0,
        }

    # 3. Sparse edges on nearly uniform background (e.g. white/dark canvas with single speck/dot)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edge_density = float(np.count_nonzero(edges) / (w * h))
    if edge_density < edge_threshold and (mean_b > bright_cutoff or mean_b < dark_cutoff):
        return True, "nearly_blank_sparse_edges", {
            "std_brightness": round(std_b, 4),
            "mean_brightness": round(mean_b, 4),
            "edge_density": round(edge_density, 6),
        }

    return False, None, {
        "std_brightness": round(std_b, 4),
        "mean_brightness": round(mean_b, 4),
        "edge_density": round(edge_density, 6),
    }


# ── Conflict Identification Helpers ───────────────────────────────────

def identify_cross_class_domains(manifest_entries: list[dict[str, Any]]) -> set[str]:
    """Identify registered domains that appear across multiple threat categories or original labels."""
    domain_labels: dict[str, set[str]] = {}
    for entry in manifest_entries:
        domain = entry.get("registered_domain", "").strip().lower()
        if not domain:
            continue
        # Use effective threat_category or original_label
        label = entry.get("threat_category") or entry.get("original_label") or "unknown"
        domain_labels.setdefault(domain, set()).add(label)

    return {domain for domain, labels in domain_labels.items() if len(labels) > 1}


def identify_cross_class_screenshot_hashes(
    manifest_entries: list[dict[str, Any]],
    blank_hashes: set[str] | None = None,
) -> set[str]:
    """Identify screenshot SHA-256 hashes associated with contradictory class labels."""
    blank_hashes = blank_hashes or set()
    hash_labels: dict[str, set[str]] = {}

    for entry in manifest_entries:
        sha = entry.get("screenshot_sha256", "").strip().lower()
        if not sha or sha in blank_hashes:
            continue
        label = entry.get("threat_category") or entry.get("original_label") or "unknown"
        hash_labels.setdefault(sha, set()).add(label)

    return {sha for sha, labels in hash_labels.items() if len(labels) > 1}


# ── Dataset Quality Cleaning ──────────────────────────────────────────

def clean_visual_dataset(
    manifest_entries: list[dict[str, Any]],
    std_threshold: float = 2.0,
    edge_threshold: float = 0.0005,
    project_root: str = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Audit and filter visual manifest records according to data quality and leakage rules.

    Parameters
    ----------
    manifest_entries : list[dict[str, Any]]
        Raw entries parsed from datasets/screenshots/manifest.jsonl.
    std_threshold : float
        Brightness std threshold for blank/nearly blank detection.
    edge_threshold : float
        Edge density threshold for sparse-edge blank detection.
    project_root : str
        Root workspace path for resolving relative screenshot paths.

    Returns
    -------
    tuple[list[dict], list[dict], dict]
        (accepted_samples, excluded_samples, cleaning_summary)
    """
    total_records = len(manifest_entries)
    logger.info("Starting dataset cleaning on %d manifest records...", total_records)

    # 1. Identify cross-class domains across the entire manifest
    cross_class_domains = identify_cross_class_domains(manifest_entries)
    logger.info("Identified %d cross-class registered domains: %s", len(cross_class_domains), cross_class_domains)

    # 2. First pass: inspect images for corrupt and blank/nearly-blank status
    blank_status: dict[str, tuple[bool, str | None, dict[str, float]]] = {}
    blank_hashes: set[str] = set()

    for entry in manifest_entries:
        p = entry.get("screenshot_path", "")
        abs_p = p if os.path.isabs(p) else os.path.abspath(os.path.join(project_root, p))
        is_blank, reason, metrics = is_blank_or_nearly_blank(
            abs_p, std_threshold=std_threshold, edge_threshold=edge_threshold
        )
        blank_status[p] = (is_blank, reason, metrics)
        if is_blank:
            blank_hashes.add(entry.get("screenshot_sha256", "").lower())

    # 3. Identify non-blank screenshot hashes that cross classification labels
    cross_class_hashes = identify_cross_class_screenshot_hashes(manifest_entries, blank_hashes=blank_hashes)
    logger.info("Identified %d non-blank cross-class screenshot hashes.", len(cross_class_hashes))

    # 4. Classification and quarantine assignment
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_domains: set[str] = set()

    reason_counts: Counter[str] = Counter()

    for entry in manifest_entries:
        path = entry.get("screenshot_path", "")
        sha = entry.get("screenshot_sha256", "").strip().lower()
        domain = entry.get("registered_domain", "").strip().lower()

        is_blank, blank_reason, metrics = blank_status.get(path, (False, None, {}))

        primary_reason: str | None = None

        if blank_reason == "corrupt_image":
            primary_reason = "corrupted_image"
        elif is_blank:
            primary_reason = blank_reason  # pure_blank_image, nearly_blank_low_std, nearly_blank_sparse_edges
        elif domain in KNOWN_CONFLICTING_DOMAINS:
            primary_reason = "known_conflict_domain"
        elif domain in cross_class_domains:
            primary_reason = "cross_class_domain_conflict"
        elif sha in cross_class_hashes:
            primary_reason = "cross_class_screenshot_conflict"
        elif sha in seen_hashes:
            primary_reason = "duplicate_screenshot_hash"
        elif domain in seen_domains:
            primary_reason = "duplicate_registered_domain"

        if primary_reason:
            reason_counts[primary_reason] += 1
            excluded_record = {
                "csv_line_num": entry.get("csv_line_num"),
                "original_url": entry.get("original_url"),
                "registered_domain": domain,
                "original_label": entry.get("original_label"),
                "threat_category": entry.get("threat_category"),
                "screenshot_sha256": sha,
                "screenshot_path": path,
                "exclusion_reason": primary_reason,
                "image_metrics": metrics,
            }
            excluded.append(excluded_record)
        else:
            seen_hashes.add(sha)
            seen_domains.add(domain)
            accepted.append(entry)

    accepted_classes = Counter(e.get("threat_category") for e in accepted)
    accepted_provenance = Counter(e.get("label_provenance") for e in accepted)

    summary = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_manifest_records": total_records,
        "total_accepted_records": len(accepted),
        "total_excluded_records": len(excluded),
        "accepted_class_distribution": dict(accepted_classes),
        "accepted_provenance_distribution": dict(accepted_provenance),
        "unique_accepted_registered_domains": len(seen_domains),
        "unique_accepted_screenshot_hashes": len(seen_hashes),
        "duplicate_hashes_in_accepted": len(accepted) - len(seen_hashes),
        "duplicate_domains_in_accepted": len(accepted) - len(seen_domains),
        "exclusion_counts_by_reason": dict(reason_counts),
        "specific_detections": {
            "known_conflicting_domains": sorted(list(KNOWN_CONFLICTING_DOMAINS.intersection(set(e.get("registered_domain", "").lower() for e in manifest_entries)))),
            "cross_class_conflicting_domains": sorted(list(cross_class_domains)),
            "cross_class_conflicting_hashes_count": len(cross_class_hashes),
            "blank_or_nearly_blank_total_count": sum(
                reason_counts[r] for r in ["pure_blank_image", "nearly_blank_low_std", "nearly_blank_sparse_edges"]
            ),
        },
    }

    logger.info(
        "Dataset cleaning complete: %d accepted, %d excluded across %d manifest records.",
        len(accepted), len(excluded), total_records,
    )
    return accepted, excluded, summary


# ── Batch Visual Feature Extraction ───────────────────────────────────

def extract_features_batch(
    samples: list[dict[str, Any]],
    output_csv_path: str | os.PathLike | None = None,
    project_root: str = PROJECT_ROOT,
    log_interval: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract all 30 canonical visual features for cleaned samples.

    Parameters
    ----------
    samples : list[dict[str, Any]]
        Cleaned, non-blank, non-quarantined sample records.
    output_csv_path : str | os.PathLike | None
        Path to save visual_features.csv (optional).
    project_root : str
        Workspace root path.
    log_interval : int
        Progress logging interval.

    Returns
    -------
    tuple[list[dict], dict]
        (extracted_records, extraction_stats)
    """
    total = len(samples)
    logger.info("Extracting 30 visual features for %d cleaned samples...", total)

    extracted_records: list[dict[str, Any]] = []
    failed_extractions: list[dict[str, Any]] = []

    nan_count = 0
    inf_count = 0

    for idx, sample in enumerate(samples, start=1):
        rel_path = sample.get("screenshot_path", "")
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.abspath(os.path.join(project_root, rel_path))

        try:
            feats = extract_visual_features(abs_path)

            # Strict validation: exactly 30 features and valid non-zero dimensions
            if len(feats) != len(VISUAL_FEATURE_NAMES):
                raise ValueError(
                    f"Feature count mismatch: expected {len(VISUAL_FEATURE_NAMES)}, got {len(feats)}"
                )

            if feats.get("width", 0) <= 0 or feats.get("height", 0) <= 0:
                raise ValueError(f"Image decode failed for {abs_path} (zero dimensions)")

            # Check for NaN / Inf
            row_nan = False
            row_inf = False
            for k, val in feats.items():
                if isinstance(val, (int, float)):
                    if math.isnan(val):
                        row_nan = True
                        nan_count += 1
                    elif math.isinf(val):
                        row_inf = True
                        inf_count += 1

            if row_nan or row_inf:
                raise ValueError(f"Extracted features contain NaN ({row_nan}) or Inf ({row_inf})")

            # Assemble complete row
            record: dict[str, Any] = {
                "csv_line_num": sample.get("csv_line_num"),
                "original_url": sample.get("original_url"),
                "normalized_url": sample.get("normalized_url"),
                "registered_domain": sample.get("registered_domain"),
                "original_label": sample.get("original_label"),
                "threat_category": sample.get("threat_category"),
                "label_provenance": sample.get("label_provenance"),
                "screenshot_path": rel_path.replace("\\", "/"),
                "screenshot_sha256": sample.get("screenshot_sha256"),
            }
            # Append in canonical feature order
            for name in VISUAL_FEATURE_NAMES:
                record[name] = feats[name]

            extracted_records.append(record)

        except Exception as exc:
            logger.error("[%d/%d] Feature extraction failed for %s: %s", idx, total, rel_path, exc)
            failed_extractions.append({
                "csv_line_num": sample.get("csv_line_num"),
                "screenshot_path": rel_path,
                "error": str(exc),
            })
            continue

        if idx % log_interval == 0 or idx == total:
            logger.info("Processed %d / %d samples (%.1f%%)", idx, total, (idx / total) * 100.0)

    stats = {
        "total_requested": total,
        "successfully_extracted": len(extracted_records),
        "failed_extractions_count": len(failed_extractions),
        "failed_extractions": failed_extractions,
        "total_features_per_record": len(VISUAL_FEATURE_NAMES),
        "nan_value_count": nan_count,
        "inf_value_count": inf_count,
    }

    # Save to CSV if requested
    if output_csv_path:
        out_p = Path(output_csv_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_csv = str(out_p) + f".tmp.{os.getpid()}"

        with open(tmp_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for rec in extracted_records:
                writer.writerow(rec)

        os.replace(tmp_csv, str(out_p))
        logger.info("Saved %d records to %s", len(extracted_records), out_p)

    return extracted_records, stats


# ── Post-Extraction Validation ────────────────────────────────────────

def validate_processed_visual_dataset(
    records: list[dict[str, Any]],
    csv_path: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Strict verification of the final visual feature dataset against Part C requirements."""
    logger.info("Validating %d processed records against Part C constraints...", len(records))

    seen_hashes: set[str] = set()
    duplicate_hashes: list[str] = []

    domain_classes: dict[str, set[str]] = {}
    quarantined_entries: list[dict[str, Any]] = []

    invalid_feature_counts: list[int] = []
    nan_values_found = 0
    inf_values_found = 0

    for idx, r in enumerate(records):
        sha = r["screenshot_sha256"]
        if sha in seen_hashes:
            duplicate_hashes.append(sha)
        seen_hashes.add(sha)

        domain = r["registered_domain"]
        domain_classes.setdefault(domain, set()).add(r["threat_category"])

        if domain in KNOWN_CONFLICTING_DOMAINS:
            quarantined_entries.append(r)

        # Check features
        feat_keys = [k for k in r.keys() if k in VISUAL_FEATURE_NAMES]
        if len(feat_keys) != 30:
            invalid_feature_counts.append(len(feat_keys))

        for k in VISUAL_FEATURE_NAMES:
            val = r[k]
            if isinstance(val, (int, float)):
                if math.isnan(val):
                    nan_values_found += 1
                elif math.isinf(val):
                    inf_values_found += 1

    cross_class_domains = {d: cs for d, cs in domain_classes.items() if len(cs) > 1}

    csv_exists = os.path.isfile(csv_path) if csv_path else False
    csv_row_count = 0
    if csv_exists:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            csv_row_count = sum(1 for _ in reader)

    validation_result = {
        "all_records_have_30_features": len(invalid_feature_counts) == 0,
        "nan_value_count": nan_values_found,
        "inf_value_count": inf_values_found,
        "duplicate_screenshot_hashes_count": len(duplicate_hashes),
        "unresolved_cross_class_domains_count": len(cross_class_domains),
        "quarantined_samples_count": len(quarantined_entries),
        "total_processed_records": len(records),
        "csv_exists": csv_exists,
        "csv_row_count": csv_row_count,
        "is_valid": (
            len(invalid_feature_counts) == 0
            and nan_values_found == 0
            and inf_values_found == 0
            and len(duplicate_hashes) == 0
            and len(cross_class_domains) == 0
            and len(quarantined_entries) == 0
        ),
    }

    if not validation_result["is_valid"]:
        logger.error("Validation failed! Metrics: %s", validation_result)
    else:
        logger.info("Validation passed! All Part C constraints satisfied.")

    return validation_result


# ── Pipeline Orchestrator ─────────────────────────────────────────────

def run_cleaning_and_feature_extraction(
    manifest_path: str | os.PathLike = DEFAULT_MANIFEST_PATH,
    output_csv_path: str | os.PathLike = DEFAULT_OUTPUT_CSV,
    report_path: str | os.PathLike = DEFAULT_REPORT_PATH,
    std_threshold: float = 2.0,
    edge_threshold: float = 0.0005,
    dry_run: bool = False,
) -> tuple[int, int, dict[str, Any]]:
    """Execute end-to-end dataset cleaning and batch feature extraction."""
    manifest_p = Path(manifest_path).resolve()
    if not manifest_p.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")

    manifest_entries: list[dict[str, Any]] = []
    with open(manifest_p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                manifest_entries.append(json.loads(line))

    logger.info("Loaded %d manifest records from %s", len(manifest_entries), manifest_p)

    # 1. Clean dataset
    accepted, excluded, cleaning_summary = clean_visual_dataset(
        manifest_entries=manifest_entries,
        std_threshold=std_threshold,
        edge_threshold=edge_threshold,
    )

    if dry_run:
        print("\n" + "=" * 65)
        print(" PHISHGUARD DATASET CLEANING (DRY-RUN AUDIT)")
        print("=" * 65)
        print(f" Total Manifest Records        : {cleaning_summary['total_manifest_records']}")
        print(f" Accepted Clean Samples        : {cleaning_summary['total_accepted_records']}")
        print(f"   ├─ Phishing Samples         : {cleaning_summary['accepted_class_distribution'].get(THREAT_CATEGORY_PHISHING, 0)}")
        print(f"   └─ Legitimate Candidates    : {cleaning_summary['accepted_class_distribution'].get(THREAT_CATEGORY_UNVERIFIED, 0)}")
        print(f" Total Excluded Samples        : {cleaning_summary['total_excluded_records']}")
        print(" Exclusion Breakdown:")
        for reason, cnt in cleaning_summary["exclusion_counts_by_reason"].items():
            print(f"   ├─ {reason:<28}: {cnt}")
        print(f" Unique Registered Domains     : {cleaning_summary['unique_accepted_registered_domains']}")
        print(f" Unique Screenshot Hashes      : {cleaning_summary['unique_accepted_screenshot_hashes']}")
        print("=" * 65)
        print("[DRY-RUN COMPLETE] No files written.\n")
        return len(accepted), len(excluded), cleaning_summary

    # 2. Batch feature extraction
    extracted_records, extraction_stats = extract_features_batch(
        samples=accepted,
        output_csv_path=output_csv_path,
    )

    # 3. Validation
    validation_stats = validate_processed_visual_dataset(
        records=extracted_records,
        csv_path=output_csv_path,
    )

    # 4. Generate Comprehensive Cleaning Report
    full_report = {
        "metadata": {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_step": "Phase 2, Step 2.2: Cleaning & Batch Visual Feature Extraction",
            "source_manifest_path": str(manifest_p).replace("\\", "/"),
            "output_csv_path": str(Path(output_csv_path).resolve()).replace("\\", "/"),
            "quality_report_path": str(Path(report_path).resolve()).replace("\\", "/"),
        },
        "summary": cleaning_summary,
        "feature_extraction_summary": extraction_stats,
        "validation_summary": validation_stats,
        "excluded_records": excluded,
    }

    rep_p = Path(report_path).resolve()
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_p, "w", encoding="utf-8") as f_rep:
        json.dump(full_report, f_rep, indent=2)

    logger.info("Saved cleaning report to %s", rep_p)
    return len(accepted), len(excluded), full_report


# ── Command Line Interface ──────────────────────────────────────────

def build_cli() -> argparse.ArgumentParser:
    """Build CLI parser for visual dataset cleaning and feature extraction."""
    parser = argparse.ArgumentParser(
        description="PhishGuard – Dataset Quality Cleaning and Batch Visual Feature Extraction"
    )
    parser.add_argument(
        "--manifest-path",
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to manifest.jsonl (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to output visual_features.csv (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help=f"Path to output cleaning_report.json (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=2.0,
        help="Luminance standard deviation threshold for blank image detection (default: 2.0)",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0005,
        help="Canny edge density threshold for sparse edge blank detection (default: 0.0005)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate dataset cleaning without extracting features or saving files.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_cli()
    args = parser.parse_args()

    try:
        accepted_cnt, excluded_cnt, report = run_cleaning_and_feature_extraction(
            manifest_path=args.manifest_path,
            output_csv_path=args.output_csv,
            report_path=args.report_path,
            std_threshold=args.std_threshold,
            edge_threshold=args.edge_threshold,
            dry_run=args.dry_run,
        )
        print(f"\n[SUCCESS] Cleaning and extraction finished: {accepted_cnt} accepted, {excluded_cnt} excluded.")
        return 0
    except Exception as exc:
        logger.exception("Pipeline execution failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
