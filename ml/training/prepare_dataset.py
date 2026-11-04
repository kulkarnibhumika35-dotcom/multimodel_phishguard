"""
PhishGuard – Dataset Preparation

Reads a raw CSV of URLs with labels, cleans/normalises the data,
extracts the 20 URL features, and writes a processed CSV.

No data is invented, generated, or downloaded.
"""

from __future__ import annotations

import csv
import logging
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.features.url_features import FEATURE_NAMES, extract_url_features

logger = logging.getLogger(__name__)

# ── Label normalisation map ─────────────────────────────────────

_LABEL_MAP: dict[str, int] = {
    # Textual labels (case-insensitive lookup)
    "legitimate": 0,
    "benign":     0,
    "safe":       0,
    "good":       0,
    "phishing":   1,
    "malicious":  1,
    "bad":        1,
    "spam":       1,
    # Binary strings
    "0": 0,
    "1": 1,
}


# ── Public API ──────────────────────────────────────────────────

def prepare_dataset(
    input_path: str,
    output_path: str,
    url_column: str = "url",
    label_column: str = "label",
) -> dict[str, int]:
    """Read *input_path* CSV, clean it, extract features, and write to *output_path*.

    Parameters
    ----------
    input_path : str
        Path to the raw CSV file.
    output_path : str
        Destination path for the processed CSV.
    url_column : str
        Name of the column containing URLs.
    label_column : str
        Name of the column containing labels.

    Returns
    -------
    dict
        Statistics: ``total_rows``, ``valid_rows``, ``dropped_missing``,
        ``dropped_invalid_label``, ``dropped_duplicate``.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    ValueError
        If the CSV is missing the required columns.
    """
    # ── Validate input file ─────────────────────────────────────
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = _read_csv(input_path, url_column, label_column)
    total_rows = len(rows)

    # ── 1. Drop rows with missing URL ───────────────────────────
    before = len(rows)
    rows = [r for r in rows if r["url"] and r["url"].strip()]
    dropped_missing = before - len(rows)

    # ── 2. Normalise labels & drop invalid ──────────────────────
    before = len(rows)
    normalised: list[dict] = []
    for r in rows:
        label = _normalise_label(r["label"])
        if label is not None:
            normalised.append({"url": r["url"].strip(), "label": label})
    rows = normalised
    dropped_invalid_label = before - len(rows)

    # ── 3. Deduplicate by URL ───────────────────────────────────
    before = len(rows)
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped.append(r)
    rows = deduped
    dropped_duplicate = before - len(rows)

    # ── 4. Extract features ─────────────────────────────────────
    processed: list[dict] = []
    for r in rows:
        features = extract_url_features(r["url"])
        row = {"url": r["url"], "label": r["label"]}
        row.update(features)
        processed.append(row)

    # ── 5. Write output CSV ─────────────────────────────────────
    _write_csv(output_path, processed)

    stats = {
        "total_rows":           total_rows,
        "valid_rows":           len(processed),
        "dropped_missing":      dropped_missing,
        "dropped_invalid_label": dropped_invalid_label,
        "dropped_duplicate":    dropped_duplicate,
    }
    logger.info("Dataset prepared: %s", stats)
    return stats


# ── Internal helpers ────────────────────────────────────────────

def _read_csv(path: str, url_col: str, label_col: str) -> list[dict]:
    """Read CSV and return list of ``{url, label}`` dicts."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty or has no header: {path}")

        # Case-insensitive column lookup
        col_map = {c.strip().lower(): c for c in reader.fieldnames}
        url_key = col_map.get(url_col.strip().lower())
        label_key = col_map.get(label_col.strip().lower())

        if url_key is None:
            raise ValueError(
                f"URL column '{url_col}' not found in CSV. "
                f"Available columns: {list(reader.fieldnames)}"
            )
        if label_key is None:
            raise ValueError(
                f"Label column '{label_col}' not found in CSV. "
                f"Available columns: {list(reader.fieldnames)}"
            )

        return [
            {"url": (row.get(url_key) or ""), "label": (row.get(label_key) or "")}
            for row in reader
        ]


def _normalise_label(raw: str) -> int | None:
    """Map a raw label string to 0 or 1, or ``None`` if unrecognised."""
    if raw is None:
        return None
    cleaned = str(raw).strip().lower()
    return _LABEL_MAP.get(cleaned)


def _write_csv(path: str, rows: list[dict]) -> None:
    """Write *rows* to a CSV file, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fieldnames = ["url", "label"] + list(FEATURE_NAMES)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── CLI entry point ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Prepare a phishing URL dataset.")
    parser.add_argument("input", help="Path to raw CSV")
    parser.add_argument("output", help="Path to write processed CSV")
    parser.add_argument("--url-column", default="url", help="URL column name")
    parser.add_argument("--label-column", default="label", help="Label column name")
    args = parser.parse_args()

    stats = prepare_dataset(args.input, args.output, args.url_column, args.label_column)
    print(stats)

