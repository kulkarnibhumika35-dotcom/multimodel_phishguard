"""
PhishGuard – Zenodo V2.0 Dataset Acquisition and Provenance Pipeline

Target:
- Dataset: "Phishing and Legitimate Websites Dataset" (V2.0)
- DOI: 10.5281/zenodo.23477065 (Parent Concept DOI: 10.5281/zenodo.21379701)
- Archive: "Phishing dataset V2.0.zip"
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

Capabilities:
1. Selective Acquisition (Recommended):
   - Inspects remote ZIP central directory over HTTP range requests (<2 MB data).
   - Fetches CSV metadata over HTTP range request (~2.4 MB compressed).
   - Selects approximately balanced phishing and legitimate candidates (configurable via --max-samples, default 5,000).
   - Enforces registered domain deduplication (1 sample per registered domain).
   - Downloads ONLY selected screenshot files directly via HTTP Range requests without downloading the 4.48 GB archive.
   - Supports dry-run mode (--dry-run) to inspect candidate counts, domain diversity, and estimated download size without downloading.
2. Full Archive Acquisition:
   - Downloads the full V2.0 archive (resumable with chunking, retry backoff, and integrity verification).
   - Safely extracts into datasets/screenshots/raw/ with zip-slip prevention.
3. Common Guarantees:
   - Reconstructs screenshot-to-URL mapping using 1-based row ID prefixes.
   - Treats Tranco legitimate samples as unverified candidates (never auto-converts to verified benign).
   - Preserves existing canonical URL normalization (ensure_apex_www=False).
   - Extracts registered apex domains using existing extract_registered_domain().
   - Generates cryptographic manifest.jsonl and quality_report.json.
   - Does NOT train any models or modify the URL Random Forest model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import logging
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.training.dataset_pipeline import (
    KNOWN_CONFLICTING_DOMAINS,
    THREAT_CATEGORY_PHISHING,
    THREAT_CATEGORY_UNVERIFIED,
    canonical_normalize_url,
    extract_registered_domain,
)

logger = logging.getLogger("phishguard.acquire_visual_dataset")

# ── Zenodo V2.0 Target Constants ─────────────────────────────────────
ZENODO_RECORD_ID = "23477065"
ZENODO_DOI = "10.5281/zenodo.23477065"
ZENODO_CONCEPT_DOI = "10.5281/zenodo.21379701"
ZENODO_FILE_NAME = "Phishing dataset V2.0.zip"
ZENODO_DOWNLOAD_URL = (
    "https://zenodo.org/api/records/23477065/files/Phishing%20dataset%20V2.0.zip/content"
)
ZENODO_FILE_SIZE = 4476190126  # ~4.17 GiB / 4.48 GB
ZENODO_EXPECTED_MD5 = "6177616951de95ee82a3aa30f4970ada"

DEFAULT_DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "datasets", "downloads")
DEFAULT_EXTRACT_DIR = os.path.join(PROJECT_ROOT, "datasets", "screenshots", "raw")
DEFAULT_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "datasets", "screenshots", "manifest.jsonl")
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "datasets", "screenshots", "quality_report.json")

# Filename regex: e.g. 10001_bluesystem.me.png or 4_sites.google.com_view_xyz.png
SCREENSHOT_REGEX = re.compile(r"^(\d+)_(.+)\.(png|jpg|jpeg)$", re.IGNORECASE)


# ── Custom Exceptions ───────────────────────────────────────────────

class RangeDownloadError(Exception):
    """Raised when an HTTP Range request fails or is unsupported by the remote server."""
    pass


# ── Cryptographic Hashing Helpers ────────────────────────────────────

def compute_file_sha256(filepath: str | os.PathLike, chunk_size: int = 65536) -> str:
    """Compute the hexadecimal SHA-256 digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_file_md5(filepath: str | os.PathLike, chunk_size: int = 65536) -> str:
    """Compute the hexadecimal MD5 digest of a file on disk."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


# ── Image Inspection ────────────────────────────────────────────────

def inspect_image(filepath: str | os.PathLike) -> dict[str, Any]:
    """Inspect and validate an image file using OpenCV without network access."""
    p = str(filepath)
    if not os.path.isfile(p):
        return {"is_valid": False, "width": 0, "height": 0, "error": "file_not_found"}

    if os.path.getsize(p) == 0:
        return {"is_valid": False, "width": 0, "height": 0, "error": "empty_file"}

    try:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            return {"is_valid": False, "width": 0, "height": 0, "error": "opencv_decode_failed"}

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return {"is_valid": False, "width": 0, "height": 0, "error": "invalid_dimensions"}

        return {"is_valid": True, "width": int(w), "height": int(h), "error": None}
    except Exception as exc:
        return {"is_valid": False, "width": 0, "height": 0, "error": f"exception_{exc}"}


# ── Remote Seekable ZIP Reader (HTTP Range) ──────────────────────────

class RemoteZipReader:
    """A seekable file-like stream using HTTP Range requests for remote ZIP inspection."""

    def __init__(self, url: str, user_agent: str = "PhishGuard-Dataset-Acquisition/2.0", timeout: float = 30.0):
        self.url = url
        self.user_agent = user_agent
        self.timeout = timeout
        self.pos = 0
        self.size = 0

        # Query total size via initial range probe with retries
        req = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Range": "bytes=0-0"},
        )
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    cr = resp.headers.get("Content-Range")
                    if cr and "/" in cr:
                        self.size = int(cr.split("/")[-1])
                    else:
                        cl = resp.headers.get("Content-Length")
                        self.size = int(cl) if cl else 0
                    break
            except (
                http.client.HTTPException,
                urllib.error.URLError,
                ConnectionError,
                TimeoutError,
                socket.timeout,
                OSError,
            ) as exc:
                last_exc = exc
                if attempt < 4:
                    time.sleep(min(2**attempt, 15))

        if self.size <= 0:
            raise RangeDownloadError(
                f"Failed to query remote file size via HTTP Range from {url}: {last_exc or 'size reported as 0'}"
            )

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        end = self.size - 1 if size < 0 else min(self.pos + size - 1, self.size - 1)
        if self.pos > end:
            return b""

        req = urllib.request.Request(
            self.url,
            headers={
                "User-Agent": self.user_agent,
                "Range": f"bytes={self.pos}-{end}",
            },
        )
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
                expected_len = end - self.pos + 1
                if len(data) != expected_len and size > 0:
                    raise RangeDownloadError(
                        f"Incomplete range read: expected {expected_len} bytes, got {len(data)}"
                    )
                self.pos += len(data)
                return data
            except (
                http.client.HTTPException,
                urllib.error.URLError,
                ConnectionError,
                TimeoutError,
                socket.timeout,
                RangeDownloadError,
                OSError,
            ) as exc:
                last_exc = exc
                if attempt < 4:
                    time.sleep(min(2**attempt, 15))

        raise RangeDownloadError(
            f"Failed to read byte range {self.pos}-{end} from {self.url} after 5 attempts: {last_exc}"
        ) from last_exc


# ── Single Entry Range Extraction ────────────────────────────────────

def fetch_entry_bytes_range(
    url: str,
    zinfo: zipfile.ZipInfo,
    max_retries: int = 5,
    timeout: float = 30.0,
    user_agent: str = "PhishGuard-Dataset-Acquisition/2.0",
    backoff_base: float = 1.0,
) -> bytes:
    """Fetch and decompress a single ZIP entry using a targeted HTTP Range request.

    Parameters
    ----------
    url : str
        Remote ZIP download URL.
    zinfo : zipfile.ZipInfo
        ZIP entry info containing header_offset, compress_size, file_size, compress_type, CRC.
    max_retries : int
        Number of retries on transient network errors (default: 5).
    timeout : float
        HTTP request timeout in seconds.
    user_agent : str
        HTTP User-Agent string.
    backoff_base : float
        Base multiplier in seconds for exponential backoff (default: 1.0).

    Returns
    -------
    bytes
        Uncompressed entry payload.
    """
    # Fetch local file header (30 bytes) + filename + extra + compressed payload
    fetch_len = 30 + len(zinfo.filename) + 256 + zinfo.compress_size
    start_byte = zinfo.header_offset
    end_byte = start_byte + fetch_len - 1

    last_error: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Range": f"bytes={start_byte}-{end_byte}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # 1. HTTP status must be 206
                if resp.status != 206:
                    raise RangeDownloadError(
                        f"HTTP status {resp.status} is not 206 Partial Content for {zinfo.filename} "
                        f"(requested range {start_byte}-{end_byte})"
                    )

                # 2. Content-Range must match the requested byte range
                cr = resp.headers.get("Content-Range") if hasattr(resp, "headers") and resp.headers else None
                if not cr or not isinstance(cr, str):
                    raise RangeDownloadError(
                        f"Missing or invalid Content-Range header in HTTP 206 response for {zinfo.filename}"
                    )
                cr_match = re.match(r"^bytes\s+(\d+)-(\d+)/(?:\d+|\*)$", cr.strip(), re.IGNORECASE)
                if not cr_match:
                    raise RangeDownloadError(
                        f"Malformed Content-Range header '{cr}' for {zinfo.filename}"
                    )
                cr_start = int(cr_match.group(1))
                cr_end = int(cr_match.group(2))
                if cr_start != start_byte or cr_end != end_byte:
                    raise RangeDownloadError(
                        f"Content-Range mismatch for {zinfo.filename}: expected bytes {start_byte}-{end_byte}, got {cr_start}-{cr_end}"
                    )

                # 3. Read raw payload safely (handles IncompleteRead via retry block)
                raw_block = resp.read()

            # 4. Received byte count must match the expected length
            if len(raw_block) != fetch_len:
                raise RangeDownloadError(
                    f"Received byte count mismatch for {zinfo.filename}: expected {fetch_len} bytes, got {len(raw_block)}"
                )

            # 5. Check local file header signature (PK\x03\x04)
            if raw_block[:4] != b"PK\x03\x04":
                raise RangeDownloadError(
                    f"Corrupt ZIP local header at offset {start_byte}: signature {raw_block[:4]!r}"
                )

            fn_len = int.from_bytes(raw_block[26:28], "little")
            extra_len = int.from_bytes(raw_block[28:30], "little")
            data_start = 30 + fn_len + extra_len

            if len(raw_block) < data_start + zinfo.compress_size:
                raise RangeDownloadError(
                    f"Incomplete entry payload for {zinfo.filename}: expected at least {data_start + zinfo.compress_size} bytes, got {len(raw_block)}"
                )

            compressed_data = raw_block[data_start : data_start + zinfo.compress_size]

            # 6. Decompress
            if zinfo.compress_type == zipfile.ZIP_DEFLATED:
                uncompressed = zlib.decompress(compressed_data, -zlib.MAX_WBITS)
            elif zinfo.compress_type == zipfile.ZIP_STORED:
                uncompressed = compressed_data
            else:
                raise RangeDownloadError(f"Unsupported ZIP compression type: {zinfo.compress_type}")

            # 7. Verify CRC-32 and uncompressed length
            actual_crc = zlib.crc32(uncompressed)
            if actual_crc != zinfo.CRC:
                raise RangeDownloadError(
                    f"CRC-32 mismatch for {zinfo.filename}: expected {zinfo.CRC}, got {actual_crc}"
                )

            if len(uncompressed) != zinfo.file_size:
                raise RangeDownloadError(
                    f"Uncompressed size mismatch for {zinfo.filename}: expected {zinfo.file_size}, got {len(uncompressed)}"
                )

            return uncompressed

        except (
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            http.client.HTTPException,
            urllib.error.URLError,
            ConnectionError,
            TimeoutError,
            socket.timeout,
            RangeDownloadError,
            zlib.error,
            OSError,
        ) as exc:
            last_error = exc
            backoff_sec = min(backoff_base * (2**attempt), 30.0) if backoff_base > 0 else 0.0
            logger.warning(
                "Attempt %d/%d failed for entry %s (%s: %s). Retrying in %.1fs...",
                attempt + 1,
                max_retries,
                zinfo.filename,
                type(exc).__name__,
                exc,
                backoff_sec,
            )
            if attempt < max_retries - 1 and backoff_sec > 0:
                time.sleep(backoff_sec)

    raise RangeDownloadError(
        f"Failed to fetch ZIP entry {zinfo.filename} via HTTP Range after {max_retries} attempts: {last_error}"
    )


# ── Resumable Full Archive Download ──────────────────────────────────

def download_file_with_resume(
    url: str,
    dest_path: str,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
    max_retries: int = 5,
    chunk_size: int = 1048576,
) -> str:
    """Download a large file supporting HTTP Range resumption, retries, and checksum checks."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    part_path = dest_path + ".part"

    if os.path.isfile(dest_path):
        current_size = os.path.getsize(dest_path)
        if expected_size is None or current_size == expected_size:
            logger.info("Destination file already exists (%d bytes). Verifying checksum...", current_size)
            if expected_md5 and compute_file_md5(dest_path) != expected_md5.lower():
                logger.warning("Existing file MD5 mismatch. Re-downloading.")
            elif expected_sha256 and compute_file_sha256(dest_path) != expected_sha256.lower():
                logger.warning("Existing file SHA-256 mismatch. Re-downloading.")
            else:
                logger.info("Existing file checksum valid: %s", dest_path)
                return dest_path

    retries = 0
    while retries < max_retries:
        existing_bytes = os.path.getsize(part_path) if os.path.isfile(part_path) else 0

        headers: dict[str, str] = {"User-Agent": "PhishGuard-Dataset-Acquisition/2.0"}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
            logger.info("Resuming download from byte offset %d...", existing_bytes)
        else:
            logger.info("Starting fresh download for %s...", url)

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                status = response.status
                content_range = response.headers.get("Content-Range")
                content_length = response.headers.get("Content-Length")

                if status == 206:
                    mode = "ab"
                    total_bytes = int(content_range.split("/")[-1]) if content_range else expected_size
                elif status == 200:
                    mode = "wb"
                    existing_bytes = 0
                    total_bytes = int(content_length) if content_length else expected_size
                else:
                    raise IOError(f"Unexpected HTTP status {status} from server")

                with open(part_path, mode) as f_out:
                    downloaded = existing_bytes
                    last_log_time = time.time()
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_log_time >= 5.0:
                            pct = (downloaded / total_bytes * 100.0) if total_bytes else 0.0
                            logger.info(
                                "Progress: %d / %s bytes (%.2f%%)",
                                downloaded,
                                total_bytes or "unknown",
                                pct,
                            )
                            last_log_time = now

            part_size = os.path.getsize(part_path)
            if expected_size and part_size != expected_size:
                logger.warning(
                    "Part file size (%d) differs from expected (%d). Retrying...",
                    part_size,
                    expected_size,
                )
                retries += 1
                time.sleep(2**retries)
                continue

            if expected_md5:
                actual_md5 = compute_file_md5(part_path)
                if actual_md5.lower() != expected_md5.lower():
                    raise ValueError(f"MD5 checksum mismatch! Expected {expected_md5}, got {actual_md5}")
                logger.info("MD5 checksum verified: %s", actual_md5)

            if expected_sha256:
                actual_sha256 = compute_file_sha256(part_path)
                if actual_sha256.lower() != expected_sha256.lower():
                    raise ValueError(f"SHA-256 mismatch! Expected {expected_sha256}, got {actual_sha256}")
                logger.info("SHA-256 checksum verified: %s", actual_sha256)

            if os.path.isfile(dest_path):
                os.remove(dest_path)
            os.rename(part_path, dest_path)
            logger.info("Download completed and verified: %s", dest_path)
            return dest_path

        except (
            urllib.error.URLError,
            TimeoutError,
            IOError,
            http.client.HTTPException,
            ConnectionError,
            socket.timeout,
        ) as exc:
            retries += 1
            wait_sec = min(2**retries, 30)
            logger.warning(
                "Download error: %s. Retry %d/%d after %ds...",
                exc,
                retries,
                max_retries,
                wait_sec,
            )
            time.sleep(wait_sec)

    raise IOError(f"Failed to download {url} after {max_retries} retries.")


# ── Safe ZIP Extraction ─────────────────────────────────────────────

def safe_extract_archive(
    zip_path: str | os.PathLike,
    extract_to_dir: str | os.PathLike,
) -> list[str]:
    """Safely extract all files from a ZIP archive, preventing Zip-Slip vulnerabilities."""
    extract_dir = os.path.abspath(str(extract_to_dir))
    os.makedirs(extract_dir, exist_ok=True)

    extracted_files: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        infolist = zf.infolist()
        total_items = len(infolist)
        logger.info("Extracting %d items from %s into %s...", total_items, zip_path, extract_dir)

        for idx, member in enumerate(infolist, start=1):
            member_path = os.path.abspath(os.path.join(extract_dir, member.filename))
            if not member_path.startswith(extract_dir + os.sep) and member_path != extract_dir:
                raise SecurityError(f"Attempted path traversal in ZIP member: {member.filename}")

            if member.is_dir():
                os.makedirs(member_path, exist_ok=True)
                continue

            os.makedirs(os.path.dirname(member_path), exist_ok=True)
            with zf.open(member) as src, open(member_path, "wb") as dst:
                while chunk := src.read(65536):
                    dst.write(chunk)

            extracted_files.append(member_path)
            if idx % 2000 == 0 or idx == total_items:
                logger.info("Extracted %d / %d files (%.1f%%)", idx, total_items, idx / total_items * 100.0)

    return extracted_files


# ── Parsing Helpers ──────────────────────────────────────────────────

def parse_csv_records(csv_content_or_path: str | bytes | os.PathLike) -> list[dict[str, Any]]:
    """Parse merged_features_final.csv lines, returning line_num, url, and label."""
    if isinstance(csv_content_or_path, (bytes, bytearray)):
        lines = csv_content_or_path.decode("utf-8", errors="replace").splitlines()
    elif isinstance(csv_content_or_path, str) and "\n" in csv_content_or_path:
        lines = csv_content_or_path.splitlines()
    else:
        with open(str(csv_content_or_path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

    records: list[dict[str, Any]] = []
    reader = csv.reader(lines, delimiter=";")
    header = next(reader, None)
    if not header:
        return records

    cols = [c.strip().lower() for c in header]
    url_idx = cols.index("url") if "url" in cols else 0
    label_idx = cols.index("label") if "label" in cols else -1

    for line_num, row in enumerate(reader, start=2):
        if not row or not any(row):
            continue
        raw_url = row[url_idx].strip() if len(row) > url_idx else ""
        raw_label = row[label_idx].strip().lower() if (label_idx >= 0 and len(row) > label_idx) else ""
        records.append({
            "line_num": line_num,
            "url": raw_url,
            "label": raw_label,
            "raw_row": row,
        })
    return records


def find_screenshot_files(images_dir: str | os.PathLike) -> list[str]:
    """Recursively collect all image files (.png, .jpg, .jpeg) in the target directory."""
    images: list[str] = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(root, f))
    return sorted(images)


# ── Selective Balanced Sampling Engine ───────────────────────────────

def select_balanced_samples(
    csv_records: list[dict[str, Any]],
    image_entries: list[Any],  # list of zipfile.ZipInfo or local file paths
    max_samples: int = 5000,
    max_per_domain: int = 1,
    balance_classes: bool = True,
    enforce_unique_domains: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select an approximately balanced set of phishing and legitimate candidates with domain deduplication.

    Parameters
    ----------
    csv_records : list[dict]
        Parsed CSV records indexed by line_num.
    image_entries : list
        List of ZipInfo objects or local image file paths available in the archive/directory.
    max_samples : int
        Maximum total samples to acquire.
    max_per_domain : int
        Maximum number of samples permitted from the same registered domain (default: 1).
    balance_classes : bool
        If True, ensures approximately equal numbers of phishing and legitimate samples
        by balancing to min(phishing_count, legitimate_count).
    enforce_unique_domains : bool
        If False, permits domain repetitions up to max_samples limit.

    Returns
    -------
    tuple[list[dict], dict]
        Selected items with mapped metadata, and a dictionary of selection statistics.
    """
    if not enforce_unique_domains:
        max_per_domain = 999999
    records_by_line: dict[int, dict[str, Any]] = {r["line_num"]: r for r in csv_records}

    phishing_candidates: list[dict[str, Any]] = []
    legitimate_candidates: list[dict[str, Any]] = []

    # Map available images to CSV records
    for item in image_entries:
        fname = item.filename if hasattr(item, "filename") else os.path.basename(str(item))
        m = SCREENSHOT_REGEX.match(os.path.basename(fname))
        if not m:
            continue

        line_num = int(m.group(1))
        if line_num not in records_by_line:
            continue

        csv_rec = records_by_line[line_num]
        raw_url = csv_rec["url"]
        csv_label = csv_rec["label"]

        # Determine class from folder or CSV label
        parent_dir = os.path.basename(os.path.dirname(fname)).lower()
        effective_label = parent_dir if parent_dir in ("phishing", "legitimate") else csv_label

        norm_url = canonical_normalize_url(raw_url, default_scheme="https", ensure_apex_www=False) or raw_url
        reg_domain = extract_registered_domain(norm_url)

        compress_size = getattr(item, "compress_size", 0)
        file_size = getattr(item, "file_size", os.path.getsize(str(item)) if os.path.isfile(str(item)) else 0)

        candidate = {
            "csv_line_num": line_num,
            "original_url": raw_url,
            "normalized_url": norm_url,
            "registered_domain": reg_domain,
            "original_label": csv_label,
            "effective_label": effective_label,
            "image_entry": item,
            "filename": fname,
            "compress_size": compress_size,
            "file_size": file_size,
        }

        if effective_label == "phishing":
            phishing_candidates.append(candidate)
        elif effective_label == "legitimate":
            legitimate_candidates.append(candidate)

    target_per_class = max_samples // 2

    # Selection with registered domain frequency capping
    def _select_from_pool(pool: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        domain_counts: dict[str, int] = {}

        for c in pool:
            dom = c["registered_domain"]
            if not dom:
                continue
            count = domain_counts.get(dom, 0)
            if count < max_per_domain:
                domain_counts[dom] = count + 1
                selected.append(c)
                if len(selected) >= limit:
                    break

        return selected

    selected_phishing = _select_from_pool(phishing_candidates, target_per_class)
    selected_legitimate = _select_from_pool(legitimate_candidates, target_per_class)

    # Balance classes to approximately equal numbers if requested
    if balance_classes:
        target_balanced = min(len(selected_phishing), len(selected_legitimate))
        selected_phishing = selected_phishing[:target_balanced]
        selected_legitimate = selected_legitimate[:target_balanced]

    combined = selected_phishing + selected_legitimate
    combined.sort(key=lambda x: x["csv_line_num"])

    # Calculate statistics
    total_compressed = sum(c["compress_size"] for c in combined)
    total_uncompressed = sum(c["file_size"] for c in combined)
    unique_domains = len(set(c["registered_domain"] for c in combined if c["registered_domain"]))
    quarantined = sum(1 for c in combined if c["registered_domain"].lower() in KNOWN_CONFLICTING_DOMAINS)

    stats = {
        "requested_max_samples": max_samples,
        "max_per_domain": max_per_domain,
        "balance_classes": balance_classes,
        "selected_total": len(combined),
        "selected_phishing": len(selected_phishing),
        "selected_legitimate_unverified": len(selected_legitimate),
        "quarantined_conflicts_count": quarantined,
        "unique_registered_domains": unique_domains,
        "estimated_compressed_download_bytes": total_compressed,
        "estimated_compressed_download_mb": round(total_compressed / (1024**2), 2),
        "estimated_uncompressed_bytes": total_uncompressed,
        "estimated_uncompressed_mb": round(total_uncompressed / (1024**2), 2),
        "available_phishing_candidates": len(phishing_candidates),
        "available_legitimate_candidates": len(legitimate_candidates),
    }

    return combined, stats


# ── Full Reconstruct & Validate (Local Extracted Files) ──────────────

def reconstruct_and_validate_dataset(
    csv_path: str | os.PathLike,
    images_dir: str | os.PathLike,
    manifest_path: str | os.PathLike = DEFAULT_MANIFEST_PATH,
    report_path: str | os.PathLike = DEFAULT_REPORT_PATH,
    source_name: str = "zenodo_23477065",
) -> tuple[int, dict[str, Any]]:
    """Reconstruct screenshot-to-URL mappings, validate integrity, and generate manifest and quality report."""
    logger.info("Parsing CSV records from %s...", csv_path)
    csv_records = parse_csv_records(csv_path)
    total_csv_records = len(csv_records)
    logger.info("Found %d CSV records in %s.", total_csv_records, csv_path)

    records_by_line: dict[int, dict[str, Any]] = {r["line_num"]: r for r in csv_records}

    logger.info("Scanning screenshot files in %s...", images_dir)
    image_paths = find_screenshot_files(images_dir)
    total_screenshots = len(image_paths)
    logger.info("Found %d screenshot image files.", total_screenshots)

    seen_urls: set[str] = set()
    duplicate_url_count = 0
    seen_screenshot_hashes: dict[str, str] = {}
    duplicate_screenshot_count = 0

    invalid_filename_count = 0
    out_of_bounds_line_count = 0
    label_mismatch_count = 0
    corrupted_image_count = 0
    quarantined_conflict_count = 0
    unverified_candidate_count = 0
    phishing_pairs_count = 0
    mapped_line_numbers: set[int] = set()

    manifest_entries: list[dict[str, Any]] = []

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        m = SCREENSHOT_REGEX.match(fname)
        if not m:
            logger.warning("Invalid screenshot filename format: %s", fname)
            invalid_filename_count += 1
            continue

        line_num = int(m.group(1))
        if line_num not in records_by_line:
            logger.warning("Screenshot line ID %d not in CSV records range: %s", line_num, fname)
            out_of_bounds_line_count += 1
            continue

        csv_record = records_by_line[line_num]
        raw_url = csv_record["url"]
        csv_label = csv_record["label"]
        mapped_line_numbers.add(line_num)

        parent_dir_name = os.path.basename(os.path.dirname(img_path)).lower()
        if parent_dir_name in ("phishing", "legitimate"):
            if parent_dir_name != csv_label:
                logger.warning(
                    "Directory label mismatch for line %d: folder='%s' vs csv='%s'",
                    line_num,
                    parent_dir_name,
                    csv_label,
                )
                label_mismatch_count += 1

        if raw_url in seen_urls:
            duplicate_url_count += 1
        seen_urls.add(raw_url)

        img_sha256 = compute_file_sha256(img_path)
        if img_sha256 in seen_screenshot_hashes:
            duplicate_screenshot_count += 1
        else:
            seen_screenshot_hashes[img_sha256] = img_path

        img_info = inspect_image(img_path)
        if not img_info["is_valid"]:
            corrupted_image_count += 1

        norm_url = canonical_normalize_url(raw_url, default_scheme="https", ensure_apex_www=False) or raw_url
        reg_domain = extract_registered_domain(norm_url)

        # Requirement: Treat Tranco legitimate samples as unverified candidates
        if csv_label == "phishing":
            threat_cat = THREAT_CATEGORY_PHISHING
            label_provenance = "phishtank_verified"
            quarantine_status = "none"
            phishing_pairs_count += 1
        elif csv_label == "legitimate":
            threat_cat = THREAT_CATEGORY_UNVERIFIED
            label_provenance = "tranco_candidate_unverified"
            quarantine_status = "unverified_candidate"
            unverified_candidate_count += 1
        else:
            threat_cat = "unknown"
            label_provenance = "unspecified"
            quarantine_status = "unverified"

        if reg_domain.lower() in KNOWN_CONFLICTING_DOMAINS:
            quarantine_status = "quarantined_conflict"
            quarantined_conflict_count += 1

        try:
            rel_screenshot_path = os.path.relpath(img_path, PROJECT_ROOT).replace("\\", "/")
        except ValueError:
            rel_screenshot_path = str(img_path).replace("\\", "/")

        entry = {
            "csv_line_num": line_num,
            "original_url": raw_url,
            "normalized_url": norm_url,
            "registered_domain": reg_domain,
            "original_label": csv_label,
            "threat_category": threat_cat,
            "screenshot_path": rel_screenshot_path,
            "screenshot_sha256": img_sha256,
            "image_width": img_info["width"],
            "image_height": img_info["height"],
            "is_valid": img_info["is_valid"],
            "dataset_source": source_name,
            "label_provenance": label_provenance,
            "quarantine_status": quarantine_status,
        }
        manifest_entries.append(entry)

    missing_screenshots_count = total_csv_records - len(mapped_line_numbers)
    unique_domains = len(set(e["registered_domain"] for e in manifest_entries if e["registered_domain"]))

    manifest_entries.sort(key=lambda x: x["csv_line_num"])

    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f_out:
        for entry in manifest_entries:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Wrote %d entries to manifest: %s", len(manifest_entries), manifest_path)

    report: dict[str, Any] = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_source": source_name,
        "zenodo_doi": ZENODO_DOI,
        "zenodo_concept_doi": ZENODO_CONCEPT_DOI,
        "total_csv_records": total_csv_records,
        "total_screenshots_scanned": total_screenshots,
        "valid_pairs_mapped": len(manifest_entries),
        "missing_screenshots_count": missing_screenshots_count,
        "phishing_pairs_count": phishing_pairs_count,
        "unverified_candidate_pairs_count": unverified_candidate_count,
        "quarantined_conflict_count": quarantined_conflict_count,
        "duplicate_url_count": duplicate_url_count,
        "duplicate_screenshot_count": duplicate_screenshot_count,
        "corrupted_image_count": corrupted_image_count,
        "invalid_filename_count": invalid_filename_count,
        "out_of_bounds_line_count": out_of_bounds_line_count,
        "label_mismatch_count": label_mismatch_count,
        "unique_registered_domains_count": unique_domains,
        "manifest_path": str(manifest_path).replace("\\", "/"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f_out:
        json.dump(report, f_out, indent=2)
    logger.info("Wrote quality report to: %s", report_path)

    return len(manifest_entries), report


# ── Selective Acquisition Runner ─────────────────────────────────────

def run_selective_acquisition(
    zip_source: str | os.PathLike,  # URL or local ZIP path
    extract_dir: str | os.PathLike = DEFAULT_EXTRACT_DIR,
    manifest_path: str | os.PathLike = DEFAULT_MANIFEST_PATH,
    report_path: str | os.PathLike = DEFAULT_REPORT_PATH,
    max_samples: int = 5000,
    max_per_domain: int = 1,
    balance_classes: bool = True,
    dry_run: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Execute selective acquisition of balanced URL-screenshot pairs.

    Parameters
    ----------
    zip_source : str | PathLike
        Either a remote URL (Zenodo) or a local ZIP archive file path.
    extract_dir : str | PathLike
        Destination directory to store screenshots.
    manifest_path : str | PathLike
        Output path for manifest.jsonl.
    report_path : str | PathLike
        Output path for quality_report.json.
    max_samples : int
        Maximum screenshot pairs to acquire (default: 5,000).
    max_per_domain : int
        Maximum samples permitted per registered domain (default: 1).
    balance_classes : bool
        If True, ensures approximately equal numbers of phishing and legitimate samples.
    dry_run : bool
        If True, plans and reports selection without downloading image files.

    Returns
    -------
    tuple[int, dict[str, Any]]
        Number of selected samples and the summary statistics / quality report.
    """
    is_remote = str(zip_source).startswith(("http://", "https://"))
    logger.info("Opening ZIP source (%s): %s", "remote HTTP" if is_remote else "local file", zip_source)

    if is_remote:
        rz = RemoteZipReader(str(zip_source))
        zf = zipfile.ZipFile(rz)
    else:
        zf = zipfile.ZipFile(str(zip_source), "r")

    try:
        # 1. Locate and parse merged_features_final.csv
        csv_candidates = [
            "Phishing dataset/url/merged_features_final.csv",
            "url/merged_features_final.csv",
            "merged_features_final.csv",
        ]
        csv_zinfo = None
        for name in zf.namelist():
            if any(name.endswith(c) for c in csv_candidates):
                csv_zinfo = zf.getinfo(name)
                break

        if not csv_zinfo:
            raise FileNotFoundError("Could not find merged_features_final.csv in ZIP archive central directory.")

        logger.info("Reading CSV metadata: %s...", csv_zinfo.filename)
        if is_remote:
            csv_bytes = fetch_entry_bytes_range(str(zip_source), csv_zinfo)
        else:
            with zf.open(csv_zinfo) as f:
                csv_bytes = f.read()

        csv_records = parse_csv_records(csv_bytes)
        logger.info("Parsed %d CSV records.", len(csv_records))

        # 2. Collect image ZipInfo entries
        image_zinfos = [
            info for info in zf.infolist()
            if not info.is_dir() and info.filename.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        logger.info("Found %d screenshot entries in ZIP central directory.", len(image_zinfos))

        # 3. Perform balanced domain-separated sampling
        selected_samples, selection_stats = select_balanced_samples(
            csv_records=csv_records,
            image_entries=image_zinfos,
            max_samples=max_samples,
            max_per_domain=max_per_domain,
            balance_classes=balance_classes,
        )

        # 4. Handle Dry-Run Mode
        if dry_run:
            print("\n" + "=" * 65)
            print(" PHISHGUARD SELECTIVE ACQUISITION – DRY-RUN AUDIT SUMMARY")
            print("=" * 65)
            print(f" Target DOI                   : {ZENODO_DOI}")
            print(f" Source Archive               : {ZENODO_FILE_NAME} (Total: 4.48 GB)")
            print(f" Requested Maximum Samples    : {max_samples}")
            print(f" Max per Registered Domain    : {max_per_domain}")
            print(f" Balance Classes              : {balance_classes}")
            print(f" Selected Total Pairs         : {selection_stats['selected_total']}")
            print(f"   ├─ Phishing Samples        : {selection_stats['selected_phishing']}")
            print(f"   └─ Unverified Legitimate   : {selection_stats['selected_legitimate_unverified']}")
            print(f" Unique Registered Domains    : {selection_stats['unique_registered_domains']}")
            print(f" Quarantined Conflict Domains : {selection_stats['quarantined_conflicts_count']}")
            print(f" Estimated Compressed Download: {selection_stats['estimated_compressed_download_mb']} MB (~{selection_stats['estimated_compressed_download_mb']/1024:.2f} GB)")
            print(f" Estimated Extracted Size     : {selection_stats['estimated_uncompressed_mb']} MB (~{selection_stats['estimated_uncompressed_mb']/1024:.2f} GB)")
            print("=" * 65)
            print("[DRY-RUN COMPLETE] No files were downloaded to disk.\n")
            return len(selected_samples), selection_stats

        # 5. Active Selective Download & Extraction
        total_samples = len(selected_samples)
        logger.info("Acquiring %d selected screenshots into %s...", total_samples, extract_dir)
        os.makedirs(extract_dir, exist_ok=True)

        manifest_entries: list[dict[str, Any]] = []
        failed_entries: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        seen_hashes: dict[str, str] = {}
        duplicate_url_count = 0
        duplicate_hash_count = 0
        corrupted_count = 0
        skipped_count = 0

        for idx, sample in enumerate(selected_samples, start=1):
            zinfo = sample["image_entry"]
            dest_file = os.path.abspath(os.path.join(extract_dir, zinfo.filename))
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)

            # Idempotency / resumption check: size + SHA-256 + OpenCV image validity
            file_already_valid = False
            img_info: dict[str, Any] = {"is_valid": False, "width": 0, "height": 0, "error": None}
            if os.path.isfile(dest_file):
                if os.path.getsize(dest_file) == zinfo.file_size:
                    try:
                        sha_test = compute_file_sha256(dest_file)
                        sha_ok = len(sha_test) == 64
                    except Exception:
                        sha_ok = False

                    img_check = inspect_image(dest_file)
                    if sha_ok and img_check["is_valid"]:
                        file_already_valid = True
                        img_info = img_check
                        skipped_count += 1
                        logger.info(
                            "[%d/%d] Skipping verified existing screenshot: %s",
                            idx,
                            total_samples,
                            zinfo.filename,
                        )
                    else:
                        logger.warning(
                            "[%d/%d] Existing file %s failed integrity check (sha_ok=%s, img_error=%s). Re-downloading...",
                            idx,
                            total_samples,
                            dest_file,
                            sha_ok,
                            img_check.get("error"),
                        )
                        try:
                            os.remove(dest_file)
                        except OSError:
                            pass
                else:
                    logger.warning(
                        "[%d/%d] Existing file %s size mismatch (%d != %d). Re-downloading...",
                        idx,
                        total_samples,
                        dest_file,
                        os.path.getsize(dest_file),
                        zinfo.file_size,
                    )
                    try:
                        os.remove(dest_file)
                    except OSError:
                        pass

            if not file_already_valid:
                logger.info(
                    "[%d/%d] Acquiring screenshot: %s (compress_size=%d bytes)",
                    idx,
                    total_samples,
                    zinfo.filename,
                    zinfo.compress_size,
                )
                tmp_dest = dest_file + f".tmp.{os.getpid()}"
                try:
                    if is_remote:
                        entry_data = fetch_entry_bytes_range(str(zip_source), zinfo, max_retries=5)
                    else:
                        with zf.open(zinfo) as f:
                            entry_data = f.read()

                    with open(tmp_dest, "wb") as f_out:
                        f_out.write(entry_data)
                    os.replace(tmp_dest, dest_file)
                except Exception as exc:
                    logger.error(
                        "[%d/%d] Failed to acquire screenshot %s after all retries: %s",
                        idx,
                        total_samples,
                        zinfo.filename,
                        exc,
                    )
                    failed_entries.append({
                        "csv_line_num": sample["csv_line_num"],
                        "filename": zinfo.filename,
                        "original_url": sample["original_url"],
                        "error": str(exc),
                    })
                    if os.path.exists(tmp_dest):
                        try:
                            os.remove(tmp_dest)
                        except OSError:
                            pass
                    continue

                img_info = inspect_image(dest_file)

            # Compute SHA-256
            img_sha = compute_file_sha256(dest_file)
            if img_sha in seen_hashes:
                duplicate_hash_count += 1
            else:
                seen_hashes[img_sha] = dest_file

            if sample["original_url"] in seen_urls:
                duplicate_url_count += 1
            seen_urls.add(sample["original_url"])

            # Image inspection
            if not img_info["is_valid"]:
                corrupted_count += 1

            # Provenance and quarantine tagging
            reg_dom = sample["registered_domain"]
            if sample["effective_label"] == "phishing":
                threat_cat = THREAT_CATEGORY_PHISHING
                label_prov = "phishtank_verified"
                quar_stat = "none"
            else:
                threat_cat = THREAT_CATEGORY_UNVERIFIED
                label_prov = "tranco_candidate_unverified"
                quar_stat = "unverified_candidate"

            if reg_dom.lower() in KNOWN_CONFLICTING_DOMAINS:
                quar_stat = "quarantined_conflict"

            try:
                rel_path = os.path.relpath(dest_file, PROJECT_ROOT).replace("\\", "/")
            except ValueError:
                rel_path = str(dest_file).replace("\\", "/")

            manifest_entry = {
                "csv_line_num": sample["csv_line_num"],
                "original_url": sample["original_url"],
                "normalized_url": sample["normalized_url"],
                "registered_domain": reg_dom,
                "original_label": sample["original_label"],
                "threat_category": threat_cat,
                "screenshot_path": rel_path,
                "screenshot_sha256": img_sha,
                "image_width": img_info["width"],
                "image_height": img_info["height"],
                "is_valid": img_info["is_valid"],
                "dataset_source": "zenodo_23477065_selective",
                "label_provenance": label_prov,
                "quarantine_status": quar_stat,
            }
            manifest_entries.append(manifest_entry)

        # Logging summary of acquisition results
        if failed_entries:
            logger.warning(
                "Selective acquisition completed with %d failed entries out of %d requested:",
                len(failed_entries),
                total_samples,
            )
            for fe in failed_entries[:10]:
                logger.warning("  - %s (line %d): %s", fe["filename"], fe["csv_line_num"], fe["error"])
            if len(failed_entries) > 10:
                logger.warning("  ... and %d more failed entries (see report).", len(failed_entries) - 10)
        else:
            logger.info(
                "Selective acquisition completed successfully: %d/%d screenshots acquired (%d verified and skipped).",
                len(manifest_entries),
                total_samples,
                skipped_count,
            )

        # Write manifest.jsonl
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        manifest_entries.sort(key=lambda x: x["csv_line_num"])
        with open(manifest_path, "w", encoding="utf-8") as f_out:
            for entry in manifest_entries:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Wrote selective manifest (%d entries): %s", len(manifest_entries), manifest_path)

        # Write quality report
        report = {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "acquisition_mode": "selective_range_acquisition",
            "zenodo_doi": ZENODO_DOI,
            "max_samples_requested": max_samples,
            "total_selected_candidates": total_samples,
            "total_pairs_acquired": len(manifest_entries),
            "skipped_already_valid_count": skipped_count,
            "failed_download_count": len(failed_entries),
            "failed_entries": failed_entries,
            "phishing_pairs_count": sum(1 for m in manifest_entries if m["threat_category"] == THREAT_CATEGORY_PHISHING),
            "unverified_candidate_pairs_count": sum(1 for m in manifest_entries if m["threat_category"] == THREAT_CATEGORY_UNVERIFIED),
            "unique_registered_domains": len({m["registered_domain"] for m in manifest_entries}),
            "quarantined_conflict_count": sum(1 for m in manifest_entries if m["quarantine_status"] == "quarantined_conflict"),
            "duplicate_url_count": duplicate_url_count,
            "duplicate_screenshot_count": duplicate_hash_count,
            "corrupted_image_count": corrupted_count,
            "total_downloaded_mb": selection_stats["estimated_compressed_download_mb"],
            "manifest_path": str(manifest_path).replace("\\", "/"),
        }

        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f_out:
            json.dump(report, f_out, indent=2)
        logger.info("Wrote selective quality report: %s", report_path)

        return len(manifest_entries), report

    finally:
        zf.close()


# ── Command Line Interface ──────────────────────────────────────────

def build_cli() -> argparse.ArgumentParser:
    """Build the command line parser for the visual acquisition pipeline."""
    parser = argparse.ArgumentParser(
        description="PhishGuard – Zenodo V2.0 Dataset Acquisition and Provenance Pipeline"
    )
    # Acquisition modes
    parser.add_argument(
        "--selective",
        action="store_true",
        help="Selectively download/extract only a subset of URL-screenshot pairs (default max: 5,000).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum number of balanced URL-screenshot pairs to acquire in selective mode (default: 5000).",
    )
    parser.add_argument(
        "--max-per-domain",
        type=int,
        default=1,
        help="Maximum samples permitted per apex registered domain (default: 1).",
    )
    parser.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ensure equal numbers of phishing and legitimate samples (default: True).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate selective acquisition: display candidate counts, domain stats, and download estimate without downloading.",
    )
    parser.add_argument(
        "--allow-domain-repeats",
        action="store_true",
        help="Allow multiple samples per registered domain if unique domains are insufficient.",
    )
    # Full archive modes
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download complete Phishing dataset V2.0.zip (4.48 GB) from Zenodo with resumption.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the downloaded zip archive into datasets/screenshots/raw/.",
    )
    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="Reconstruct mappings, validate screenshots against CSV, and build manifest.jsonl and quality_report.json.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run download, extract, and build-manifest for complete archive.",
    )
    # Path configuration
    parser.add_argument(
        "--zip-path",
        default=None,
        help="Path to local ZIP file (if omitted in selective mode, uses remote Zenodo byte ranges).",
    )
    parser.add_argument(
        "--extract-dir",
        default=DEFAULT_EXTRACT_DIR,
        help=f"Directory to extract raw images and CSVs (default: {DEFAULT_EXTRACT_DIR}).",
    )
    parser.add_argument(
        "--manifest-path",
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to manifest.jsonl output (default: {DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help=f"Path to quality_report.json output (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Explicit path to merged_features_final.csv.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Explicit path to directory containing screenshot images.",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_cli()
    args = parser.parse_args()

    # Safety constraint: require explicit action
    if not (args.selective or args.dry_run or args.download or args.extract or args.build_manifest or args.all):
        parser.print_help()
        print("\n[SAFETY] No action specified. Use --dry-run or --selective to perform selective acquisition.")
        return 0

    extract_dir = os.path.abspath(args.extract_dir)

    # 1. Selective Acquisition / Dry-Run
    if args.selective or args.dry_run:
        zip_source = args.zip_path if args.zip_path else ZENODO_DOWNLOAD_URL
        max_per_domain = 999999 if args.allow_domain_repeats else args.max_per_domain

        run_selective_acquisition(
            zip_source=zip_source,
            extract_dir=extract_dir,
            manifest_path=args.manifest_path,
            report_path=args.report_path,
            max_samples=args.max_samples,
            max_per_domain=max_per_domain,
            balance_classes=args.balance_classes,
            dry_run=args.dry_run,
        )
        return 0

    # 2. Full Archive Stages
    zip_path = os.path.abspath(args.zip_path or os.path.join(DEFAULT_DOWNLOAD_DIR, ZENODO_FILE_NAME))

    if args.download or args.all:
        logger.info("Executing Full Archive Download Stage: %s", ZENODO_DOWNLOAD_URL)
        download_file_with_resume(
            url=ZENODO_DOWNLOAD_URL,
            dest_path=zip_path,
            expected_size=ZENODO_FILE_SIZE,
            expected_md5=ZENODO_EXPECTED_MD5,
        )

    if args.extract or args.all:
        if not os.path.isfile(zip_path):
            logger.error("ZIP archive not found at %s. Run with --download first.", zip_path)
            return 1
        logger.info("Executing Full Archive Extract Stage: %s -> %s", zip_path, extract_dir)
        safe_extract_archive(zip_path, extract_dir)

    if args.build_manifest or args.all:
        logger.info("Executing Build Manifest Stage...")
        csv_path = args.csv_path
        if not csv_path:
            candidates = [
                os.path.join(extract_dir, "Phishing dataset", "url", "merged_features_final.csv"),
                os.path.join(extract_dir, "url", "merged_features_final.csv"),
                os.path.join(extract_dir, "merged_features_final.csv"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    csv_path = c
                    break

        if not csv_path or not os.path.isfile(csv_path):
            logger.error("Could not find merged_features_final.csv in %s.", extract_dir)
            return 1

        images_dir = args.images_dir
        if not images_dir:
            cand_img = [
                os.path.join(extract_dir, "Phishing dataset", "image"),
                os.path.join(extract_dir, "image"),
                extract_dir,
            ]
            for ci in cand_img:
                if os.path.isdir(ci):
                    images_dir = ci
                    break

        if not images_dir or not os.path.isdir(images_dir):
            logger.error("Could not find image directory in %s.", extract_dir)
            return 1

        reconstruct_and_validate_dataset(
            csv_path=csv_path,
            images_dir=images_dir,
            manifest_path=args.manifest_path,
            report_path=args.report_path,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
