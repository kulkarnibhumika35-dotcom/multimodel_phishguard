"""
Unit Tests for Zenodo V2.0 Dataset Acquisition and Provenance Pipeline
(datasets/scripts/acquire_visual_dataset.py)
"""

import hashlib
import http.client
import io
import json
import os
import socket
import urllib.error
import zipfile
import zlib
import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch

from datasets.scripts.acquire_visual_dataset import (
    RangeDownloadError,
    SCREENSHOT_REGEX,
    compute_file_md5,
    compute_file_sha256,
    fetch_entry_bytes_range,
    inspect_image,
    parse_csv_records,
    reconstruct_and_validate_dataset,
    run_selective_acquisition,
    safe_extract_archive,
    select_balanced_samples,
)
from ml.training.dataset_pipeline import (
    KNOWN_CONFLICTING_DOMAINS,
    THREAT_CATEGORY_PHISHING,
    THREAT_CATEGORY_UNVERIFIED,
)


@pytest.fixture
def synthetic_image_png(tmp_path):
    """Create a valid synthetic 100x100 BGR PNG image."""
    img_path = tmp_path / "test_sample.png"
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[20:80, 20:80] = [0, 255, 0]  # Green square
    cv2.imwrite(str(img_path), arr)
    return img_path


@pytest.fixture
def synthetic_corrupted_image(tmp_path):
    """Create an invalid/corrupted image file with random garbage bytes."""
    corrupt_path = tmp_path / "corrupted.png"
    corrupt_path.write_bytes(b"NOT_A_VALID_IMAGE_BYTES_XYZ123")
    return corrupt_path


def test_compute_file_sha256_and_md5(tmp_path):
    """Verify cryptographic hash calculations match hashlib ground truth."""
    test_file = tmp_path / "sample.bin"
    test_bytes = b"PhishGuard Multimodal Acquisition Pipeline Test 2026"
    test_file.write_bytes(test_bytes)

    expected_sha256 = hashlib.sha256(test_bytes).hexdigest()
    expected_md5 = hashlib.md5(test_bytes).hexdigest()

    assert compute_file_sha256(test_file) == expected_sha256
    assert compute_file_md5(test_file) == expected_md5


def test_inspect_image_valid_and_invalid(synthetic_image_png, synthetic_corrupted_image, tmp_path):
    """Verify inspect_image correctly identifies valid and corrupted images."""
    res_valid = inspect_image(synthetic_image_png)
    assert res_valid["is_valid"] is True
    assert res_valid["width"] == 100
    assert res_valid["height"] == 100
    assert res_valid["error"] is None

    res_corrupt = inspect_image(synthetic_corrupted_image)
    assert res_corrupt["is_valid"] is False
    assert res_corrupt["width"] == 0
    assert res_corrupt["height"] == 0
    assert res_corrupt["error"] == "opencv_decode_failed"

    res_missing = inspect_image(tmp_path / "non_existent.png")
    assert res_missing["is_valid"] is False
    assert res_missing["error"] == "file_not_found"

    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")
    res_empty = inspect_image(empty_file)
    assert res_empty["is_valid"] is False
    assert res_empty["error"] == "empty_file"


def test_screenshot_filename_regex():
    """Verify regular expression parses valid screenshot filename patterns and rejects invalid ones."""
    m1 = SCREENSHOT_REGEX.match("2_manua.ls.png")
    assert m1 is not None
    assert int(m1.group(1)) == 2
    assert m1.group(2) == "manua.ls"

    m2 = SCREENSHOT_REGEX.match("10001_bluesystem.me.PNG")
    assert m2 is not None
    assert int(m2.group(1)) == 10001
    assert m2.group(2) == "bluesystem.me"

    assert SCREENSHOT_REGEX.match("invalid_name.png") is None
    assert SCREENSHOT_REGEX.match("no_extension") is None


def test_safe_extract_archive(tmp_path):
    """Verify safe extraction unpacks files and prevents zip-slip path traversal."""
    zip_path = tmp_path / "test_archive.zip"
    extract_dir = tmp_path / "extracted"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("subfolder/file1.txt", b"Content 1")
        zf.writestr("file2.txt", b"Content 2")

    extracted = safe_extract_archive(zip_path, extract_dir)
    assert len(extracted) == 2
    assert os.path.isfile(extract_dir / "subfolder" / "file1.txt")
    assert os.path.isfile(extract_dir / "file2.txt")

    bad_zip_path = tmp_path / "bad_slip.zip"
    with zipfile.ZipFile(bad_zip_path, "w") as zf:
        zf.writestr("../evil.txt", b"Malicious slip content")

    with pytest.raises(Exception):
        safe_extract_archive(bad_zip_path, extract_dir)


def test_csv_to_image_row_mapping_and_validation(tmp_path):
    """Verify end-to-end CSV-to-screenshot row mapping, quarantine, and quality audit."""
    csv_content = (
        "url;entropy;url_length;label\n"
        "https://example.org;3.2;19;legitimate\n"
        "https://phish-target.com/login;4.1;30;phishing\n"
        "https://kakaku.com/deals;3.8;24;legitimate\n"
        "https://phish-target.com/login;4.1;30;phishing\n"
        "https://another-benign.com;3.0;26;legitimate\n"
    )
    csv_file = tmp_path / "merged_features_final.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    images_root = tmp_path / "images"
    legit_dir = images_root / "legitimate"
    phish_dir = images_root / "phishing"
    legit_dir.mkdir(parents=True)
    phish_dir.mkdir(parents=True)

    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[:, :] = [100, 150, 200]
    cv2.imwrite(str(legit_dir / "2_example.org.png"), arr)

    arr2 = np.zeros((90, 90, 3), dtype=np.uint8)
    arr2[:, :] = [50, 50, 220]
    cv2.imwrite(str(phish_dir / "3_phish-target.com_login.png"), arr2)

    cv2.imwrite(str(legit_dir / "4_kakaku.com_deals.png"), arr)

    arr3 = np.zeros((100, 100, 3), dtype=np.uint8)
    arr3[:, :] = [10, 20, 30]
    cv2.imwrite(str(phish_dir / "5_phish-target.com_login.png"), arr3)

    (phish_dir / "invalid_unmatched_file.png").write_bytes(b"FAKE_PNG_BYTES")

    manifest_path = tmp_path / "manifest.jsonl"
    report_path = tmp_path / "quality_report.json"

    num_mapped, report = reconstruct_and_validate_dataset(
        csv_path=csv_file,
        images_dir=images_root,
        manifest_path=manifest_path,
        report_path=report_path,
    )

    assert num_mapped == 4
    assert os.path.isfile(manifest_path)
    assert os.path.isfile(report_path)

    assert report["total_csv_records"] == 5
    assert report["valid_pairs_mapped"] == 4
    assert report["missing_screenshots_count"] == 1
    assert report["invalid_filename_count"] == 1
    assert report["duplicate_url_count"] == 1
    assert report["duplicate_screenshot_count"] == 1
    assert report["quarantined_conflict_count"] == 1
    assert report["unverified_candidate_pairs_count"] == 2

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) == 4
    assert records[0]["csv_line_num"] == 2
    assert records[0]["threat_category"] == THREAT_CATEGORY_UNVERIFIED
    assert records[0]["label_provenance"] == "tranco_candidate_unverified"
    assert records[0]["quarantine_status"] == "unverified_candidate"

    assert records[1]["csv_line_num"] == 3
    assert records[1]["threat_category"] == THREAT_CATEGORY_PHISHING
    assert records[1]["label_provenance"] == "phishtank_verified"
    assert records[1]["quarantine_status"] == "none"

    assert records[2]["csv_line_num"] == 4
    assert records[2]["registered_domain"] == "kakaku.com"
    assert records[2]["quarantine_status"] == "quarantined_conflict"


def test_select_balanced_samples_limit_and_class_balancing():
    """Verify select_balanced_samples caps at max_samples with balanced distribution."""
    # 10 CSV records: 5 phishing, 5 legitimate
    csv_records = [
        {"line_num": 2, "url": "https://p1.org/a", "label": "phishing"},
        {"line_num": 3, "url": "https://l1.org/a", "label": "legitimate"},
        {"line_num": 4, "url": "https://p2.com/b", "label": "phishing"},
        {"line_num": 5, "url": "https://l2.com/b", "label": "legitimate"},
        {"line_num": 6, "url": "https://p3.net/c", "label": "phishing"},
        {"line_num": 7, "url": "https://l3.net/c", "label": "legitimate"},
        {"line_num": 8, "url": "https://p4.io/d", "label": "phishing"},
        {"line_num": 9, "url": "https://l4.io/d", "label": "legitimate"},
        {"line_num": 10, "url": "https://p5.xyz/e", "label": "phishing"},
        {"line_num": 11, "url": "https://l5.xyz/e", "label": "legitimate"},
    ]

    # Mock ZipInfo items
    class DummyZipInfo:
        def __init__(self, filename, comp=1000, orig=2000):
            self.filename = filename
            self.compress_size = comp
            self.file_size = orig

    image_entries = [
        DummyZipInfo("Phishing dataset/image/phishing/2_p1.org_a.png"),
        DummyZipInfo("Phishing dataset/image/legitimate/3_l1.org_a.png"),
        DummyZipInfo("Phishing dataset/image/phishing/4_p2.com_b.png"),
        DummyZipInfo("Phishing dataset/image/legitimate/5_l2.com_b.png"),
        DummyZipInfo("Phishing dataset/image/phishing/6_p3.net_c.png"),
        DummyZipInfo("Phishing dataset/image/legitimate/7_l3.net_c.png"),
        DummyZipInfo("Phishing dataset/image/phishing/8_p4.io_d.png"),
        DummyZipInfo("Phishing dataset/image/legitimate/9_l4.io_d.png"),
        DummyZipInfo("Phishing dataset/image/phishing/10_p5.xyz_e.png"),
        DummyZipInfo("Phishing dataset/image/legitimate/11_l5.xyz_e.png"),
    ]

    # Request max_samples = 6 (should get exactly 3 phishing and 3 legitimate)
    selected, stats = select_balanced_samples(
        csv_records=csv_records,
        image_entries=image_entries,
        max_samples=6,
        enforce_unique_domains=True,
    )

    assert len(selected) == 6
    assert stats["selected_total"] == 6
    assert stats["selected_phishing"] == 3
    assert stats["selected_legitimate_unverified"] == 3
    assert stats["unique_registered_domains"] == 6
    assert stats["estimated_compressed_download_bytes"] == 6 * 1000
    assert stats["estimated_uncompressed_bytes"] == 6 * 2000


def test_select_balanced_samples_domain_deduplication():
    """Verify duplicate registered domains are excluded within each class."""
    csv_records = [
        {"line_num": 2, "url": "https://docs.google.com/doc1", "label": "phishing"},
        {"line_num": 3, "url": "https://sites.google.com/site2", "label": "phishing"},
        {"line_num": 4, "url": "https://drive.google.com/drive3", "label": "phishing"},
        {"line_num": 5, "url": "https://paypal-security.com/login", "label": "phishing"},
        {"line_num": 6, "url": "https://bank-verify.net/auth", "label": "phishing"},
        {"line_num": 7, "url": "https://tranco1.com", "label": "legitimate"},
        {"line_num": 8, "url": "https://sub.tranco1.com", "label": "legitimate"},
        {"line_num": 9, "url": "https://tranco2.org", "label": "legitimate"},
    ]

    class DummyZipInfo:
        def __init__(self, filename):
            self.filename = filename
            self.compress_size = 500
            self.file_size = 1000

    image_entries = [
        DummyZipInfo("Phishing dataset/image/phishing/2_docs.google.com_doc1.png"),
        DummyZipInfo("Phishing dataset/image/phishing/3_sites.google.com_site2.png"),
        DummyZipInfo("Phishing dataset/image/phishing/4_drive.google.com_drive3.png"),
        DummyZipInfo("Phishing dataset/image/phishing/5_paypal-security.com_login.png"),
        DummyZipInfo("Phishing dataset/image/phishing/6_bank-verify.net_auth.png"),
        DummyZipInfo("Phishing dataset/image/legitimate/7_tranco1.com.png"),
        DummyZipInfo("Phishing dataset/image/legitimate/8_sub.tranco1.com.png"),
        DummyZipInfo("Phishing dataset/image/legitimate/9_tranco2.org.png"),
    ]

    selected, stats = select_balanced_samples(
        csv_records=csv_records,
        image_entries=image_entries,
        max_samples=10,
        enforce_unique_domains=True,
    )

    phishing_domains = [s["registered_domain"] for s in selected if s["effective_label"] == "phishing"]
    legit_domains = [s["registered_domain"] for s in selected if s["effective_label"] == "legitimate"]

    # google.com should appear ONLY ONCE despite 3 candidates
    assert phishing_domains.count("google.com") == 1
    # tranco1.com should appear ONLY ONCE despite 2 candidates
    assert legit_domains.count("tranco1.com") == 1


def test_fetch_entry_bytes_range_success_and_crc(tmp_path):
    """Verify fetch_entry_bytes_range correctly parses local header and decompresses data."""
    # 1. Create a zip archive with one compressed file
    zip_path = tmp_path / "range_test.zip"
    payload = b"A" * 1024 + b"PhishGuard Range Download Content 2026"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("test_folder/sample.txt", payload)

    # 2. Inspect entry in local zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        zinfo = zf.getinfo("test_folder/sample.txt")

    # Read binary bytes of the zip file
    zip_bytes = zip_path.read_bytes()
    fetch_len = 30 + len(zinfo.filename) + 256 + zinfo.compress_size
    extended_bytes = zip_bytes + b"\x00" * fetch_len

    # 3. Mock urllib.request.urlopen returning the byte slice for the entry
    def fake_urlopen(req, timeout=30):
        # Extract Range header
        range_header = req.headers.get("Range")
        assert range_header.startswith("bytes=")
        parts = range_header[6:].split("-")
        start = int(parts[0])
        end = int(parts[1]) if parts[1] else len(extended_bytes) - 1
        sliced = extended_bytes[start : end + 1]

        mock_resp = MagicMock()
        mock_resp.status = 206
        mock_resp.headers = {"Content-Range": f"bytes {start}-{end}/{len(extended_bytes)}"}
        mock_resp.read.return_value = sliced
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        extracted = fetch_entry_bytes_range("https://fake.zenodo.org/file.zip", zinfo)

    assert extracted == payload
    assert len(extracted) == zinfo.file_size
    assert zlib.crc32(extracted) == zinfo.CRC


def test_fetch_entry_bytes_range_error_handling(tmp_path):
    """Verify RangeDownloadError is raised on HTTP errors or corrupted local headers."""
    dummy_zinfo = zipfile.ZipInfo("test.png")
    dummy_zinfo.header_offset = 100
    dummy_zinfo.compress_size = 500
    dummy_zinfo.file_size = 600
    dummy_zinfo.compress_type = zipfile.ZIP_DEFLATED
    dummy_zinfo.CRC = 12345

    fetch_len = 30 + len(dummy_zinfo.filename) + 256 + dummy_zinfo.compress_size
    end_byte = dummy_zinfo.header_offset + fetch_len - 1

    # 1. Test HTTP 416 error
    def mock_http_416(req, timeout=30):
        raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable", {}, None)

    with patch("urllib.request.urlopen", side_effect=mock_http_416), patch("time.sleep"):
        with pytest.raises(RangeDownloadError, match="416"):
            fetch_entry_bytes_range("https://fake.zenodo.org/file.zip", dummy_zinfo, max_retries=1)

    # 2. Test corrupted ZIP magic signature
    def mock_corrupt_header(req, timeout=30):
        mock_resp = MagicMock()
        mock_resp.status = 206
        mock_resp.headers = {"Content-Range": f"bytes {dummy_zinfo.header_offset}-{end_byte}/100000"}
        mock_resp.read.return_value = b"CORRUPTED_MAGIC_NOT_PK" + b"\x00" * (fetch_len - 22)
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_corrupt_header), patch("time.sleep"):
        with pytest.raises(RangeDownloadError, match="Corrupt ZIP local header"):
            fetch_entry_bytes_range("https://fake.zenodo.org/file.zip", dummy_zinfo, max_retries=1)


def test_run_selective_acquisition_dry_run_and_active(tmp_path):
    """Verify run_selective_acquisition performs dry run and active download against a local ZIP archive."""
    # 1. Build a synthetic ZIP archive mimicking the Zenodo structure
    zip_path = tmp_path / "synthetic_zenodo_v2.zip"
    csv_text = (
        "url;entropy;url_length;label\n"
        "https://example-target1.com/login;3.5;32;phishing\n"
        "https://legit-site1.org;3.0;20;legitimate\n"
        "https://kakaku.com/deals;3.2;22;legitimate\n"
        "https://example-target2.com/secure;3.6;33;phishing\n"
    )

    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :] = [255, 128, 0]
    _, png_bytes = cv2.imencode(".png", arr)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Phishing dataset/url/merged_features_final.csv", csv_text)
        zf.writestr("Phishing dataset/image/phishing/2_example-target1.com_login.png", png_bytes.tobytes())
        zf.writestr("Phishing dataset/image/legitimate/3_legit-site1.org.png", png_bytes.tobytes())
        zf.writestr("Phishing dataset/image/legitimate/4_kakaku.com_deals.png", png_bytes.tobytes())
        zf.writestr("Phishing dataset/image/phishing/5_example-target2.com_secure.png", png_bytes.tobytes())

    extract_dir = tmp_path / "extracted_screenshots"
    manifest_path = tmp_path / "selective_manifest.jsonl"
    report_path = tmp_path / "selective_report.json"

    # 2. Test Dry-Run Mode
    count_dry, stats_dry = run_selective_acquisition(
        zip_source=zip_path,
        extract_dir=extract_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        max_samples=4,
        dry_run=True,
    )

    assert count_dry == 4
    assert stats_dry["selected_phishing"] == 2
    assert stats_dry["selected_legitimate_unverified"] == 2
    assert stats_dry["quarantined_conflicts_count"] == 1  # kakaku.com
    # Verify dry-run did NOT create files on disk
    assert not os.path.exists(extract_dir)
    assert not os.path.exists(manifest_path)
    assert not os.path.exists(report_path)

    # 3. Test Active Selective Acquisition Mode
    count_active, report_active = run_selective_acquisition(
        zip_source=zip_path,
        extract_dir=extract_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        max_samples=4,
        dry_run=False,
    )

    assert count_active == 4
    assert os.path.isfile(manifest_path)
    assert os.path.isfile(report_path)

    # Verify extracted images on disk
    extracted_imgs = [
        f for root, _, files in os.walk(extract_dir) for f in files if f.endswith(".png")
    ]
    assert len(extracted_imgs) == 4

    # Verify manifest records
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f]

    assert len(records) == 4
    # Check Tranco legitimate labeled unverified_candidate
    legit_recs = [r for r in records if r["original_label"] == "legitimate"]
    for lr in legit_recs:
        assert lr["threat_category"] == THREAT_CATEGORY_UNVERIFIED
        assert lr["label_provenance"] == "tranco_candidate_unverified"

    # Check kakaku.com quarantined
    kakaku_rec = next(r for r in records if r["registered_domain"] == "kakaku.com")
    assert kakaku_rec["quarantine_status"] == "quarantined_conflict"

    # Check image width/height correctly recorded from OpenCV
    assert all(r["image_width"] == 64 for r in records)
    assert all(r["image_height"] == 64 for r in records)
    assert all(r["is_valid"] is True for r in records)


def test_url_normalization_preserves_hostname_without_apex_www():
    """Verify that canonical_normalize_url does NOT alter apex hostnames when ensure_apex_www=False."""
    from ml.training.dataset_pipeline import canonical_normalize_url

    assert canonical_normalize_url("https://github.com/", ensure_apex_www=False) == "https://github.com"
    assert canonical_normalize_url("http://tbcpl.lol/", ensure_apex_www=False) == "http://tbcpl.lol"
    assert canonical_normalize_url("https://bluesystem.me", ensure_apex_www=False) == "https://bluesystem.me"
    assert canonical_normalize_url("https://docs.google.com/test", ensure_apex_www=False) == "https://docs.google.com/test"
    assert canonical_normalize_url("https://www.paypal.com/signin", ensure_apex_www=False) == "https://www.paypal.com/signin"


def test_fetch_entry_bytes_range_incomplete_read_retry(tmp_path):
    """Verify fetch_entry_bytes_range catches http.client.IncompleteRead and retries with backoff."""
    zip_path = tmp_path / "incomplete_test.zip"
    payload = b"Reliable IncompleteRead recovery test content " * 20
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("img/sample.png", payload)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zinfo = zf.getinfo("img/sample.png")

    zip_bytes = zip_path.read_bytes()
    fetch_len = 30 + len(zinfo.filename) + 256 + zinfo.compress_size
    extended_bytes = zip_bytes + b"\x00" * fetch_len
    valid_slice = extended_bytes[zinfo.header_offset : zinfo.header_offset + fetch_len]

    attempt_count = 0

    def mock_urlopen_with_incomplete(req, timeout=30):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            # Simulate IncompleteRead on first 2 attempts
            raise http.client.IncompleteRead(b"partial_data", 500)

        mock_resp = MagicMock()
        mock_resp.status = 206
        mock_resp.headers = {
            "Content-Range": f"bytes {zinfo.header_offset}-{zinfo.header_offset + fetch_len - 1}/{len(extended_bytes)}"
        }
        mock_resp.read.return_value = valid_slice
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen_with_incomplete), \
         patch("time.sleep") as mock_sleep:
        res = fetch_entry_bytes_range("https://zenodo.org/fake.zip", zinfo, max_retries=5, backoff_base=0.01)

    assert res == payload
    assert attempt_count == 3
    assert mock_sleep.call_count == 2

    # Verify that if all 5 retries fail with IncompleteRead, RangeDownloadError is raised
    def mock_always_incomplete(req, timeout=30):
        raise http.client.IncompleteRead(b"broken", 100)

    with patch("urllib.request.urlopen", side_effect=mock_always_incomplete), \
         patch("time.sleep"):
        with pytest.raises(RangeDownloadError, match="after 5 attempts"):
            fetch_entry_bytes_range("https://zenodo.org/fake.zip", zinfo, max_retries=5, backoff_base=0.01)


def test_fetch_entry_bytes_range_connection_interruption_retry(tmp_path):
    """Verify fetch_entry_bytes_range retries on connection reset and succeeds."""
    zip_path = tmp_path / "conn_test.zip"
    payload = b"Connection reset recovery payload test content"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("img/conn.png", payload)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zinfo = zf.getinfo("img/conn.png")

    zip_bytes = zip_path.read_bytes()
    fetch_len = 30 + len(zinfo.filename) + 256 + zinfo.compress_size
    extended_bytes = zip_bytes + b"\x00" * fetch_len
    valid_slice = extended_bytes[zinfo.header_offset : zinfo.header_offset + fetch_len]

    attempt_count = 0

    def mock_urlopen_conn_reset(req, timeout=30):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise ConnectionResetError("Connection reset by peer")

        mock_resp = MagicMock()
        mock_resp.status = 206
        mock_resp.headers = {
            "Content-Range": f"bytes {zinfo.header_offset}-{zinfo.header_offset + fetch_len - 1}/{len(extended_bytes)}"
        }
        mock_resp.read.return_value = valid_slice
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen_conn_reset), \
         patch("time.sleep") as mock_sleep:
        res = fetch_entry_bytes_range("https://zenodo.org/fake.zip", zinfo, max_retries=5, backoff_base=0.01)

    assert res == payload
    assert attempt_count == 2
    assert mock_sleep.call_count == 1


def test_fetch_entry_bytes_range_invalid_content_range(tmp_path):
    """Verify validation rejects missing, malformed, or mismatched Content-Range headers."""
    dummy_zinfo = zipfile.ZipInfo("img/test.png")
    dummy_zinfo.header_offset = 1000
    dummy_zinfo.compress_size = 500
    dummy_zinfo.file_size = 800
    dummy_zinfo.compress_type = zipfile.ZIP_DEFLATED
    dummy_zinfo.CRC = 9999

    fetch_len = 30 + len(dummy_zinfo.filename) + 256 + dummy_zinfo.compress_size

    # Case A: Missing Content-Range header
    def mock_missing_cr(req, timeout=30):
        mock_resp = MagicMock()
        mock_resp.status = 206
        mock_resp.headers = {}
        mock_resp.read.return_value = b"\x00" * fetch_len
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_missing_cr), patch("time.sleep"):
        with pytest.raises(RangeDownloadError, match="Missing or invalid Content-Range header"):
            fetch_entry_bytes_range("https://fake.url", dummy_zinfo, max_retries=1)

    # Case B: Malformed Content-Range header
    def mock_malformed_cr(req, timeout=30):
        mock_resp = MagicMock()
        mock_resp.status = 206
        mock_resp.headers = {"Content-Range": "items 0-500/1000"}
        mock_resp.read.return_value = b"\x00" * fetch_len
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_malformed_cr), patch("time.sleep"):
        with pytest.raises(RangeDownloadError, match="Malformed Content-Range header"):
            fetch_entry_bytes_range("https://fake.url", dummy_zinfo, max_retries=1)

    # Case C: Mismatched range offsets
    def mock_mismatched_cr(req, timeout=30):
        mock_resp = MagicMock()
        mock_resp.status = 206
        mock_resp.headers = {"Content-Range": f"bytes 0-{fetch_len - 1}/50000"}
        mock_resp.read.return_value = b"\x00" * fetch_len
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_mismatched_cr), patch("time.sleep"):
        with pytest.raises(RangeDownloadError, match="Content-Range mismatch"):
            fetch_entry_bytes_range("https://fake.url", dummy_zinfo, max_retries=1)


def test_fetch_entry_bytes_range_short_response_body(tmp_path):
    """Verify validation rejects responses where received byte count does not match fetch_len."""
    dummy_zinfo = zipfile.ZipInfo("img/short.png")
    dummy_zinfo.header_offset = 200
    dummy_zinfo.compress_size = 400
    dummy_zinfo.file_size = 700
    dummy_zinfo.compress_type = zipfile.ZIP_DEFLATED
    dummy_zinfo.CRC = 8888

    fetch_len = 30 + len(dummy_zinfo.filename) + 256 + dummy_zinfo.compress_size
    end_byte = dummy_zinfo.header_offset + fetch_len - 1

    def mock_short_body(req, timeout=30):
        mock_resp = MagicMock()
        mock_resp.status = 206
        mock_resp.headers = {"Content-Range": f"bytes {dummy_zinfo.header_offset}-{end_byte}/100000"}
        mock_resp.read.return_value = b"\x00" * 50
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_short_body), patch("time.sleep"):
        with pytest.raises(RangeDownloadError, match="Received byte count mismatch"):
            fetch_entry_bytes_range("https://fake.url", dummy_zinfo, max_retries=1)


def test_fetch_entry_bytes_range_successful_retry_transient_error(tmp_path):
    """Verify fetch_entry_bytes_range retries after URLError and TimeoutError and succeeds."""
    zip_path = tmp_path / "transient_test.zip"
    payload = b"Transient network glitch payload recovery"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("img/transient.png", payload)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zinfo = zf.getinfo("img/transient.png")

    zip_bytes = zip_path.read_bytes()
    fetch_len = 30 + len(zinfo.filename) + 256 + zinfo.compress_size
    extended_bytes = zip_bytes + b"\x00" * fetch_len
    valid_slice = extended_bytes[zinfo.header_offset : zinfo.header_offset + fetch_len]

    attempt_count = 0

    def mock_transient_failures(req, timeout=30):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise urllib.error.URLError("Temporary name resolution failure")
        elif attempt_count == 2:
            raise TimeoutError("Socket read timed out")

        mock_resp = MagicMock()
        mock_resp.status = 206
        mock_resp.headers = {
            "Content-Range": f"bytes {zinfo.header_offset}-{zinfo.header_offset + fetch_len - 1}/{len(extended_bytes)}"
        }
        mock_resp.read.return_value = valid_slice
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_transient_failures), patch("time.sleep") as mock_sleep:
        res = fetch_entry_bytes_range("https://fake.zenodo.org/file.zip", zinfo, max_retries=5, backoff_base=0.01)

    assert res == payload
    assert attempt_count == 3
    assert mock_sleep.call_count == 2


def test_selective_acquisition_resumption_and_corruption_recovery(tmp_path):
    """Verify run_selective_acquisition skips valid existing files and re-downloads corrupted files."""
    # 1. Build synthetic ZIP archive with 4 balanced images (2 phishing, 2 legitimate)
    zip_path = tmp_path / "resumption_zenodo.zip"
    csv_text = (
        "url;entropy;url_length;label\n"
        "https://site-valid.com;3.1;22;phishing\n"
        "https://site-corrupt.com;3.2;24;legitimate\n"
        "https://site-missing.com;3.3;24;phishing\n"
        "https://site-legit2.org;3.4;24;legitimate\n"
    )

    arr = np.zeros((50, 50, 3), dtype=np.uint8)
    arr[:, :] = [100, 200, 50]
    _, valid_png_bytes = cv2.imencode(".png", arr)
    valid_data = valid_png_bytes.tobytes()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Phishing dataset/url/merged_features_final.csv", csv_text)
        zf.writestr("Phishing dataset/image/phishing/2_site-valid.com.png", valid_data)
        zf.writestr("Phishing dataset/image/legitimate/3_site-corrupt.com.png", valid_data)
        zf.writestr("Phishing dataset/image/phishing/4_site-missing.com.png", valid_data)
        zf.writestr("Phishing dataset/image/legitimate/5_site-legit2.org.png", valid_data)

    extract_dir = tmp_path / "extracted_screenshots"
    manifest_path = tmp_path / "manifest.jsonl"
    report_path = tmp_path / "quality_report.json"

    # Pre-create sample 1 (valid)
    valid_dest = extract_dir / "Phishing dataset" / "image" / "phishing" / "2_site-valid.com.png"
    valid_dest.parent.mkdir(parents=True, exist_ok=True)
    valid_dest.write_bytes(valid_data)

    # Pre-create sample 2 (corrupt: invalid PNG content)
    corrupt_dest = extract_dir / "Phishing dataset" / "image" / "legitimate" / "3_site-corrupt.com.png"
    corrupt_dest.parent.mkdir(parents=True, exist_ok=True)
    corrupt_dest.write_bytes(b"CORRUPTED_NOT_A_PNG")

    # Samples 3 and 4 do not exist yet

    count, report = run_selective_acquisition(
        zip_source=zip_path,
        extract_dir=extract_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        max_samples=4,
        dry_run=False,
    )

    assert count == 4
    assert report["skipped_already_valid_count"] == 1
    assert report["total_pairs_acquired"] == 4

    # Sample 1 was skipped: file size preserved
    assert os.path.getsize(valid_dest) == len(valid_data)

    # Sample 2 was re-downloaded: now valid PNG
    assert os.path.getsize(corrupt_dest) == len(valid_data)
    img_check2 = inspect_image(corrupt_dest)
    assert img_check2["is_valid"] is True

    # Sample 3 was downloaded: now valid PNG
    missing_dest = extract_dir / "Phishing dataset" / "image" / "phishing" / "4_site-missing.com.png"
    assert os.path.isfile(missing_dest)
    assert inspect_image(missing_dest)["is_valid"] is True


def test_fetch_entry_bytes_range_crc_mismatch(tmp_path):
    """Verify fetch_entry_bytes_range rejects decompressed data that does not match ZIP CRC-32."""
    zip_path = tmp_path / "crc_test.zip"
    payload = b"Original uncorrupted test payload content 12345"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("img/crc.png", payload)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zinfo = zf.getinfo("img/crc.png")

    # Alter expected CRC in zinfo to force mismatch
    zinfo.CRC = 99999999

    zip_bytes = zip_path.read_bytes()
    fetch_len = 30 + len(zinfo.filename) + 256 + zinfo.compress_size
    extended_bytes = zip_bytes + b"\x00" * fetch_len
    valid_slice = extended_bytes[zinfo.header_offset : zinfo.header_offset + fetch_len]

    def mock_urlopen(req, timeout=30):
        mock_resp = MagicMock()
        mock_resp.status = 206
        mock_resp.headers = {
            "Content-Range": f"bytes {zinfo.header_offset}-{zinfo.header_offset + fetch_len - 1}/{len(extended_bytes)}"
        }
        mock_resp.read.return_value = valid_slice
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen), patch("time.sleep"):
        with pytest.raises(RangeDownloadError, match="CRC-32 mismatch"):
            fetch_entry_bytes_range("https://fake.url", zinfo, max_retries=1)


def test_selective_acquisition_failed_entry_isolation(tmp_path):
    """Verify failed screenshot downloads do not abort dataset acquisition and are reported."""
    zip_path = tmp_path / "isolation_test.zip"
    csv_text = (
        "url;entropy;url_length;label\n"
        "https://failing-entry.com;3.1;25;phishing\n"
        "https://succeeding-entry.org;3.0;27;legitimate\n"
    )

    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:, :] = [0, 255, 0]
    _, valid_png_bytes = cv2.imencode(".png", arr)
    valid_data = valid_png_bytes.tobytes()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Phishing dataset/url/merged_features_final.csv", csv_text)
        zf.writestr("Phishing dataset/image/phishing/2_failing-entry.com.png", valid_data)
        zf.writestr("Phishing dataset/image/legitimate/3_succeeding-entry.org.png", valid_data)

    extract_dir = tmp_path / "extracted_screenshots"
    manifest_path = tmp_path / "manifest.jsonl"
    report_path = tmp_path / "quality_report.json"

    # Simulate fetch error for failing-entry.com only
    orig_open = zipfile.ZipFile.open

    def selective_open_mock(self, name_or_info, mode="r", *args, **kwargs):
        fname = name_or_info.filename if hasattr(name_or_info, "filename") else str(name_or_info)
        if "failing-entry.com" in fname:
            raise RangeDownloadError("Simulated persistent network drop on failing entry")
        return orig_open(self, name_or_info, mode, *args, **kwargs)

    with patch.object(zipfile.ZipFile, "open", selective_open_mock):
        count, report = run_selective_acquisition(
            zip_source=zip_path,
            extract_dir=extract_dir,
            manifest_path=manifest_path,
            report_path=report_path,
            max_samples=2,
            dry_run=False,
        )

    assert count == 1
    assert report["total_pairs_acquired"] == 1
    assert report["failed_download_count"] == 1
    assert len(report["failed_entries"]) == 1
    assert report["failed_entries"][0]["filename"] == "Phishing dataset/image/phishing/2_failing-entry.com.png"
    assert "Simulated persistent network drop" in report["failed_entries"][0]["error"]

    # Succeeding entry was written
    success_file = extract_dir / "Phishing dataset" / "image" / "legitimate" / "3_succeeding-entry.org.png"
    assert os.path.isfile(success_file)
    assert inspect_image(success_file)["is_valid"] is True
