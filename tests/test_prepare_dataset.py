"""Unit tests for ml.training.prepare_dataset."""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from ml.features.url_features import FEATURE_NAMES
from ml.training.prepare_dataset import prepare_dataset


# ── Helpers ─────────────────────────────────────────────────────

def _write_csv(path: str, header: list[str], rows: list[list[str]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _read_output(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


EXPECTED_COLUMNS = {"url", "label"} | set(FEATURE_NAMES)


# ── 1. Basic happy-path ────────────────────────────────────────

class TestBasicPreparation:
    def test_processes_valid_csv(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["url", "label"], [
            ["https://google.com", "legitimate"],
            ["http://evil.xyz/login", "phishing"],
        ])

        stats = prepare_dataset(inp, out)
        assert stats["total_rows"] == 2
        assert stats["valid_rows"] == 2
        assert os.path.isfile(out)

        rows = _read_output(out)
        assert len(rows) == 2
        assert set(rows[0].keys()) == EXPECTED_COLUMNS

    def test_preserves_original_url(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        url = "https://example.com/Some/Path?q=1"
        _write_csv(inp, ["url", "label"], [[url, "benign"]])

        prepare_dataset(inp, out)
        rows = _read_output(out)
        assert rows[0]["url"] == url

    def test_all_20_features_present(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["url", "label"], [
            ["https://test.com", "0"],
        ])

        prepare_dataset(inp, out)
        rows = _read_output(out)
        for name in FEATURE_NAMES:
            assert name in rows[0], f"Missing feature: {name}"


# ── 2. Label normalisation ─────────────────────────────────────

class TestLabelNormalisation:
    @pytest.mark.parametrize("label_text,expected", [
        ("legitimate", "0"),
        ("benign", "0"),
        ("safe", "0"),
        ("0", "0"),
        ("phishing", "1"),
        ("malicious", "1"),
        ("bad", "1"),
        ("1", "1"),
    ])
    def test_text_labels_mapped(self, tmp_path, label_text, expected):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["url", "label"], [
            ["https://example.com", label_text],
        ])

        prepare_dataset(inp, out)
        rows = _read_output(out)
        assert rows[0]["label"] == expected

    def test_case_insensitive_labels(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["url", "label"], [
            ["https://a.com", "Phishing"],
            ["https://b.com", "LEGITIMATE"],
            ["https://c.com", "Benign"],
        ])

        prepare_dataset(inp, out)
        rows = _read_output(out)
        assert rows[0]["label"] == "1"
        assert rows[1]["label"] == "0"
        assert rows[2]["label"] == "0"


# ── 3. Cleaning: missing, invalid, duplicates ──────────────────

class TestDataCleaning:
    def test_drops_missing_urls(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["url", "label"], [
            ["https://ok.com", "legitimate"],
            ["", "phishing"],
            ["   ", "phishing"],
        ])

        stats = prepare_dataset(inp, out)
        assert stats["dropped_missing"] == 2
        assert stats["valid_rows"] == 1

    def test_drops_invalid_labels(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["url", "label"], [
            ["https://ok.com", "legitimate"],
            ["https://unknown.com", "maybe"],
            ["https://another.com", "2"],
        ])

        stats = prepare_dataset(inp, out)
        assert stats["dropped_invalid_label"] == 2
        assert stats["valid_rows"] == 1

    def test_drops_duplicate_urls(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["url", "label"], [
            ["https://dup.com", "legitimate"],
            ["https://dup.com", "phishing"],
            ["https://unique.com", "legitimate"],
        ])

        stats = prepare_dataset(inp, out)
        assert stats["dropped_duplicate"] == 1
        assert stats["valid_rows"] == 2

    def test_combined_cleaning(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["url", "label"], [
            ["https://good.com", "legitimate"],     # keep
            ["", "phishing"],                        # missing URL
            ["https://bad-label.com", "unknown"],    # invalid label
            ["https://good.com", "phishing"],        # duplicate
            ["https://phish.xyz", "malicious"],      # keep
        ])

        stats = prepare_dataset(inp, out)
        assert stats["total_rows"] == 5
        assert stats["dropped_missing"] == 1
        assert stats["dropped_invalid_label"] == 1
        assert stats["dropped_duplicate"] == 1
        assert stats["valid_rows"] == 2


# ── 4. Custom column names ─────────────────────────────────────

class TestCustomColumns:
    def test_custom_url_and_label_columns(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["website", "status"], [
            ["https://example.com", "benign"],
            ["http://bad.tk", "malicious"],
        ])

        stats = prepare_dataset(inp, out, url_column="website", label_column="status")
        assert stats["valid_rows"] == 2

    def test_case_insensitive_column_match(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["URL", "Label"], [
            ["https://a.com", "legitimate"],
        ])

        stats = prepare_dataset(inp, out, url_column="url", label_column="label")
        assert stats["valid_rows"] == 1


# ── 5. Error handling ──────────────────────────────────────────

class TestErrorHandling:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            prepare_dataset("nonexistent.csv", "out.csv")

    def test_missing_url_column(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        _write_csv(inp, ["link", "label"], [
            ["https://a.com", "legitimate"],
        ])
        with pytest.raises(ValueError, match="URL column"):
            prepare_dataset(inp, str(tmp_path / "out.csv"))

    def test_missing_label_column(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        _write_csv(inp, ["url", "type"], [
            ["https://a.com", "legitimate"],
        ])
        with pytest.raises(ValueError, match="Label column"):
            prepare_dataset(inp, str(tmp_path / "out.csv"))

    def test_empty_csv(self, tmp_path):
        inp = str(tmp_path / "empty.csv")
        with open(inp, "w") as f:
            f.write("")
        with pytest.raises(ValueError, match="empty"):
            prepare_dataset(inp, str(tmp_path / "out.csv"))


# ── 6. Output format validation ────────────────────────────────

class TestOutputFormat:
    def test_output_column_order(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["url", "label"], [
            ["https://example.com", "legitimate"],
        ])

        prepare_dataset(inp, out)

        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header[0] == "url"
        assert header[1] == "label"
        assert header[2:] == list(FEATURE_NAMES)

    def test_numeric_feature_values(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "processed.csv")
        _write_csv(inp, ["url", "label"], [
            ["https://test.com/path?q=1", "1"],
        ])

        prepare_dataset(inp, out)
        rows = _read_output(out)
        for name in FEATURE_NAMES:
            val = rows[0][name]
            # Every feature must be parseable as a number
            float(val)  # Raises ValueError if not numeric

    def test_creates_output_directory(self, tmp_path):
        inp = str(tmp_path / "raw.csv")
        out = str(tmp_path / "sub" / "dir" / "processed.csv")
        _write_csv(inp, ["url", "label"], [
            ["https://example.com", "benign"],
        ])

        prepare_dataset(inp, out)
        assert os.path.isfile(out)

