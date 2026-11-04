"""Unit tests for ml.features.url_features.extract_url_features."""

import sys
import os

# Ensure the project root is on sys.path so `ml` is importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from ml.features.url_features import extract_url_features, FEATURE_NAMES


# ── Helpers ─────────────────────────────────────────────────────

def _assert_all_keys(features: dict):
    """Every call must return exactly the 20 canonical keys."""
    assert set(features.keys()) == set(FEATURE_NAMES)
    assert len(features) == 20


def _assert_all_numeric(features: dict):
    """Every value must be int or float."""
    for key, val in features.items():
        assert isinstance(val, (int, float)), f"{key} is {type(val)}"


# ── 1. Legitimate URL ──────────────────────────────────────────

class TestLegitimateURL:
    URL = "https://www.google.com/search?q=openai&lang=en"

    def test_returns_all_features(self):
        f = extract_url_features(self.URL)
        _assert_all_keys(f)
        _assert_all_numeric(f)

    def test_url_length(self):
        f = extract_url_features(self.URL)
        assert f["url_length"] == len(self.URL)

    def test_is_https(self):
        f = extract_url_features(self.URL)
        assert f["is_https"] == 1

    def test_no_ip_address(self):
        f = extract_url_features(self.URL)
        assert f["has_ip_address"] == 0

    def test_no_suspicious_tld(self):
        f = extract_url_features(self.URL)
        assert f["has_suspicious_tld"] == 0

    def test_no_suspicious_keyword(self):
        f = extract_url_features(self.URL)
        assert f["has_suspicious_keyword"] == 0

    def test_num_subdomains(self):
        f = extract_url_features(self.URL)
        # www.google.com → 1 subdomain
        assert f["num_subdomains"] == 1

    def test_query_length(self):
        f = extract_url_features(self.URL)
        # query = "q=openai&lang=en"
        assert f["query_length"] == len("q=openai&lang=en")

    def test_character_counts(self):
        f = extract_url_features(self.URL)
        assert f["num_dots"] == self.URL.count(".")
        assert f["num_slashes"] == self.URL.count("/")
        assert f["num_question_marks"] == 1
        assert f["num_ampersands"] == 1
        assert f["num_equals"] == 2

    def test_entropy_positive(self):
        f = extract_url_features(self.URL)
        assert f["url_entropy"] > 0


# ── 2. Suspicious / Phishing URL ──────────────────────────────

class TestSuspiciousURL:
    URL = "http://192.168.1.1/login/verify-account?user=admin&pass=1234"

    def test_returns_all_features(self):
        f = extract_url_features(self.URL)
        _assert_all_keys(f)
        _assert_all_numeric(f)

    def test_is_not_https(self):
        f = extract_url_features(self.URL)
        assert f["is_https"] == 0

    def test_has_ip_address(self):
        f = extract_url_features(self.URL)
        assert f["has_ip_address"] == 1

    def test_has_suspicious_keyword(self):
        f = extract_url_features(self.URL)
        # "login" and "verify" are both in the URL
        assert f["has_suspicious_keyword"] == 1

    def test_many_digits(self):
        f = extract_url_features(self.URL)
        assert f["num_digits"] >= 10  # IP digits + 1234

    def test_suspicious_tld_detection(self):
        f = extract_url_features("http://evil-site.xyz/free-prize")
        assert f["has_suspicious_tld"] == 1
        assert f["has_suspicious_keyword"] == 1  # "free" + "prize"

    def test_deep_subdomains(self):
        f = extract_url_features("http://a.b.c.d.example.com/phish")
        # a.b.c.d.example.com → 4 subdomains
        assert f["num_subdomains"] == 4


# ── 3. Malformed URLs ──────────────────────────────────────────

class TestMalformedURL:
    def test_no_scheme(self):
        f = extract_url_features("example.com/path")
        _assert_all_keys(f)
        assert f["url_length"] > 0
        assert f["hostname_length"] > 0

    def test_garbage_string(self):
        f = extract_url_features("not-a-url-at-all")
        _assert_all_keys(f)
        _assert_all_numeric(f)

    def test_only_scheme(self):
        f = extract_url_features("https://")
        _assert_all_keys(f)
        assert f["hostname_length"] == 0

    def test_special_characters(self):
        f = extract_url_features("http://exam%20ple.com/a@b?c=d&e=f#g")
        _assert_all_keys(f)
        _assert_all_numeric(f)
        assert f["num_percent"] >= 1
        assert f["num_at"] >= 1


# ── 4. Empty / None / Invalid Input ───────────────────────────

class TestEmptyInput:
    def test_empty_string(self):
        f = extract_url_features("")
        _assert_all_keys(f)
        assert all(v == 0 or v == 0.0 for v in f.values())

    def test_none(self):
        f = extract_url_features(None)
        _assert_all_keys(f)
        assert all(v == 0 or v == 0.0 for v in f.values())

    def test_whitespace_only(self):
        f = extract_url_features("   ")
        _assert_all_keys(f)
        assert all(v == 0 or v == 0.0 for v in f.values())

    def test_non_string_input(self):
        f = extract_url_features(12345)
        _assert_all_keys(f)
        assert all(v == 0 or v == 0.0 for v in f.values())


# ── 5. Specific Feature Values ─────────────────────────────────

class TestSpecificValues:
    def test_path_length(self):
        f = extract_url_features("https://example.com/a/long/nested/path")
        assert f["path_length"] == len("/a/long/nested/path")

    def test_query_length_no_query(self):
        f = extract_url_features("https://example.com/page")
        assert f["query_length"] == 0

    def test_http_vs_https(self):
        f_http = extract_url_features("http://example.com")
        f_https = extract_url_features("https://example.com")
        assert f_http["is_https"] == 0
        assert f_https["is_https"] == 1

    def test_hyphens_and_underscores(self):
        f = extract_url_features("http://my-site_test.com/path-a_b")
        assert f["num_hyphens"] == 2
        assert f["num_underscores"] == 2

    def test_entropy_single_char(self):
        # A URL of repeated chars has entropy 0 ... but real URLs won't be
        # all-same-char.  A short varied URL has nonzero entropy.
        f = extract_url_features("aaa")
        assert f["url_entropy"] == 0.0

    def test_feature_count_is_20(self):
        f = extract_url_features("https://test.com")
        assert len(f) == 20

