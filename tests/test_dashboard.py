"""
Tests for the PhishGuard Dashboard

Validates statistics computation, three-class threat categorization,
record filtering, and dataframe formatting without requiring a live backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from dashboard.app import (
    _format_category,
    _format_timestamp,
    _prediction_label,
    _risk_display,
    _threat_category_label,
    analyze_new_url,
    compute_statistics,
    filter_records,
    records_to_dataframe,
    validate_url_input,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mixed_records() -> list[dict]:
    """A realistic mix of phishing, legitimate, and model_unavailable records."""
    return [
        {
            "id": 1,
            "url": "https://example.com",
            "prediction": 0,
            "threat_category": "legitimate",
            "risk_score": 0.12,
            "analysis_status": "completed",
            "analyzed_at": "2026-11-01T10:00:00",
        },
        {
            "id": 2,
            "url": "http://evil-login.xyz/verify",
            "prediction": 1,
            "threat_category": "phishing",
            "risk_score": 0.94,
            "analysis_status": "completed",
            "analyzed_at": "2026-11-01T11:00:00",
        },
        {
            "id": 3,
            "url": "https://another-site.com",
            "prediction": None,
            "threat_category": None,
            "risk_score": None,
            "analysis_status": "model_unavailable",
            "analyzed_at": "2026-11-01T12:00:00",
        },
        {
            "id": 4,
            "url": "https://safe-bank.com",
            "prediction": 0,
            "threat_category": "legitimate",
            "risk_score": 0.05,
            "analysis_status": "completed",
            "analyzed_at": "2026-11-02T08:00:00",
        },
        {
            "id": 5,
            "url": "http://phish.tk/account",
            "prediction": 1,
            "threat_category": "phishing",
            "risk_score": 0.88,
            "analysis_status": "completed",
            "analyzed_at": "2026-11-02T09:00:00",
        },
    ]


@pytest.fixture
def three_class_records() -> list[dict]:
    """A full 3-class dataset fixture with legitimate, phishing, and malware samples."""
    return [
        {
            "id": 1,
            "url": "https://www.wikipedia.org",
            "prediction": 0,
            "threat_category": "legitimate",
            "risk_score": 0.051,
            "analysis_status": "completed",
            "class_probabilities": {"legitimate": 0.949, "phishing": 0.051, "malware_distribution": 0.0},
            "top_features": [
                {"feature_name": "is_https", "feature_value": 1, "contribution": 0.151, "impact": "supports_legitimate", "explanation": "HTTPS enabled"}
            ],
            "analyzed_at": "2026-11-03T10:00:00",
        },
        {
            "id": 2,
            "url": "https://secure-paypal-login.tk/signin",
            "prediction": 1,
            "threat_category": "phishing",
            "risk_score": 0.985,
            "analysis_status": "completed",
            "class_probabilities": {"legitimate": 0.015, "phishing": 0.985, "malware_distribution": 0.0},
            "top_features": [
                {"feature_name": "has_suspicious_tld", "feature_value": 1, "contribution": 0.28, "impact": "increases_phishing_risk", "explanation": "Suspicious TLD"}
            ],
            "analyzed_at": "2026-11-03T11:00:00",
        },
        {
            "id": 3,
            "url": "http://192.168.1.1:8080/setup.exe",
            "prediction": 2,
            "threat_category": "malware_distribution",
            "risk_score": 0.999,
            "analysis_status": "completed",
            "class_probabilities": {"legitimate": 0.001, "phishing": 0.009, "malware_distribution": 0.990},
            "top_features": [
                {"feature_name": "has_ip_address", "feature_value": 1, "contribution": 0.35, "impact": "increases_malware_risk", "explanation": "Direct raw IP"}
            ],
            "analyzed_at": "2026-11-03T12:00:00",
        },
        {
            "id": 4,
            "url": "https://offline-test.com",
            "prediction": None,
            "threat_category": None,
            "risk_score": None,
            "analysis_status": "model_unavailable",
            "class_probabilities": None,
            "top_features": None,
            "analyzed_at": "2026-11-03T13:00:00",
        },
    ]


# ── Tests: compute_statistics ───────────────────────────────────────

class TestComputeStatistics:
    """Statistics accuracy across legitimate, phishing, malware, and unavailable."""

    def test_correct_totals(self, mixed_records: list[dict]):
        stats = compute_statistics(mixed_records)
        assert stats["total"] == 5

    def test_phishing_count(self, mixed_records: list[dict]):
        stats = compute_statistics(mixed_records)
        assert stats["phishing"] == 2

    def test_legitimate_count(self, mixed_records: list[dict]):
        stats = compute_statistics(mixed_records)
        assert stats["legitimate"] == 2

    def test_unavailable_count(self, mixed_records: list[dict]):
        stats = compute_statistics(mixed_records)
        assert stats["unavailable"] == 1

    def test_model_unavailable_never_counted_as_phishing_or_legitimate(self):
        records = [
            {"prediction": None, "analysis_status": "model_unavailable"},
            {"prediction": None, "analysis_status": "model_unavailable"},
            {"prediction": None, "analysis_status": "model_unavailable"},
        ]
        stats = compute_statistics(records)
        assert stats["phishing"] == 0
        assert stats["legitimate"] == 0
        assert stats["malware"] == 0
        assert stats["unavailable"] == 3

    def test_empty_records(self):
        stats = compute_statistics([])
        assert stats == {
            "total": 0,
            "phishing": 0,
            "legitimate": 0,
            "malware": 0,
            "unavailable": 0,
            "threat_total": 0,
        }

    def test_three_class_statistics(self, three_class_records: list[dict]):
        stats = compute_statistics(three_class_records)
        assert stats["total"] == 4
        assert stats["legitimate"] == 1
        assert stats["phishing"] == 1
        assert stats["malware"] == 1
        assert stats["unavailable"] == 1
        assert stats["threat_total"] == 2  # phishing + malware

    def test_all_phishing(self):
        records = [
            {"prediction": 1, "analysis_status": "completed"},
            {"prediction": 1, "analysis_status": "completed"},
        ]
        stats = compute_statistics(records)
        assert stats["phishing"] == 2
        assert stats["legitimate"] == 0
        assert stats["malware"] == 0
        assert stats["threat_total"] == 2

    def test_all_legitimate(self):
        records = [
            {"prediction": 0, "analysis_status": "completed"},
            {"prediction": 0, "analysis_status": "completed"},
        ]
        stats = compute_statistics(records)
        assert stats["phishing"] == 0
        assert stats["legitimate"] == 2
        assert stats["malware"] == 0
        assert stats["threat_total"] == 0

    def test_all_malware(self):
        records = [
            {"prediction": 2, "analysis_status": "completed"},
            {"prediction": "2", "threat_category": "malware_distribution", "analysis_status": "completed"},
        ]
        stats = compute_statistics(records)
        assert stats["malware"] == 2
        assert stats["phishing"] == 0
        assert stats["legitimate"] == 0
        assert stats["threat_total"] == 2

    def test_string_prediction_values(self):
        records = [
            {"prediction": "1", "analysis_status": "completed"},
            {"prediction": "0", "analysis_status": "completed"},
            {"prediction": "2", "analysis_status": "completed"},
        ]
        stats = compute_statistics(records)
        assert stats["phishing"] == 1
        assert stats["legitimate"] == 1
        assert stats["malware"] == 1
        assert stats["threat_total"] == 2

    def test_null_prediction_with_completed_status_is_unavailable(self):
        records = [{"prediction": None, "analysis_status": "completed"}]
        stats = compute_statistics(records)
        assert stats["unavailable"] == 1
        assert stats["phishing"] == 0
        assert stats["legitimate"] == 0
        assert stats["malware"] == 0

    def test_does_not_invent_fake_statistics(self):
        records = [
            {"prediction": 1, "analysis_status": "completed"},
        ]
        stats = compute_statistics(records)
        assert stats["total"] == 1
        assert stats["phishing"] == 1
        assert stats["legitimate"] == 0
        assert stats["malware"] == 0
        assert stats["unavailable"] == 0


# ── Tests: _prediction_label ────────────────────────────────────────

class TestPredictionLabel:
    def test_phishing_label(self):
        assert "Phishing" in _prediction_label(1, "completed")

    def test_legitimate_label(self):
        assert "Legitimate" in _prediction_label(0, "completed")

    def test_malware_label(self):
        assert "Malware" in _prediction_label(2, "completed")

    def test_unavailable_label(self):
        label = _prediction_label(None, "model_unavailable")
        assert "Unavailable" in label
        assert "Phishing" not in label
        assert "Legitimate" not in label
        assert "Malware" not in label

    def test_none_prediction_not_labelled_safe(self):
        label = _prediction_label(None, "completed")
        assert "Legitimate" not in label
        assert "Phishing" not in label
        assert "Malware" not in label

    def test_string_prediction(self):
        assert "Phishing" in _prediction_label("1", "completed")
        assert "Legitimate" in _prediction_label("0", "completed")
        assert "Malware" in _prediction_label("2", "completed")

    def test_threat_category_parameter(self):
        assert "Malware" in _prediction_label(None, "completed", threat_category="malware_distribution")
        assert "Phishing" in _prediction_label(None, "completed", threat_category="phishing")
        assert "Legitimate" in _prediction_label(None, "completed", threat_category="legitimate")


# ── Tests: _threat_category_label ───────────────────────────────────

class TestThreatCategoryLabel:
    def test_categories(self):
        assert _threat_category_label("legitimate") == "Legitimate"
        assert _threat_category_label("phishing") == "Phishing"
        assert _threat_category_label("malware_distribution") == "Malware Distribution"
        assert _threat_category_label("malware") == "Malware Distribution"
        assert _threat_category_label(None, status="model_unavailable") == "Unavailable"
        assert _threat_category_label(None, prediction=2) == "Malware Distribution"


# ── Tests: _risk_display ────────────────────────────────────────────

class TestRiskDisplay:
    def test_normal_risk_score(self):
        assert _risk_display(0.94, "completed") == "94.0%"

    def test_zero_risk(self):
        assert _risk_display(0.0, "completed") == "0.0%"

    def test_unavailable_status_shows_dash(self):
        assert _risk_display(None, "model_unavailable") == "—"

    def test_none_risk_shows_dash(self):
        assert _risk_display(None, "completed") == "—"


# ── Tests: _format_timestamp ────────────────────────────────────────

class TestFormatTimestamp:
    def test_iso_format(self):
        result = _format_timestamp("2026-11-01T10:00:00")
        assert "2026-11-01" in result
        assert "10:00:00" in result

    def test_none_returns_dash(self):
        assert _format_timestamp(None) == "—"

    def test_empty_string_returns_dash(self):
        assert _format_timestamp("") == "—"

    def test_iso_with_z_suffix(self):
        result = _format_timestamp("2026-11-01T10:00:00Z")
        assert "2026-11-01" in result


# ── Tests: filter_records ───────────────────────────────────────────

class TestFilterRecords:
    def test_filter_by_query(self, three_class_records: list[dict]):
        results = filter_records(three_class_records, query="wikipedia")
        assert len(results) == 1
        assert results[0]["url"] == "https://www.wikipedia.org"

    def test_filter_by_category_legitimate(self, three_class_records: list[dict]):
        results = filter_records(three_class_records, category="Legitimate")
        assert len(results) == 1
        assert results[0]["threat_category"] == "legitimate"

    def test_filter_by_category_phishing(self, three_class_records: list[dict]):
        results = filter_records(three_class_records, category="Phishing")
        assert len(results) == 1
        assert results[0]["threat_category"] == "phishing"

    def test_filter_by_category_malware(self, three_class_records: list[dict]):
        results = filter_records(three_class_records, category="Malware Distribution")
        assert len(results) == 1
        assert results[0]["threat_category"] == "malware_distribution"

    def test_filter_by_status(self, three_class_records: list[dict]):
        results = filter_records(three_class_records, status="model_unavailable")
        assert len(results) == 1
        assert results[0]["analysis_status"] == "model_unavailable"

    def test_filter_combined(self, three_class_records: list[dict]):
        results = filter_records(three_class_records, query="paypal", category="Phishing", status="completed")
        assert len(results) == 1
        assert "paypal" in results[0]["url"]

    def test_filter_no_match(self, three_class_records: list[dict]):
        results = filter_records(three_class_records, query="nonexistent-xyz-domain")
        assert len(results) == 0


# ── Tests: records_to_dataframe ─────────────────────────────────────

class TestRecordsToDataFrame:
    def test_empty_records(self):
        df = records_to_dataframe([])
        assert df.empty
        assert "URL" in df.columns
        assert "Prediction" in df.columns
        assert "Threat Category" in df.columns

    def test_correct_columns(self, mixed_records: list[dict]):
        df = records_to_dataframe(mixed_records)
        expected_cols = {"ID", "URL", "Prediction", "Threat Category", "Risk Score", "Status", "Analyzed At"}
        assert set(df.columns) == expected_cols

    def test_correct_row_count(self, mixed_records: list[dict]):
        df = records_to_dataframe(mixed_records)
        assert len(df) == 5

    def test_prediction_labels_in_dataframe(self, mixed_records: list[dict]):
        df = records_to_dataframe(mixed_records)
        predictions = df["Prediction"].tolist()
        assert any("Phishing" in p for p in predictions)
        assert any("Legitimate" in p for p in predictions)
        assert any("Unavailable" in p for p in predictions)

    def test_three_class_dataframe(self, three_class_records: list[dict]):
        df = records_to_dataframe(three_class_records)
        assert len(df) == 4
        categories = df["Threat Category"].tolist()
        assert "Legitimate" in categories
        assert "Phishing" in categories
        assert "Malware Distribution" in categories
        assert "Unavailable" in categories

    def test_model_unavailable_not_counted_as_safe_in_dataframe(self, mixed_records: list[dict]):
        df = records_to_dataframe(mixed_records)
        unavailable_rows = df[df["Status"] == "model_unavailable"]
        for _, row in unavailable_rows.iterrows():
            assert "Legitimate" not in row["Prediction"]
            assert "Phishing" not in row["Prediction"]
            assert "Malware" not in row["Prediction"]


# ── URL Input Validation Tests ──────────────────────────────────────

class TestValidateURLInput:
    def test_valid_https_url(self):
        valid, cleaned = validate_url_input("https://www.example.com/test?q=1")
        assert valid is True
        assert cleaned == "https://www.example.com/test?q=1"

    def test_valid_http_url(self):
        valid, cleaned = validate_url_input("http://sub.domain.org:8080/path")
        assert valid is True
        assert cleaned == "http://sub.domain.org:8080/path"

    def test_whitespace_stripped(self):
        valid, cleaned = validate_url_input("   https://clean-me.com   ")
        assert valid is True
        assert cleaned == "https://clean-me.com"

    def test_reject_empty_and_none(self):
        valid, err = validate_url_input("")
        assert valid is False
        assert "empty" in err.lower()

        valid, err = validate_url_input(None)
        assert valid is False
        assert "empty" in err.lower()

        valid, err = validate_url_input("   ")
        assert valid is False
        assert "empty" in err.lower()

    def test_reject_unsupported_schemes(self):
        for bad in ["ftp://example.com", "javascript:alert(1)", "file:///etc/passwd", "data:text/html,abc"]:
            valid, err = validate_url_input(bad)
            assert valid is False
            assert "HTTP and HTTPS" in err

    def test_reject_missing_host(self):
        for bad in ["http://", "https://", "https:///path"]:
            valid, err = validate_url_input(bad)
            assert valid is False
            assert "domain" in err.lower() or "host" in err.lower()


# ── Analyze New URL API Client Tests ────────────────────────────────

class TestAnalyzeNewURL:
    def test_client_side_validation_rejects_invalid_url_without_http_request(self):
        with patch("requests.post") as mock_post:
            result, err = analyze_new_url("not-a-valid-url")
            assert result is None
            assert err is not None
            assert "HTTP and HTTPS" in err
            mock_post.assert_not_called()

    def test_successful_legitimate_url_analysis(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "url": "https://www.wikipedia.org",
            "prediction": 0,
            "threat_category": "legitimate",
            "risk_score": 0.051,
            "class_probabilities": {
                "legitimate": 0.949,
                "phishing": 0.051,
                "malware_distribution": 0.0,
            },
            "top_features": [
                {
                    "feature_name": "is_https",
                    "feature_value": 1,
                    "contribution": 0.151,
                    "impact": "supports_legitimate",
                    "explanation": "HTTPS encryption is active.",
                }
            ],
            "analysis_status": "completed",
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            data, err = analyze_new_url("https://www.wikipedia.org", backend_url="http://127.0.0.1:8000")
            assert err is None
            assert data is not None
            assert data["prediction"] == 0
            assert data["threat_category"] == "legitimate"
            assert data["risk_score"] == 0.051
            assert data["class_probabilities"]["legitimate"] == 0.949

            mock_post.assert_called_once_with(
                "http://127.0.0.1:8000/analyze",
                json={"url": "https://www.wikipedia.org"},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

    def test_successful_phishing_url_analysis(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "url": "http://paypal-verification-account.security-check.com",
            "prediction": 1,
            "threat_category": "phishing",
            "risk_score": 0.892,
            "class_probabilities": {
                "legitimate": 0.045,
                "phishing": 0.892,
                "malware_distribution": 0.063,
            },
            "top_features": [
                {
                    "feature_name": "has_ip",
                    "feature_value": 0,
                    "contribution": 0.32,
                    "impact": "supports_phishing",
                    "explanation": "Suspicious domain structure.",
                }
            ],
            "analysis_status": "completed",
        }

        with patch("requests.post", return_value=mock_response):
            data, err = analyze_new_url("http://paypal-verification-account.security-check.com")
            assert err is None
            assert data is not None
            assert data["prediction"] == 1
            assert data["threat_category"] == "phishing"
            assert data["risk_score"] == 0.892

    def test_successful_malware_url_analysis(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "url": "http://192.168.1.100/payload.exe",
            "prediction": 2,
            "threat_category": "malware_distribution",
            "risk_score": 0.965,
            "class_probabilities": {
                "legitimate": 0.012,
                "phishing": 0.023,
                "malware_distribution": 0.965,
            },
            "top_features": [
                {
                    "feature_name": "has_ip",
                    "feature_value": 1,
                    "contribution": 0.45,
                    "impact": "supports_malware",
                    "explanation": "Raw IP address used as host.",
                }
            ],
            "analysis_status": "completed",
        }

        with patch("requests.post", return_value=mock_response):
            data, err = analyze_new_url("http://192.168.1.100/payload.exe")
            assert err is None
            assert data is not None
            assert data["prediction"] == 2
            assert data["threat_category"] == "malware_distribution"
            assert data["risk_score"] == 0.965

    def test_successful_model_unavailable_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "url": "https://example.com",
            "prediction": None,
            "threat_category": None,
            "risk_score": None,
            "analysis_status": "model_unavailable",
        }

        with patch("requests.post", return_value=mock_response):
            data, err = analyze_new_url("https://example.com")
            assert err is None
            assert data is not None
            assert data["analysis_status"] == "model_unavailable"
            assert data["prediction"] is None

    def test_backend_connection_error(self):
        with patch("requests.post", side_effect=requests.ConnectionError("Connection refused")):
            data, err = analyze_new_url("https://example.com", backend_url="http://127.0.0.1:9999")
            assert data is None
            assert err is not None
            assert "Cannot connect to the PhishGuard backend" in err
            assert "http://127.0.0.1:9999" in err

    def test_backend_timeout_error(self):
        with patch("requests.post", side_effect=requests.Timeout("Read timeout")):
            data, err = analyze_new_url("https://example.com", timeout=5)
            assert data is None
            assert err is not None
            assert "timed out after 5 seconds" in err

    def test_backend_422_validation_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.json.return_value = {
            "detail": [{"loc": ["body", "url"], "msg": "URL must begin with http:// or https://", "type": "value_error"}]
        }

        with patch("requests.post", return_value=mock_response):
            data, err = analyze_new_url("https://example.com")
            assert data is None
            assert err is not None
            assert "Backend validation rejected URL" in err
            assert "URL must begin with http:// or https://" in err

    def test_backend_500_http_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        http_err = requests.HTTPError("500 Server Error", response=mock_response)
        mock_response.raise_for_status.side_effect = http_err

        with patch("requests.post", return_value=mock_response):
            data, err = analyze_new_url("https://example.com")
            assert data is None
            assert err is not None
            assert "Backend error: HTTP 500" in err

    def test_backend_unexpected_request_exception(self):
        with patch("requests.post", side_effect=requests.RequestException("Unknown network issue")):
            data, err = analyze_new_url("https://example.com")
            assert data is None
            assert err is not None
            assert "Unexpected error during analysis" in err
