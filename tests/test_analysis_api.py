"""Unit and integration tests for PhishGuard URL Analysis API."""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sklearn.ensemble import RandomForestClassifier

from backend.database.session import create_db_engine, get_db, init_db
from backend.main import app
from ml.features.url_features import FEATURE_NAMES


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def app_with_temp_db(tmp_path, monkeypatch):
    """Point FastAPI database session dependency to an isolated SQLite file."""
    db_file = str(tmp_path / "api_test.db")
    monkeypatch.setenv("PHISHGUARD_DB_PATH", db_file)

    engine = create_db_engine(db_file)
    init_db(engine_override=engine)

    from sqlalchemy.orm import sessionmaker

    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield db_file
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture
async def client(app_with_temp_db):
    """Async test client bound to the FastAPI application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def trained_test_model(tmp_path, monkeypatch):
    """Create and configure a small test RandomForest model."""
    model_file = str(tmp_path / "test_url_model.joblib")

    # Synthetic binary dataset of 20 features
    X = np.array(
        [[0.0] * 20, [0.1] * 20, [0.05] * 20, [5.0] * 20, [10.0] * 20, [8.0] * 20],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, model_file)

    monkeypatch.setenv("PHISHGUARD_MODEL_PATH", model_file)
    return model_file


# ── 1. URL Scheme & Format Validation ──────────────────────────

class TestURLValidation:
    @pytest.mark.anyio
    async def test_valid_https_url(self, client, monkeypatch, tmp_path):
        monkeypatch.setenv("PHISHGUARD_MODEL_PATH", str(tmp_path / "nonexistent.joblib"))
        resp = await client.post("/analyze", json={"url": "https://example.com/login"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["url"] == "https://example.com/login"

    @pytest.mark.anyio
    async def test_valid_http_url(self, client, monkeypatch, tmp_path):
        monkeypatch.setenv("PHISHGUARD_MODEL_PATH", str(tmp_path / "nonexistent.joblib"))
        resp = await client.post("/analyze", json={"url": "http://insecure-site.org"})
        assert resp.status_code == 200
        assert resp.json()["url"] == "http://insecure-site.org"

    @pytest.mark.anyio
    async def test_rejects_ftp_scheme(self, client):
        resp = await client.post("/analyze", json={"url": "ftp://files.example.com"})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_missing_scheme(self, client):
        resp = await client.post("/analyze", json={"url": "example.com"})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_empty_url(self, client):
        resp = await client.post("/analyze", json={"url": ""})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_non_string_url(self, client):
        resp = await client.post("/analyze", json={"url": 12345})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_missing_url_field(self, client):
        resp = await client.post("/analyze", json={})
        assert resp.status_code == 422


# ── 2. Model Unavailable Behavior ──────────────────────────────

class TestModelUnavailable:
    @pytest.mark.anyio
    async def test_model_unavailable_response(self, client, monkeypatch, tmp_path):
        # Explicitly point to a non-existent model file
        missing_model = str(tmp_path / "no_model.joblib")
        monkeypatch.setenv("PHISHGUARD_MODEL_PATH", missing_model)

        resp = await client.post("/analyze", json={"url": "https://example.com"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["prediction"] is None
        assert data["risk_score"] is None
        assert data["analysis_status"] == "model_unavailable"
        assert "id" in data
        assert "analyzed_at" in data

    @pytest.mark.anyio
    async def test_persists_even_when_model_unavailable(self, client, monkeypatch, tmp_path):
        missing_model = str(tmp_path / "no_model.joblib")
        monkeypatch.setenv("PHISHGUARD_MODEL_PATH", missing_model)

        resp = await client.post("/analyze", json={"url": "https://stored-without-model.com"})
        data = resp.json()
        analysis_id = data["id"]

        # Fetch from database via GET /analyses/{id}
        fetch_resp = await client.get(f"/analyses/{analysis_id}")
        assert fetch_resp.status_code == 200
        fetched = fetch_resp.json()
        assert fetched["url"] == "https://stored-without-model.com"
        assert fetched["analysis_status"] == "model_unavailable"


# ── 3. Model Available Prediction & Scoring ────────────────────

class TestModelInference:
    @pytest.mark.anyio
    async def test_prediction_with_model(self, client, trained_test_model):
        resp = await client.post("/analyze", json={"url": "https://normal-page.com"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["prediction"] in (0, 1)
        assert isinstance(data["risk_score"], (int, float))
        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["analysis_status"] == "completed"

    @pytest.mark.anyio
    async def test_phishing_prediction_persisted(self, client, trained_test_model):
        resp = await client.post("/analyze", json={"url": "http://suspicious-target.xyz/login"})
        data = resp.json()

        get_resp = await client.get(f"/analyses/{data['id']}")
        assert get_resp.status_code == 200
        persisted = get_resp.json()
        assert persisted["prediction"] == data["prediction"]
        assert persisted["risk_score"] == data["risk_score"]
        assert persisted["analysis_status"] == "completed"


# ── 4. History Retrieval (GET /analyses) ───────────────────────

class TestAnalysesHistory:
    @pytest.mark.anyio
    async def test_list_analyses_empty(self, client):
        resp = await client.get("/analyses")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.anyio
    async def test_list_analyses_ordered_newest_first(self, client, monkeypatch, tmp_path):
        monkeypatch.setenv("PHISHGUARD_MODEL_PATH", str(tmp_path / "none.joblib"))

        await client.post("/analyze", json={"url": "https://first.com"})
        await client.post("/analyze", json={"url": "https://second.com"})
        await client.post("/analyze", json={"url": "https://third.com"})

        resp = await client.get("/analyses")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 3
        # Newest first
        assert items[0]["url"] == "https://third.com"
        assert items[1]["url"] == "https://second.com"
        assert items[2]["url"] == "https://first.com"

    @pytest.mark.anyio
    async def test_list_analyses_pagination(self, client, monkeypatch, tmp_path):
        monkeypatch.setenv("PHISHGUARD_MODEL_PATH", str(tmp_path / "none.joblib"))

        for i in range(5):
            await client.post("/analyze", json={"url": f"https://site{i}.org"})

        resp_limit = await client.get("/analyses?limit=2&offset=0")
        assert resp_limit.status_code == 200
        assert len(resp_limit.json()) == 2

        resp_offset = await client.get("/analyses?limit=2&offset=2")
        assert resp_offset.status_code == 200
        assert len(resp_offset.json()) == 2


# ── 5. Single Record Retrieval & 404 Handling ──────────────────

class TestSingleRecordRetrieval:
    @pytest.mark.anyio
    async def test_get_analysis_success(self, client, monkeypatch, tmp_path):
        monkeypatch.setenv("PHISHGUARD_MODEL_PATH", str(tmp_path / "none.joblib"))
        post_resp = await client.post("/analyze", json={"url": "https://target-test.net"})
        created = post_resp.json()

        get_resp = await client.get(f"/analyses/{created['id']}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["id"] == created["id"]
        assert data["url"] == "https://target-test.net"

    @pytest.mark.anyio
    async def test_get_analysis_not_found_returns_404(self, client):
        resp = await client.get("/analyses/999999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ── 6. Preserved Existing Routes ───────────────────────────────

class TestPreservedRoutes:
    @pytest.mark.anyio
    async def test_root_route_preserved(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["project"] == "PhishGuard"
        assert data["status"] == "online"

    @pytest.mark.anyio
    async def test_health_route_preserved(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


# ── 7. Swagger UI & OpenAPI Route Registration ─────────────────

class TestOpenAPISwaggerUI:
    @pytest.mark.anyio
    async def test_openapi_schema_contains_all_analysis_routes(self, client):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        paths = schema.get("paths", {})

        assert "/" in paths
        assert "/health" in paths
        assert "/analyze" in paths
        assert "post" in paths["/analyze"]
        assert "/analyses" in paths
        assert "get" in paths["/analyses"]
        assert "/analyses/{analysis_id}" in paths
        assert "get" in paths["/analyses/{analysis_id}"]

    @pytest.mark.anyio
    async def test_swagger_ui_endpoint_available(self, client):
        resp = await client.get("/docs")
        assert resp.status_code == 200
        assert "swagger-ui" in resp.text.lower()


# ── 8. Three-Class Model & Multiclass SHAP API Tests ────────────

class TestThreeClassAndSHAPIntegration:
    @pytest.mark.anyio
    async def test_three_class_legitimate_url(self, client):
        """Verify class 0 prediction, threat_category='legitimate', probabilities, and safe explanations."""
        resp = await client.post("/analyze", json={"url": "https://www.google.com"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["analysis_status"] == "completed"
        assert data["prediction"] == 0
        assert data["threat_category"] == "legitimate"
        assert isinstance(data["risk_score"], (int, float))
        assert 0.0 <= data["risk_score"] <= 0.5

        # Class probabilities validation
        probs = data["class_probabilities"]
        assert isinstance(probs, dict)
        assert "legitimate" in probs
        assert "phishing" in probs
        assert "malware_distribution" in probs
        assert abs(sum(probs.values()) - 1.0) < 0.02
        assert data["probabilities"] == probs

        # Top features & explanation validation
        top_feats = data["top_features"]
        assert isinstance(top_feats, list)
        assert len(top_feats) > 0
        for feat in top_feats:
            assert "feature_name" in feat
            assert "contribution" in feat
            assert "explanation" in feat
            # Safe phrasing for legitimate URLs: never describe as malicious
            expl = feat["explanation"].lower()
            assert "phishing" not in expl
            assert "malware" not in expl
            assert "malicious" not in expl

        assert data["explanation"] is not None
        assert data["explanation"]["status"] == "explained"
        assert data["explanation"]["target_class"] == 0

    @pytest.mark.anyio
    async def test_three_class_phishing_url(self, client):
        """Verify class 1 prediction, threat_category='phishing', and phishing indicators."""
        phish_url = "https://secure-paypal-login.account-verification.tk/login.php"
        resp = await client.post("/analyze", json={"url": phish_url})
        assert resp.status_code == 200
        data = resp.json()

        assert data["analysis_status"] == "completed"
        assert data["prediction"] == 1
        assert data["threat_category"] == "phishing"
        assert data["risk_score"] > 0.5

        probs = data["class_probabilities"]
        assert probs["phishing"] > 0.5
        assert abs(sum(probs.values()) - 1.0) < 0.02

        top_feats = data["top_features"]
        assert len(top_feats) > 0
        feat_names = [f["feature_name"] for f in top_feats]
        assert any(fn in ("has_suspicious_tld", "has_suspicious_keyword", "num_subdomains", "url_length") for fn in feat_names)

    @pytest.mark.anyio
    async def test_three_class_malware_url(self, client):
        """Verify class 2 prediction, threat_category='malware_distribution', and malware indicators."""
        malware_url = "http://192.168.1.1:8080/setup.exe"
        resp = await client.post("/analyze", json={"url": malware_url})
        assert resp.status_code == 200
        data = resp.json()

        assert data["analysis_status"] == "completed"
        assert data["prediction"] == 2
        assert data["threat_category"] == "malware_distribution"
        assert data["risk_score"] > 0.5

        probs = data["class_probabilities"]
        assert probs["malware_distribution"] > 0.5
        assert abs(sum(probs.values()) - 1.0) < 0.02

        top_feats = data["top_features"]
        assert len(top_feats) > 0
        feat_names = [f["feature_name"] for f in top_feats]
        assert any(fn in ("has_ip_address", "is_https", "num_digits", "url_entropy") for fn in feat_names)

    @pytest.mark.anyio
    async def test_incompatible_feature_count_handled_gracefully(self, client, monkeypatch, tmp_path):
        """Requirement 8: Model with 10 features instead of 20 must be handled gracefully."""
        bad_model_file = str(tmp_path / "incompatible_model.joblib")
        # Train model with 10 features
        X_bad = np.zeros((10, 10))
        y_bad = np.zeros(10, dtype=int)
        clf = RandomForestClassifier(n_estimators=2, random_state=42)
        clf.fit(X_bad, y_bad)
        joblib.dump(clf, bad_model_file)

        monkeypatch.setenv("PHISHGUARD_MODEL_PATH", bad_model_file)
        resp = await client.post("/analyze", json={"url": "https://example.com"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_status"] == "model_unavailable"
        assert data["prediction"] is None
        assert data["risk_score"] is None

    @pytest.mark.anyio
    async def test_corrupt_model_file_handled_gracefully(self, client, monkeypatch, tmp_path):
        """Requirement 8: Corrupt/non-joblib file handled gracefully."""
        corrupt_file = str(tmp_path / "corrupt.joblib")
        with open(corrupt_file, "w") as f:
            f.write("not a joblib file")

        monkeypatch.setenv("PHISHGUARD_MODEL_PATH", corrupt_file)
        resp = await client.post("/analyze", json={"url": "https://example.com"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_status"] == "model_unavailable"
        assert data["prediction"] is None

    @pytest.mark.anyio
    async def test_never_report_model_unavailable_for_valid_model(self, client):
        """Requirement 9: Valid model must never return model_unavailable."""
        resp = await client.post("/analyze", json={"url": "https://example.com"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_status"] == "completed"
        assert data["prediction"] is not None
        assert data["threat_category"] in ("legitimate", "phishing", "malware_distribution")

    @pytest.mark.anyio
    async def test_persistence_includes_threat_category(self, client):
        """Verify persisted record retrieved via GET /analyses/{id} includes threat_category."""
        resp = await client.post("/analyze", json={"url": "https://www.google.com"})
        assert resp.status_code == 200
        created = resp.json()
        analysis_id = created["id"]

        get_resp = await client.get(f"/analyses/{analysis_id}")
        assert get_resp.status_code == 200
        retrieved = get_resp.json()
        assert retrieved["id"] == analysis_id
        assert retrieved["threat_category"] == "legitimate"
        assert retrieved["prediction"] == 0


# ── 9. Database Persistence Audit Tests ─────────────────────────

class TestDatabasePersistenceAudit:
    @pytest.mark.anyio
    async def test_full_fields_survive_storage_and_retrieval(self, client):
        """Audit requirement: verify prediction, threat_category, risk_score,
        class_probabilities, probabilities, top_features, explanation, and
        analysis_status survive storage and retrieval via GET /analyses/{id}
        and GET /analyses.
        """
        target_url = "https://secure-paypal-login.account-verification.tk/login.php"
        post_resp = await client.post("/analyze", json={"url": target_url})
        assert post_resp.status_code == 200
        post_data = post_resp.json()

        analysis_id = post_data["id"]

        # Fetch via GET /analyses/{id}
        get_resp = await client.get(f"/analyses/{analysis_id}")
        assert get_resp.status_code == 200
        get_data = get_resp.json()

        # Check all required audit fields survive
        assert get_data["id"] == post_data["id"]
        assert get_data["url"] == post_data["url"]
        assert get_data["prediction"] == post_data["prediction"]
        assert get_data["threat_category"] == post_data["threat_category"]
        assert get_data["risk_score"] == post_data["risk_score"]
        assert get_data["analysis_status"] == post_data["analysis_status"]
        assert get_data["analyzed_at"] == post_data["analyzed_at"]

        # Probabilities & class_probabilities
        assert get_data["class_probabilities"] is not None
        assert get_data["class_probabilities"] == post_data["class_probabilities"]
        assert get_data["probabilities"] is not None
        assert get_data["probabilities"] == post_data["probabilities"]

        # Top features & explanations
        assert get_data["top_features"] is not None
        assert len(get_data["top_features"]) == len(post_data["top_features"])
        assert get_data["top_features"] == post_data["top_features"]

        assert get_data["explanation"] is not None
        assert get_data["explanation"] == post_data["explanation"]

        # Also verify in GET /analyses list
        list_resp = await client.get("/analyses?limit=10")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        matching = [item for item in list_data if item["id"] == analysis_id]
        assert len(matching) == 1
        item = matching[0]
        assert item["class_probabilities"] == post_data["class_probabilities"]
        assert item["probabilities"] == post_data["probabilities"]
        assert item["top_features"] == post_data["top_features"]
        assert item["explanation"] == post_data["explanation"]
        assert item["threat_category"] == post_data["threat_category"]

    @pytest.mark.anyio
    async def test_malware_fields_survive_storage_and_retrieval(self, client):
        """Verify malware_distribution predictions and explainability survive persistence."""
        malware_url = "http://192.168.1.1:8080/setup.exe"
        post_resp = await client.post("/analyze", json={"url": malware_url})
        assert post_resp.status_code == 200
        post_data = post_resp.json()

        analysis_id = post_data["id"]
        get_resp = await client.get(f"/analyses/{analysis_id}")
        assert get_resp.status_code == 200
        get_data = get_resp.json()

        assert get_data["prediction"] == 2
        assert get_data["threat_category"] == "malware_distribution"
        assert get_data["class_probabilities"]["malware_distribution"] > 0.5
        assert len(get_data["top_features"]) > 0
        assert get_data["explanation"]["status"] == "explained"

    @pytest.mark.anyio
    async def test_historical_legacy_record_compatibility(self, client, app_with_temp_db):
        """Requirement 5: Historical records with NULL threat_category,
        class_probabilities, top_features, explanation must retrieve safely
        without crashing and with backwards-compatible resolved fields.
        """
        from backend.database.session import create_db_engine
        from sqlalchemy.orm import sessionmaker
        from backend.database.models import Analysis

        engine = create_db_engine(app_with_temp_db)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        try:
            # Seed a legacy record with nulls in new columns
            legacy_record = Analysis(
                url="https://legacy-record.com",
                prediction=1,
                risk_score=0.91,
                analysis_status="completed",
                threat_category=None,
                class_probabilities=None,
                top_features=None,
                explanation=None,
            )
            db.add(legacy_record)
            db.commit()
            db.refresh(legacy_record)
            legacy_id = legacy_record.id
        finally:
            db.close()
            engine.dispose()

        # Retrieve legacy record via GET /analyses/{id}
        resp = await client.get(f"/analyses/{legacy_id}")
        assert resp.status_code == 200
        data = resp.json()

        assert data["id"] == legacy_id
        assert data["url"] == "https://legacy-record.com"
        assert data["prediction"] == 1
        # Model validator computes threat_category for legacy records
        assert data["threat_category"] == "phishing"
        assert data["risk_score"] == 0.91
        assert data["class_probabilities"] is None
        assert data["probabilities"] is None
        assert data["top_features"] is None
        assert data["explanation"] is None
        assert data["analysis_status"] == "completed"

        # Also retrieve via GET /analyses list
        list_resp = await client.get("/analyses")
        assert list_resp.status_code == 200
        matching = [item for item in list_resp.json() if item["id"] == legacy_id]
        assert len(matching) == 1
        assert matching[0]["threat_category"] == "phishing"


class TestWikipediaAndLegitimateURLRegression:
    """Regression tests for false-positive prevention on legitimate URLs with trailing slashes."""

    @pytest.mark.anyio
    async def test_wikipedia_with_trailing_slash_classified_as_legitimate(self, client):
        """Verify https://www.wikipedia.org/ is classified as legitimate with low risk."""
        resp = await client.post("/analyze", json={"url": "https://www.wikipedia.org/"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["analysis_status"] == "completed"
        assert data["prediction"] == 0
        assert data["threat_category"] == "legitimate"
        assert data["risk_score"] < 0.15  # Risk score ~5.1%
        assert data["class_probabilities"]["legitimate"] > 0.90
        assert data["class_probabilities"]["phishing"] < 0.10
        assert data["class_probabilities"]["malware_distribution"] < 0.05

        # Verify SHAP explanation is generated for legitimate class
        assert data["explanation"] is not None
        assert data["explanation"]["status"] == "explained"
        assert data["explanation"]["target_class"] == 0
        assert len(data["top_features"]) > 0

        # Features should support legitimacy
        for feat in data["top_features"]:
            expl = feat["explanation"].lower()
            assert "phishing" not in expl
            assert "malware" not in expl

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.wikipedia.org/",
            "https://www.google.com/",
            "https://www.microsoft.com/",
            "https://www.github.com/",
        ],
    )
    async def test_known_legitimate_urls_with_trailing_slashes(self, client, url):
        """Verify known benign URLs with root trailing slashes are predicted as legitimate."""
        resp = await client.post("/analyze", json={"url": url})
        assert resp.status_code == 200
        data = resp.json()

        assert data["analysis_status"] == "completed"
        assert data["prediction"] == 0
        assert data["threat_category"] == "legitimate"
        assert data["risk_score"] < 0.15
        assert data["class_probabilities"]["legitimate"] > 0.90

    @pytest.mark.anyio
    async def test_tbcpl_lol_classified_as_phishing(self, client):
        """Verify https://tbcpl.lol/ is not normalized to www.tbcpl.lol and is classified as phishing."""
        resp = await client.post("/analyze", json={"url": "https://tbcpl.lol/"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["analysis_status"] == "completed"
        assert data["prediction"] == 1
        assert data["threat_category"] == "phishing"
        assert data["risk_score"] > 0.90
        assert data["class_probabilities"]["phishing"] > 0.90
        assert data["class_probabilities"]["legitimate"] < 0.10

        # Verify normalized URL preserved domain and did not mutate to www.tbcpl.lol
        assert "www.tbcpl.lol" not in data["url"]

    @pytest.mark.anyio
    async def test_threat_urls_remain_properly_classified(self, client):
        """Ensure phishing and malware fixtures are not degraded by canonical normalization."""
        # Phishing fixture
        phish_resp = await client.post(
            "/analyze",
            json={"url": "https://secure-paypal-login.account-verification.tk/login.php"},
        )
        assert phish_resp.status_code == 200
        pdata = phish_resp.json()
        assert pdata["prediction"] == 1
        assert pdata["threat_category"] == "phishing"
        assert pdata["risk_score"] > 0.90

        # Malware fixture
        malware_resp = await client.post(
            "/analyze",
            json={"url": "http://192.168.1.1:8080/setup.exe"},
        )
        assert malware_resp.status_code == 200
        mdata = malware_resp.json()
        assert mdata["prediction"] == 2
        assert mdata["threat_category"] == "malware_distribution"
        assert mdata["risk_score"] > 0.90


# ── 15. Webpage Screenshot Capture & Visual Feature Extraction ──

class TestScreenshotAnalysisAPI:
    """Tests for user-triggered screenshot ingestion and visual feature extraction."""

    @staticmethod
    def _create_sample_png_bytes(width: int = 80, height: int = 60) -> bytes:
        """Create a synthetic PNG image with varied colors and shapes."""
        import cv2
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Fill regions with colors
        img[: height // 2, : width // 2] = [255, 0, 0]  # Blue
        img[height // 2 :, : width // 2] = [0, 255, 0]  # Green
        img[: height // 2, width // 2 :] = [0, 0, 255]  # Red
        img[height // 2 :, width // 2 :] = [255, 255, 255]  # White
        success, encoded = cv2.imencode(".png", img)
        assert success, "Failed to encode test PNG image"
        return encoded.tobytes()

    @pytest.mark.anyio
    async def test_valid_screenshot_png_payload(self, client):
        """Verify valid PNG base64 extracts all 30 visual features and preserves URL prediction."""
        import base64
        from ml.features.visual_features import VISUAL_FEATURE_NAMES

        png_bytes = self._create_sample_png_bytes(80, 60)
        b64_str = base64.b64encode(png_bytes).decode("ascii")

        resp = await client.post(
            "/analyze",
            json={
                "url": "https://www.wikipedia.org/",
                "screenshot_base64": b64_str,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Non-phishing multimodal classification (unverified candidate)
        assert data["analysis_status"] == "completed"
        assert data["prediction"] == 0
        assert data["threat_category"] == "unverified_candidate"
        assert data["risk_score"] < 0.30

        # Visual feature extraction
        assert data["visual_status"] == "extracted"
        assert isinstance(data["visual_features"], dict)
        assert len(data["visual_features"]) == len(VISUAL_FEATURE_NAMES)
        assert data["visual_features"]["width"] == 80
        assert data["visual_features"]["height"] == 60
        assert data["visual_features"]["aspect_ratio"] == round(80 / 60, 4)
        assert "mean_brightness" in data["visual_features"]
        assert "image_entropy" in data["visual_features"]
        assert "dominant_color_b" in data["visual_features"]
        assert "Visual features extracted successfully" in (data.get("visual_message") or "")

    @pytest.mark.anyio
    async def test_valid_screenshot_data_url_payload(self, client):
        """Verify data:image/png;base64,... Data URL format is accepted and processed."""
        import base64

        png_bytes = self._create_sample_png_bytes(100, 50)
        b64_str = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64_str}"

        resp = await client.post(
            "/analyze",
            json={
                "url": "https://www.google.com/",
                "screenshot_base64": data_url,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["visual_status"] == "extracted"
        assert data["visual_features"]["width"] == 100
        assert data["visual_features"]["height"] == 50

    @pytest.mark.anyio
    async def test_valid_screenshot_jpeg_data_url_payload(self, client):
        """Verify JPEG Data URL format is accepted and processed."""
        import base64
        import cv2

        img = np.ones((70, 90, 3), dtype=np.uint8) * 128
        _, encoded = cv2.imencode(".jpg", img)
        b64_str = base64.b64encode(encoded.tobytes()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64_str}"

        resp = await client.post(
            "/analyze",
            json={
                "url": "https://www.microsoft.com/",
                "screenshot_base64": data_url,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["visual_status"] == "extracted"
        assert data["visual_features"]["width"] == 90
        assert data["visual_features"]["height"] == 70

    @pytest.mark.anyio
    async def test_request_omitting_screenshot_field(self, client):
        """Verify requests without screenshot return not_provided without error."""
        resp = await client.post(
            "/analyze",
            json={"url": "https://www.wikipedia.org/"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["visual_status"] == "not_provided"
        assert data["visual_features"] is None
        assert data["visual_message"] is None
        assert data["prediction"] == 0

    @pytest.mark.anyio
    async def test_request_with_explicit_null_screenshot(self, client):
        """Verify requests with screenshot_base64: null return not_provided."""
        resp = await client.post(
            "/analyze",
            json={"url": "https://www.wikipedia.org/", "screenshot_base64": None},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["visual_status"] == "not_provided"
        assert data["visual_features"] is None

    @pytest.mark.anyio
    async def test_rejects_empty_screenshot_string(self, client):
        """Verify empty string screenshot payload returns 422."""
        resp = await client.post(
            "/analyze",
            json={"url": "https://www.wikipedia.org/", "screenshot_base64": ""},
        )
        assert resp.status_code == 422
        errors = resp.json().get("detail", [])
        assert any("empty" in str(err).lower() for err in errors)

    @pytest.mark.anyio
    async def test_rejects_whitespace_screenshot_string(self, client):
        """Verify whitespace string screenshot payload returns 422."""
        resp = await client.post(
            "/analyze",
            json={"url": "https://www.wikipedia.org/", "screenshot_base64": "   \n\t  "},
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_invalid_base64_payload(self, client):
        """Verify non-base64 characters return 422."""
        resp = await client.post(
            "/analyze",
            json={
                "url": "https://www.wikipedia.org/",
                "screenshot_base64": "Not-Valid-Base64!@#$%",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_non_image_base64_payload(self, client):
        """Verify valid base64 containing plain text (not an image) returns 422."""
        import base64

        plain_text_b64 = base64.b64encode(b"This is a text string, not an image.").decode("ascii")
        resp = await client.post(
            "/analyze",
            json={
                "url": "https://www.wikipedia.org/",
                "screenshot_base64": plain_text_b64,
            },
        )
        assert resp.status_code == 422
        errors = resp.json().get("detail", [])
        assert any("image" in str(err).lower() for err in errors)

    @pytest.mark.anyio
    async def test_rejects_non_image_data_url_mime_type(self, client):
        """Verify non-image MIME types in Data URL return 422."""
        import base64

        b64 = base64.b64encode(b"PDF dummy content").decode("ascii")
        resp = await client.post(
            "/analyze",
            json={
                "url": "https://www.wikipedia.org/",
                "screenshot_base64": f"data:application/pdf;base64,{b64}",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_data_url_without_base64_specifier(self, client):
        """Verify Data URL missing ';base64,' returns 422."""
        resp = await client.post(
            "/analyze",
            json={
                "url": "https://www.wikipedia.org/",
                "screenshot_base64": "data:image/png,plainrawcontent",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_rejects_oversized_base64_payload(self, client):
        """Verify base64 payload exceeding 15MB limit returns 422."""
        # 15 MB + 100 bytes of base64
        oversized = "A" * (15 * 1024 * 1024 + 100)
        resp = await client.post(
            "/analyze",
            json={
                "url": "https://www.wikipedia.org/",
                "screenshot_base64": oversized,
            },
        )
        assert resp.status_code == 422
        errors = resp.json().get("detail", [])
        assert any("exceeds" in str(err).lower() or "maximum" in str(err).lower() for err in errors)

    @pytest.mark.anyio
    async def test_model_unavailable_handles_screenshot_gracefully(
        self, client, monkeypatch, tmp_path
    ):
        """Verify model_unavailable mode still extracts visual features cleanly."""
        import base64

        monkeypatch.setenv("PHISHGUARD_MODEL_PATH", str(tmp_path / "nonexistent_model.joblib"))
        png_bytes = self._create_sample_png_bytes(64, 64)
        b64_str = base64.b64encode(png_bytes).decode("ascii")

        resp = await client.post(
            "/analyze",
            json={
                "url": "https://www.wikipedia.org/",
                "screenshot_base64": b64_str,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_status"] == "model_unavailable"
        assert data["prediction"] is None
        assert data["threat_category"] is None
        assert data["visual_status"] == "extracted"
        assert data["visual_features"]["width"] == 64
        assert data["visual_features"]["height"] == 64

    @pytest.mark.anyio
    async def test_screenshot_does_not_mutate_url_prediction(self, client):
        """Verify phishing fixture prediction is identical with and without screenshot."""
        import base64

        phish_url = "https://secure-paypal-login.account-verification.tk/login.php"

        # 1. Without screenshot
        resp1 = await client.post("/analyze", json={"url": phish_url})
        assert resp1.status_code == 200
        data1 = resp1.json()

        # 2. With screenshot
        png_bytes = self._create_sample_png_bytes(100, 100)
        b64_str = base64.b64encode(png_bytes).decode("ascii")
        resp2 = await client.post(
            "/analyze",
            json={"url": phish_url, "screenshot_base64": b64_str},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()

        # Both predictions must classify as phishing threat
        assert data1["prediction"] == data2["prediction"] == 1
        assert data1["threat_category"] == data2["threat_category"] == "phishing"
        assert data1["risk_score"] >= 0.85
        assert data2["risk_score"] >= 0.85
        assert data1["model_routing"] == "url_model_fallback"
        assert data2["model_routing"] == "multimodal_full"

    @pytest.mark.anyio
    async def test_post_and_subsequent_get_analyses_persists_visual_features(self, client):
        """Verify that visual_status and all 30 visual features are persisted and returned by GET /analyses."""
        import base64
        from ml.features.visual_features import VISUAL_FEATURE_NAMES

        png_bytes = self._create_sample_png_bytes(120, 80)
        b64_str = base64.b64encode(png_bytes).decode("ascii")

        # 1. POST /analyze with valid screenshot
        post_resp = await client.post(
            "/analyze",
            json={
                "url": "https://www.wikipedia.org/",
                "screenshot_base64": b64_str,
            },
        )
        assert post_resp.status_code == 200
        post_data = post_resp.json()
        analysis_id = post_data["id"]
        assert post_data["visual_status"] == "extracted"
        assert post_data["visual_features"]["width"] == 120
        assert post_data["visual_features"]["height"] == 80

        # 2. GET /analyses - verify persisted record in list
        list_resp = await client.get("/analyses?limit=10")
        assert list_resp.status_code == 200
        records = list_resp.json()
        target_rec = next((r for r in records if r["id"] == analysis_id), None)
        assert target_rec is not None, f"Record with id {analysis_id} not found in GET /analyses"
        assert target_rec["visual_status"] == "extracted"
        assert isinstance(target_rec["visual_features"], dict)
        assert len(target_rec["visual_features"]) == len(VISUAL_FEATURE_NAMES)
        assert target_rec["visual_features"]["width"] == 120
        assert target_rec["visual_features"]["height"] == 80

        # 3. GET /analyses/{id} - verify persisted record by ID
        get_resp = await client.get(f"/analyses/{analysis_id}")
        assert get_resp.status_code == 200
        single_data = get_resp.json()
        assert single_data["visual_status"] == "extracted"
        assert single_data["visual_features"]["width"] == 120
        assert single_data["visual_features"]["height"] == 80

    @pytest.mark.anyio
    async def test_get_analyses_url_only_record_has_not_provided_status(self, client):
        """Verify that analysis performed without screenshot persists visual_status as not_provided."""
        post_resp = await client.post(
            "/analyze",
            json={"url": "https://www.google.com/"},
        )
        assert post_resp.status_code == 200
        analysis_id = post_resp.json()["id"]

        get_resp = await client.get(f"/analyses/{analysis_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["visual_status"] == "not_provided"
        assert data["visual_features"] is None

    @pytest.mark.anyio
    async def test_raw_screenshot_is_never_persisted_in_db(self, client, app_with_temp_db):
        """Verify that neither base64 payload nor raw image bytes are stored in SQLite."""
        import base64
        import sqlite3

        png_bytes = self._create_sample_png_bytes(60, 60)
        unique_token = "UNIQUE_PAYLOAD_TEST_STRING_ABCD"
        b64_str = base64.b64encode(png_bytes).decode("ascii")

        post_resp = await client.post(
            "/analyze",
            json={
                "url": "https://www.microsoft.com/",
                "screenshot_base64": f"data:image/png;base64,{b64_str}",
            },
        )
        assert post_resp.status_code == 200

        # Query raw SQLite file directly
        db_file = app_with_temp_db
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis")
        rows = cursor.fetchall()
        for row in rows:
            for val in row:
                if isinstance(val, (str, bytes)):
                    val_str = str(val)
                    # Verify no raw screenshot base64 was saved
                    assert b64_str not in val_str
                    assert "data:image" not in val_str
        conn.close()





