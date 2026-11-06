"""
Unit and integration tests for Decision Arbiter and Multimodal Service Layer.

Tests:
1. Model loading, feature counts (20, 30, 51), and metadata validation.
2. Exact 51-feature ordering and fusion vector integrity.
3. Malware preemption invariant (malware is never overruled by binary multimodal model).
4. Dynamic raw risk scoring (no arbitrary 95% clamping).
5. Full multimodal inference when valid screenshot is available.
6. Graceful standalone URL model fallback when screenshot is missing or capture fails.
7. Invalid / corrupt screenshot handling without unhandled exceptions.
8. Model failure and model_unavailable safe degradation.
9. API backward compatibility and response serialization.
10. Database persistence and shadow-mode telemetry invariants.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import joblib
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sklearn.ensemble import RandomForestClassifier

from backend.database.models import Analysis
from backend.database.session import create_db_engine, get_db, init_db
from backend.main import app
from backend.schemas.analysis import AnalysisResponse
from backend.services.analyzer import (
    ModelManager,
    analyze_url,
    get_model_manager,
    get_multimodal_model_path,
    get_url_model_path,
    get_visual_model_path,
)
from backend.services.decision_arbiter import DecisionArbiter, ArbiterResult
from ml.features.feature_fusion import FUSED_FEATURE_NAMES
from ml.features.url_features import FEATURE_NAMES as URL_FEATURE_NAMES, extract_url_features
from ml.features.visual_features import VISUAL_FEATURE_NAMES

ROOT_DIR = Path(__file__).resolve().parent.parent


# ── Helpers & Fixtures ───────────────────────────────────────────────

@pytest.fixture
def test_png_base64() -> str:
    """Generate a valid, simple PNG image encoded as base64."""
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    # Add some variation so it's not completely blank
    img[20:80, 20:80] = [50, 100, 150]
    success, encoded = cv2.imencode(".png", img)
    assert success
    b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@pytest.fixture
def mock_models():
    """Create lightweight mock models for URL (3 classes), Visual (2 classes), and Multimodal (2 classes)."""
    # 1. URL Model (20 features, 3 classes: 0=legit, 1=phish, 2=malware)
    X_url = np.random.RandomState(42).randn(12, 20)
    y_url = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)
    clf_url = RandomForestClassifier(n_estimators=3, random_state=42)
    clf_url.fit(X_url, y_url)

    # 2. Visual Model (30 features, 2 classes: 0=candidate, 1=phish)
    X_vis = np.random.RandomState(42).randn(10, 30)
    y_vis = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    clf_vis = RandomForestClassifier(n_estimators=3, random_state=42)
    clf_vis.fit(X_vis, y_vis)

    # 3. Multimodal Model (51 features, 2 classes: 0=candidate, 1=phish)
    X_multi = np.random.RandomState(42).randn(10, 51)
    y_multi = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    clf_multi = RandomForestClassifier(n_estimators=3, random_state=42)
    clf_multi.fit(X_multi, y_multi)

    return {
        "url_model": clf_url,
        "visual_model": clf_vis,
        "multimodal_model": clf_multi,
    }


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Set up an isolated SQLite database file."""
    db_file = str(tmp_path / "arbiter_test.db")
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
    yield TestingSession
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture
async def async_client(isolated_db):
    """Async HTTP client for FastAPI integration testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── 1. Model Loading & Metadata Validation Tests ─────────────────────

class TestModelLoadingAndMetadata:
    def test_load_all_three_models_from_disk(self):
        """Verify that all three trained models load with expected feature dimensions."""
        manager = ModelManager()

        url_clf, explainer, url_cmap = manager.get_url_model_and_explainer()
        vis_clf, vis_cmap = manager.get_visual_model()
        multi_clf, multi_cmap = manager.get_multimodal_model()

        assert url_clf is not None, "URL model must load"
        assert vis_clf is not None, "Visual model must load"
        assert multi_clf is not None, "Multimodal model must load"

        assert getattr(url_clf, "n_features_in_", None) == 20
        assert getattr(vis_clf, "n_features_in_", None) == 30
        assert getattr(multi_clf, "n_features_in_", None) == 51

        # Verify class counts
        assert len(url_clf.classes_) == 3
        assert len(vis_clf.classes_) == 2
        assert len(multi_clf.classes_) == 2

    def test_metadata_files_dimensions_and_order(self):
        """Verify feature count and naming agreement across metadata JSON files."""
        with open(ROOT_DIR / "ml" / "models" / "feature_names.json", "r") as f:
            url_names = json.load(f)
        with open(ROOT_DIR / "ml" / "models" / "visual_feature_names.json", "r") as f:
            vis_names = json.load(f)
        with open(ROOT_DIR / "ml" / "models" / "multimodal_feature_names.json", "r") as f:
            multi_names = json.load(f)

        assert len(url_names) == 20
        assert len(vis_names) == 30
        assert len(multi_names) == 51

        assert multi_names == FUSED_FEATURE_NAMES
        assert multi_names[:20] == url_names
        assert multi_names[20] == "has_visual_features"
        assert multi_names[21:] == vis_names

    def test_missing_model_file_fails_safely(self, tmp_path):
        """Verify that a nonexistent model path returns None without raising an uncaught exception."""
        manager = ModelManager()
        missing_path = str(tmp_path / "nonexistent.joblib")

        url_clf, _, _ = manager.load_url_model(missing_path)
        assert url_clf is None

        vis_clf, _ = manager.load_visual_model(missing_path)
        assert vis_clf is None

        multi_clf, _ = manager.load_multimodal_model(missing_path)
        assert multi_clf is None

    def test_incompatible_feature_count_rejected(self, tmp_path):
        """Verify that a model with wrong feature count is safely rejected."""
        manager = ModelManager()
        bad_model_path = str(tmp_path / "bad_dim.joblib")

        # Create dummy model with 10 features instead of 20
        X = np.random.randn(6, 10)
        y = np.array([0, 0, 0, 1, 1, 1])
        clf = RandomForestClassifier(n_estimators=2).fit(X, y)
        joblib.dump(clf, bad_model_path)

        url_clf, _, _ = manager.load_url_model(bad_model_path)
        assert url_clf is None, "Model with incompatible feature count must be rejected"


# ── 2. Decision Arbiter Unit Tests ───────────────────────────────────

class TestDecisionArbiter:
    def test_malware_preemption_overrules_multimodal_prediction(self, test_png_base64):
        """CRITICAL: If URL model detects malware, multimodal model MUST NOT overrule it."""
        arbiter = DecisionArbiter(malware_threshold=0.50)

        # Mock URL model predicting malware (class 2)
        url_model = MagicMock()
        url_model.predict.return_value = np.array([2])
        url_model.predict_proba.return_value = np.array([[0.02, 0.03, 0.95]])

        # Mock Multimodal model predicting legitimate (class 0)
        multi_model = MagicMock()
        multi_model.predict.return_value = np.array([0])
        multi_model.predict_proba.return_value = np.array([[0.99, 0.01]])

        url_feats = extract_url_features("http://192.168.1.1/malware.exe")

        res = arbiter.arbitrate(
            url="http://192.168.1.1/malware.exe",
            url_features=url_feats,
            url_model=url_model,
            multimodal_model=multi_model,
            screenshot_base64=test_png_base64,
        )

        # Assertions
        assert res.prediction == 2, "Prediction must be locked to 2 (malware_distribution)"
        assert res.threat_category == "malware_distribution"
        assert res.model_routing == "url_malware_preempted"
        assert res.risk_score == 0.98  # 1.0 - P(legit) = 1.0 - 0.02 = 0.98 (raw ensemble threat score, not static 95%)
        assert res.confidence == 0.95  # P(malware)
        assert res.class_probabilities["malware_distribution"] == 0.95
        assert res.multimodal_model_result["status"] == "overridden"

    def test_malware_probability_threshold_preemption(self):
        """Verify preemption triggers if malware probability meets threshold (>= 0.50)."""
        arbiter = DecisionArbiter(malware_threshold=0.50)

        url_model = MagicMock()
        url_model.predict.return_value = np.array([1])  # Phishing argmax
        url_model.predict_proba.return_value = np.array([[0.05, 0.40, 0.55]])  # P(malware) = 0.55 >= 0.50

        url_feats = extract_url_features("http://example.com/exploit")

        res = arbiter.arbitrate(
            url="http://example.com/exploit",
            url_features=url_feats,
            url_model=url_model,
            screenshot_base64=None,
        )

        assert res.prediction == 2
        assert res.threat_category == "malware_distribution"
        assert res.model_routing == "url_malware_preempted"

    def test_screenshot_success_routes_to_multimodal_full(self, test_png_base64):
        """When valid screenshot is provided, arbiter routes to full multimodal model."""
        arbiter = DecisionArbiter()

        url_model = MagicMock()
        url_model.predict.return_value = np.array([1])  # URL model thinks phishing (e.g. false alarm on bare domain)
        url_model.predict_proba.return_value = np.array([[0.10, 0.90, 0.00]])

        multi_model = MagicMock()
        multi_model.predict.return_value = np.array([0])  # Multimodal correctly resolves as candidate/legit
        multi_model.predict_proba.return_value = np.array([[0.94, 0.06]])

        vis_model = MagicMock()
        vis_model.predict.return_value = np.array([0])
        vis_model.predict_proba.return_value = np.array([[0.90, 0.10]])

        url_feats = extract_url_features("http://example-candidate.org")

        res = arbiter.arbitrate(
            url="http://example-candidate.org",
            url_features=url_feats,
            url_model=url_model,
            visual_model=vis_model,
            multimodal_model=multi_model,
            screenshot_base64=test_png_base64,
        )

        assert res.prediction == 0
        assert res.threat_category == "unverified_candidate"
        assert res.model_routing == "multimodal_full"
        assert res.visual_status == "extracted"
        assert res.risk_score == 0.06
        assert res.class_probabilities["phishing"] == 0.06
        assert res.class_probabilities["unverified_candidate"] == 0.94
        assert res.class_probabilities["legitimate"] == 0.94
        assert res.visual_model_result["status"] == "completed"
        assert res.multimodal_model_result["status"] == "completed"

    def test_screenshot_missing_falls_back_to_url_model(self):
        """When screenshot is not provided, arbiter falls back to standalone URL model."""
        arbiter = DecisionArbiter()

        url_model = MagicMock()
        url_model.predict.return_value = np.array([1])
        url_model.predict_proba.return_value = np.array([[0.20, 0.75, 0.05]])

        multi_model = MagicMock()  # Must NOT be evaluated when screenshot is missing

        url_feats = extract_url_features("http://phishing-site.xyz/login")

        res = arbiter.arbitrate(
            url="http://phishing-site.xyz/login",
            url_features=url_feats,
            url_model=url_model,
            multimodal_model=multi_model,
            screenshot_base64=None,
        )

        assert res.prediction == 1
        assert res.threat_category == "phishing"
        assert res.model_routing == "url_model_fallback"
        assert res.visual_status == "not_provided"
        assert "not provided" in res.fallback_reason.lower()
        multi_model.predict.assert_not_called()

    def test_invalid_screenshot_falls_back_safely(self):
        """Corrupt or non-image screenshot string falls back to URL model without crashing."""
        arbiter = DecisionArbiter()

        url_model = MagicMock()
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.85, 0.15, 0.00]])

        url_feats = extract_url_features("http://example.com")

        res = arbiter.arbitrate(
            url="http://example.com",
            url_features=url_feats,
            url_model=url_model,
            screenshot_base64="not_a_valid_base64_string!@#$",
        )

        assert res.prediction == 0
        assert res.threat_category == "legitimate"
        assert res.model_routing == "url_model_fallback"
        assert res.visual_status == "invalid_image"
        assert res.risk_score == 0.15

    def test_all_models_unavailable_degrades_gracefully(self):
        """When no models are available, arbiter returns model_unavailable status."""
        arbiter = DecisionArbiter()
        url_feats = extract_url_features("http://example.com")

        res = arbiter.arbitrate(
            url="http://example.com",
            url_features=url_feats,
            url_model=None,
            multimodal_model=None,
        )

        assert res.prediction is None
        assert res.threat_category is None
        assert res.risk_score is None
        assert res.analysis_status == "model_unavailable"
        assert res.model_routing == "model_unavailable"


# ── 3. FastAPI Service & Persistence Integration Tests ───────────────

class TestServiceAndAPIIntegration:
    @pytest.mark.anyio
    async def test_api_analyze_url_only_fallback(self, async_client):
        """Verify POST /analyze without screenshot returns url_model_fallback routing."""
        resp = await async_client.post("/analyze", json={"url": "https://www.wikipedia.org"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["url"] == "https://www.wikipedia.org"
        assert data["model_routing"] == "url_model_fallback"
        assert data["visual_status"] == "not_provided"
        assert data["is_shadow_mode"] is True
        assert "class_probabilities" in data
        assert "explanation" in data
        assert data["explanation"]["multi_model_telemetry"]["model_routing"] == "url_model_fallback"

    @pytest.mark.anyio
    async def test_api_analyze_multimodal_full(self, async_client, test_png_base64):
        """Verify POST /analyze with screenshot triggers multimodal_full routing."""
        resp = await async_client.post(
            "/analyze",
            json={
                "url": "https://www.google.com",
                "screenshot_base64": test_png_base64,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["url"] == "https://www.google.com"
        assert data["model_routing"] == "multimodal_full"
        assert data["visual_status"] == "extracted"
        assert data["is_shadow_mode"] is True
        assert data["multimodal_model_result"] is not None
        assert data["multimodal_model_result"]["status"] == "completed"

    @pytest.mark.anyio
    async def test_database_persistence_and_retrieval(self, isolated_db, test_png_base64):
        """Verify that multi-model telemetry is persisted in SQLite and retrieved cleanly."""
        Session = isolated_db
        db = Session()
        try:
            record = analyze_url(
                url="https://github.com",
                db=db,
                screenshot_base64=test_png_base64,
            )
            record_id = record.id
        finally:
            db.close()

        # Query back using a new session
        db2 = Session()
        try:
            retrieved = db2.query(Analysis).filter(Analysis.id == record_id).first()
            assert retrieved is not None
            assert retrieved.url == "https://github.com"
            assert retrieved.model_routing == "multimodal_full"
            assert retrieved.is_shadow_mode is True

            # Validate serialization through Pydantic
            schema = AnalysisResponse.model_validate(retrieved)
            assert schema.model_routing == "multimodal_full"
            assert schema.is_shadow_mode is True
            assert schema.multimodal_model_result is not None
        finally:
            db2.close()

    def test_shadow_mode_invariants(self):
        """Verify shadow mode flag is always true and no blocking actions exist."""
        arbiter = DecisionArbiter()
        url_feats = extract_url_features("https://suspicious-test.com")
        res = arbiter.arbitrate(
            url="https://suspicious-test.com",
            url_features=url_feats,
            url_model=None,
        )
        assert res.is_shadow_mode is True

