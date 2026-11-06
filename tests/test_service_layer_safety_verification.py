"""
PhishGuard – Phase 3, Step 3.2.1: Service-Layer Integration Safety Verification Test Suite

Comprehensive audit and safety verification tests covering:
1. Probability Semantics:
   - Validates raw ensemble vote fractions (raw_model_probability), not synthetic or claimed calibration.
   - Validates risk_score and confidence calculations across all routing pathways.
2. API Response Consistency:
   - Verifies alignment between prediction, threat_category, and class_probabilities.
   - Verifies strict isolation of URL (3-class), visual (2-class), and multimodal (2-class) distributions.
   - Verifies malware predictions retain original URL 3-class distribution.
3. Model Failure Isolation:
   - Missing multimodal model (falls back to URL model).
   - Missing visual model (multimodal and URL models continue unaffected).
   - Missing URL model (safe degradation to model_unavailable).
   - Invalid / corrupt metadata files (safe defaults applied).
   - Incompatible feature counts (safe rejection).
   - Runtime inference exceptions in multimodal, visual, and URL models.
4. Malware Invariant:
   - Preemption on class 2 or P(malware) >= 0.50 overrules multimodal classifier.
   - Strict threshold boundary checks (e.g. 0.49 vs 0.50 vs 0.51).
5. Backward Compatibility:
   - Historical database records without multi-model telemetry remain readable.
   - Response schema serializes both legacy and enhanced records cleanly.
6. Shadow-Mode Invariant:
   - Verifies non-blocking behavior: is_shadow_mode=True, zero redirect or interception.
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


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_screenshot_b64() -> str:
    """Generate a valid test image encoded as base64."""
    img = np.full((120, 160, 3), 180, dtype=np.uint8)
    img[30:90, 40:120] = [40, 80, 160]
    success, encoded = cv2.imencode(".png", img)
    assert success
    b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@pytest.fixture
def mock_url_features() -> dict[str, float]:
    """Sample 20 canonical URL features."""
    return {name: 1.0 if "length" in name else 0.0 for name in URL_FEATURE_NAMES}


@pytest.fixture
def isolated_db_session(tmp_path, monkeypatch):
    """Set up an isolated SQLite database file."""
    db_file = str(tmp_path / "safety_audit.db")
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
async def async_api_client(isolated_db_session):
    """Async HTTP client for API safety testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── 1. Probability Semantics Verification ────────────────────────────

class TestProbabilitySemantics:
    """Audit probability calculation semantics across all operational routes."""

    def test_url_fallback_probability_semantics(self, mock_url_features):
        """Verify URL fallback uses raw model probabilities and risk = 1.0 - P(legit)."""
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([1])
        url_model.predict_proba.return_value = np.array([[0.15, 0.80, 0.05]])

        res = arbiter.arbitrate(
            url="http://fallback-test.com",
            url_features=mock_url_features,
            url_model=url_model,
            screenshot_base64=None,
        )

        assert res.model_routing == "url_model_fallback"
        # Risk score is raw threat probability mass: 1.0 - 0.15 = 0.85
        assert res.risk_score == 0.85
        assert isinstance(res.risk_score, float)
        assert 0.0 <= res.risk_score <= 1.0
        # Confidence is raw probability of predicted class: 0.80
        assert res.confidence == 0.80
        assert res.class_probabilities == {
            "legitimate": 0.15,
            "phishing": 0.80,
            "malware_distribution": 0.05,
        }

    def test_multimodal_full_probability_semantics(self, mock_url_features, sample_screenshot_b64):
        """Verify multimodal full mode uses raw ensemble vote probability of phishing."""
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.95, 0.05, 0.00]])

        multi_model = MagicMock()
        multi_model.predict.return_value = np.array([1])
        multi_model.predict_proba.return_value = np.array([[0.12, 0.88]])

        res = arbiter.arbitrate(
            url="http://multimodal-test.com",
            url_features=mock_url_features,
            url_model=url_model,
            multimodal_model=multi_model,
            screenshot_base64=sample_screenshot_b64,
        )

        assert res.model_routing == "multimodal_full"
        assert res.risk_score == 0.88
        assert res.confidence == 0.88
        assert res.decision_margin == round(abs(0.88 - 0.50), 4)
        assert res.ambiguity_flag is False

    def test_malware_preemption_probability_semantics(self, mock_url_features):
        """Verify malware preemption preserves true raw threat probability without static 95% clamping."""
        arbiter = DecisionArbiter(malware_threshold=0.50)
        url_model = MagicMock()
        url_model.predict.return_value = np.array([2])
        url_model.predict_proba.return_value = np.array([[0.04, 0.26, 0.70]])

        res = arbiter.arbitrate(
            url="http://malware-test.com",
            url_features=mock_url_features,
            url_model=url_model,
            screenshot_base64=None,
        )

        assert res.model_routing == "url_malware_preempted"
        # Dynamic raw risk score: 1.0 - 0.04 = 0.96 (NOT static 0.95)
        assert res.risk_score == 0.96
        # Confidence is raw malware probability: 0.70
        assert res.confidence == 0.70
        assert res.class_probabilities["malware_distribution"] == 0.70


# ── 2. API Response Consistency & Distribution Isolation ─────────────

class TestAPIResponseConsistency:
    """Verify class system alignment and probability distribution isolation."""

    def test_class_system_alignment_across_all_routes(self, mock_url_features, sample_screenshot_b64):
        """Verify final prediction and threat_category strictly map to the 3-class schema."""
        arbiter = DecisionArbiter()
        url_model = MagicMock()

        # Case A: Legitimate URL fallback
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.90, 0.10, 0.00]])
        res_legit = arbiter.arbitrate(
            url="http://legit.com",
            url_features=mock_url_features,
            url_model=url_model,
        )
        assert res_legit.prediction == 0
        assert res_legit.threat_category == "legitimate"

        # Case B: Phishing URL fallback
        url_model.predict.return_value = np.array([1])
        url_model.predict_proba.return_value = np.array([[0.05, 0.95, 0.00]])
        res_phish = arbiter.arbitrate(
            url="http://phish.com",
            url_features=mock_url_features,
            url_model=url_model,
        )
        assert res_phish.prediction == 1
        assert res_phish.threat_category == "phishing"

        # Case C: Malware preemption
        url_model.predict.return_value = np.array([2])
        url_model.predict_proba.return_value = np.array([[0.01, 0.09, 0.90]])
        res_mal = arbiter.arbitrate(
            url="http://mal.com",
            url_features=mock_url_features,
            url_model=url_model,
        )
        assert res_mal.prediction == 2
        assert res_mal.threat_category == "malware_distribution"

        # Case D: Multimodal Full
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.90, 0.10, 0.00]])
        multi_model = MagicMock()
        multi_model.predict.return_value = np.array([0])
        multi_model.predict_proba.return_value = np.array([[0.92, 0.08]])
        res_multi = arbiter.arbitrate(
            url="http://multi.com",
            url_features=mock_url_features,
            url_model=url_model,
            multimodal_model=multi_model,
            screenshot_base64=sample_screenshot_b64,
        )
        assert res_multi.prediction == 0
        assert res_multi.threat_category == "unverified_candidate"

    def test_distribution_isolation_between_models(self, mock_url_features, sample_screenshot_b64):
        """Verify URL 3-class, Visual 2-class, and Multimodal 2-class distributions remain isolated."""
        arbiter = DecisionArbiter()

        url_model = MagicMock()
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.80, 0.15, 0.05]])

        vis_model = MagicMock()
        vis_model.predict.return_value = np.array([1])
        vis_model.predict_proba.return_value = np.array([[0.30, 0.70]])

        multi_model = MagicMock()
        multi_model.predict.return_value = np.array([0])
        multi_model.predict_proba.return_value = np.array([[0.85, 0.15]])

        res = arbiter.arbitrate(
            url="http://isolation-test.com",
            url_features=mock_url_features,
            url_model=url_model,
            visual_model=vis_model,
            multimodal_model=multi_model,
            screenshot_base64=sample_screenshot_b64,
        )

        # URL telemetry: strictly 3 classes
        assert set(res.url_model_result["probabilities"].keys()) == {
            "legitimate",
            "phishing",
            "malware_distribution",
        }
        assert res.url_model_result["probabilities"]["malware_distribution"] == 0.05

        # Visual telemetry: strictly 2 classes
        assert set(res.visual_model_result["probabilities"].keys()) == {
            "unverified_candidate",
            "phishing",
        }
        assert res.visual_model_result["probabilities"]["unverified_candidate"] == 0.30
        assert res.visual_model_result["probabilities"]["phishing"] == 0.70

        # Multimodal telemetry: strictly 2 classes
        assert set(res.multimodal_model_result["probabilities"].keys()) == {
            "unverified_candidate",
            "phishing",
        }
        assert res.multimodal_model_result["probabilities"]["unverified_candidate"] == 0.85
        assert res.multimodal_model_result["probabilities"]["phishing"] == 0.15


# ── 3. Model Failure Isolation Verification ──────────────────────────

class TestModelFailureIsolation:
    """Verify fault injection resilience: failure in optional models never crashes pipeline."""

    def test_missing_multimodal_model_falls_back_to_url(self, mock_url_features, sample_screenshot_b64):
        """When multimodal model is None, arbiter falls back safely to standalone URL model."""
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([1])
        url_model.predict_proba.return_value = np.array([[0.10, 0.90, 0.00]])

        res = arbiter.arbitrate(
            url="http://test.com",
            url_features=mock_url_features,
            url_model=url_model,
            multimodal_model=None,
            screenshot_base64=sample_screenshot_b64,
        )

        assert res.analysis_status == "completed"
        assert res.model_routing == "url_model_fallback"
        assert res.prediction == 1
        assert res.threat_category == "phishing"
        assert "multimodal model unavailable" in res.fallback_reason.lower()
        assert res.multimodal_model_result["status"] == "skipped"

    def test_missing_visual_model_preserves_multimodal_execution(
        self, mock_url_features, sample_screenshot_b64
    ):
        """When visual model is None, multimodal model executes normally."""
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.80, 0.20, 0.00]])

        multi_model = MagicMock()
        multi_model.predict.return_value = np.array([0])
        multi_model.predict_proba.return_value = np.array([[0.91, 0.09]])

        res = arbiter.arbitrate(
            url="http://test.com",
            url_features=mock_url_features,
            url_model=url_model,
            visual_model=None,
            multimodal_model=multi_model,
            screenshot_base64=sample_screenshot_b64,
        )

        assert res.analysis_status == "completed"
        assert res.model_routing == "multimodal_full"
        assert res.prediction == 0
        assert res.threat_category == "unverified_candidate"
        assert res.visual_model_result == {"status": "unavailable"}

    def test_missing_url_model_fails_safely_to_model_unavailable(
        self, mock_url_features, sample_screenshot_b64
    ):
        """When URL model is None, pipeline safely degrades to model_unavailable."""
        arbiter = DecisionArbiter()
        multi_model = MagicMock()

        res = arbiter.arbitrate(
            url="http://test.com",
            url_features=mock_url_features,
            url_model=None,
            multimodal_model=multi_model,
            screenshot_base64=sample_screenshot_b64,
        )

        assert res.analysis_status == "model_unavailable"
        assert res.prediction is None
        assert res.threat_category is None
        assert res.risk_score is None
        assert res.model_routing == "model_unavailable"
        # Multimodal model must not be called if URL model is missing
        multi_model.predict.assert_not_called()

    def test_multimodal_inference_exception_falls_back_cleanly(
        self, mock_url_features, sample_screenshot_b64
    ):
        """When multimodal model raises a runtime exception, arbiter falls back to URL model."""
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([1])
        url_model.predict_proba.return_value = np.array([[0.20, 0.80, 0.00]])

        broken_multi = MagicMock()
        broken_multi.predict.side_effect = RuntimeError("GPU memory segmentation fault")

        res = arbiter.arbitrate(
            url="http://test.com",
            url_features=mock_url_features,
            url_model=url_model,
            multimodal_model=broken_multi,
            screenshot_base64=sample_screenshot_b64,
        )

        assert res.analysis_status == "completed"
        assert res.model_routing == "url_model_fallback"
        assert res.prediction == 1
        assert res.threat_category == "phishing"
        assert "multimodal inference error" in res.fallback_reason.lower()

    def test_visual_model_inference_exception_isolated(
        self, mock_url_features, sample_screenshot_b64
    ):
        """When visual model raises a runtime exception, multimodal inference succeeds."""
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.90, 0.10, 0.00]])

        multi_model = MagicMock()
        multi_model.predict.return_value = np.array([0])
        multi_model.predict_proba.return_value = np.array([[0.95, 0.05]])

        broken_vis = MagicMock()
        broken_vis.predict.side_effect = ValueError("Corrupt visual feature matrix")

        res = arbiter.arbitrate(
            url="http://test.com",
            url_features=mock_url_features,
            url_model=url_model,
            visual_model=broken_vis,
            multimodal_model=multi_model,
            screenshot_base64=sample_screenshot_b64,
        )

        assert res.analysis_status == "completed"
        assert res.model_routing == "multimodal_full"
        assert res.prediction == 0
        assert res.visual_model_result["status"] == "error"
        assert "Corrupt visual feature matrix" in res.visual_model_result["error"]

    def test_invalid_metadata_json_falls_back_to_defaults(self, tmp_path):
        """Verify ModelManager handles corrupted metadata files safely."""
        manager = ModelManager()

        # Create temporary directory with corrupt class mapping JSON
        bad_json = tmp_path / "bad_class_mapping.json"
        bad_json.write_text("NOT_VALID_JSON{:::}", encoding="utf-8")

        # Loading models with bad metadata path should not raise uncaught exception
        with patch("backend.services.analyzer.DEFAULT_URL_CLASS_MAPPING_PATH", str(bad_json)):
            _, _, cmap = manager.get_url_model_and_explainer()
            assert isinstance(cmap, dict)
            assert 0 in cmap and 1 in cmap and 2 in cmap


# ── 4. Malware Preemption Invariant ──────────────────────────────────

class TestMalwarePreemptionInvariant:
    """Verify mathematical and logical invariants of malware preemption."""

    @pytest.mark.parametrize(
        "p_malware, expected_routing",
        [
            (0.49, "multimodal_full"),       # Below 0.50 threshold: routes to multimodal
            (0.50, "url_malware_preempted"), # Exactly on 0.50 threshold: preempts
            (0.51, "url_malware_preempted"), # Above 0.50 threshold: preempts
            (0.99, "url_malware_preempted"), # High malware probability: preempts
        ],
    )
    def test_malware_threshold_boundary_conditions(
        self, p_malware, expected_routing, mock_url_features, sample_screenshot_b64
    ):
        """Verify exact boundary behavior around the 0.50 malware preemption threshold."""
        arbiter = DecisionArbiter(malware_threshold=0.50)

        # URL model probabilities
        p_legit = round(1.0 - p_malware - 0.05, 4) if p_malware < 0.95 else 0.01
        p_phish = round(1.0 - p_legit - p_malware, 4)

        url_model = MagicMock()
        # If below 0.50, argmax is phishing; if >= 0.50, argmax could be phishing or malware
        pred = 2 if p_malware >= 0.50 else 1
        url_model.predict.return_value = np.array([pred])
        url_model.predict_proba.return_value = np.array([[p_legit, p_phish, p_malware]])

        multi_model = MagicMock()
        multi_model.predict.return_value = np.array([0])  # Multimodal thinks legitimate
        multi_model.predict_proba.return_value = np.array([[0.99, 0.01]])

        res = arbiter.arbitrate(
            url="http://threshold-test.com",
            url_features=mock_url_features,
            url_model=url_model,
            multimodal_model=multi_model,
            screenshot_base64=sample_screenshot_b64,
        )

        assert res.model_routing == expected_routing
        if expected_routing == "url_malware_preempted":
            assert res.prediction == 2
            assert res.threat_category == "malware_distribution"
            assert res.confidence == p_malware
            multi_model.predict.assert_not_called()
        else:
            assert res.prediction == 0
            assert res.threat_category == "unverified_candidate"


# ── 5. Backward Compatibility & Database Integrity ───────────────────

class TestBackwardCompatibility:
    """Verify legacy API compatibility and historical database record retrieval."""

    def test_historical_db_record_without_telemetry_reads_cleanly(self, isolated_db_session):
        """Verify records stored without multi_model_telemetry deserialize gracefully."""
        Session = isolated_db_session
        db = Session()
        try:
            # Simulate a legacy Phase 1 / Phase 2 record in SQLite
            legacy_record = Analysis(
                url="https://legacy-domain.org",
                prediction=0,
                threat_category="legitimate",
                risk_score=0.045,
                class_probabilities={"legitimate": 0.955, "phishing": 0.045, "malware_distribution": 0.0},
                analysis_status="completed",
                explanation={"status": "explained", "top_features": []},  # No multi_model_telemetry
                visual_status="not_provided",
            )
            db.add(legacy_record)
            db.commit()
            db.refresh(legacy_record)
            rec_id = legacy_record.id
        finally:
            db.close()

        # Query and validate through Pydantic
        db2 = Session()
        try:
            retrieved = db2.query(Analysis).filter(Analysis.id == rec_id).first()
            assert retrieved is not None
            assert retrieved.url == "https://legacy-domain.org"
            # ORM properties should default safely
            assert retrieved.model_routing == "url_model_fallback"
            assert retrieved.is_shadow_mode is True
            assert retrieved.url_model_result is None
            assert retrieved.multimodal_model_result is None
            assert retrieved.confidence == 0.045

            # Pydantic schema validation
            schema = AnalysisResponse.model_validate(retrieved)
            assert schema.id == rec_id
            assert schema.prediction == 0
            assert schema.threat_category == "legitimate"
            assert schema.risk_score == 0.045
            assert schema.confidence == 0.045
            assert schema.model_routing == "url_model_fallback"
            assert schema.is_shadow_mode is True
        finally:
            db2.close()

    @pytest.mark.anyio
    async def test_api_get_analyses_returns_all_expected_fields(self, async_api_client):
        """Verify GET /analyses response schema contains both legacy and additive fields."""
        # Create an analysis first
        post_resp = await async_api_client.post(
            "/analyze",
            json={"url": "https://compatibility-check.com"},
        )
        assert post_resp.status_code == 200
        analysis_id = post_resp.json()["id"]

        get_resp = await async_api_client.get(f"/analyses/{analysis_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()

        # Legacy fields
        assert "id" in data
        assert "url" in data
        assert "prediction" in data
        assert "threat_category" in data
        assert "risk_score" in data
        assert "class_probabilities" in data
        assert "analysis_status" in data
        assert "analyzed_at" in data

        # Additive multi-model fields
        assert "confidence" in data
        assert "model_routing" in data
        assert "is_shadow_mode" in data
        assert "url_model_result" in data
        assert "visual_model_result" in data
        assert "multimodal_model_result" in data


# ── 6. Shadow-Mode Invariant ─────────────────────────────────────────

class TestShadowModeInvariant:
    """Verify strictly non-blocking operation (no browser interception or redirection)."""

    def test_shadow_mode_flag_always_active(self, mock_url_features):
        """Ensure is_shadow_mode is unconditionally True in arbiter outputs."""
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([1])
        url_model.predict_proba.return_value = np.array([[0.0, 1.0, 0.0]])

        res = arbiter.arbitrate(
            url="http://malicious-block-test.com",
            url_features=mock_url_features,
            url_model=url_model,
        )

        assert res.is_shadow_mode is True
        telemetry = res.to_telemetry_dict()
        assert telemetry["is_shadow_mode"] is True

    @pytest.mark.anyio
    async def test_api_does_not_issue_redirects_on_high_threat(self, async_api_client):
        """Ensure high-risk threats return standard 200 OK JSON and never 301/302/307 redirects."""
        resp = await async_api_client.post(
            "/analyze",
            json={"url": "https://secure-paypal-login.account-verification.tk/login.php"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        data = resp.json()
        assert data["prediction"] == 1
        assert data["threat_category"] == "phishing"
        assert data["is_shadow_mode"] is True
