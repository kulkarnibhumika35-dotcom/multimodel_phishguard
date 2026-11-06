"""
PhishGuard – Phase 3, Step 3.5: Complete System End-to-End Validation Test Suite

Validates:
- Part 1: Component Connectivity (FastAPI, 3 Models, Chrome Extension, SQLite, Dashboard)
- Part 2: End-to-End Scenarios (10 Scenarios from Specification)
- Part 3: Classification Consistency (4 Categories, Candidate Isolation, Malware Preemption, Probability Separation)
- Part 4: Database & Dashboard Parity (Persistence, Historical Records, Chart Consistency, Duplicate Prevention)
- Part 5: Shadow-Mode Security (Zero Navigation Blocking, Informational Only, Shadow Indicators)
- Part 6: Performance & Latency Tolerances
"""

import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.crud import create_analysis, get_analysis_by_id, list_analyses
from backend.database.models import Analysis
from backend.database.session import create_db_engine, get_db, init_db
from backend.main import app
from backend.services.analyzer import (
    analyze_url,
    get_model_manager,
    get_multimodal_model_path,
    get_url_model_path,
    get_visual_model_path,
)
from backend.services.decision_arbiter import DecisionArbiter
from dashboard.app import (
    _format_category,
    _format_routing_label,
    _format_visual_status_label,
    _prediction_label,
    _threat_category_label,
    compute_statistics,
    filter_records,
    prepare_daily_activity,
    prepare_four_class_distribution,
    prepare_model_disagreement_statistics,
    prepare_model_routing_distribution,
    prepare_risk_score_distribution,
    prepare_screenshot_availability,
    records_to_dataframe,
)
from ml.features.feature_fusion import FUSED_FEATURE_NAMES, fuse_features
from ml.features.url_features import FEATURE_NAMES as URL_FEATURE_NAMES, extract_url_features
from ml.features.visual_features import VISUAL_FEATURE_NAMES, extract_visual_features


# ── Shared Helpers & Fixtures ────────────────────────────────────────

def create_valid_test_screenshot_b64(width=320, height=240, color=(200, 100, 50)) -> str:
    """Generate a clean synthetic screenshot with identifiable features."""
    img = np.full((height, width, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (width - 20, 60), color, -1)
    cv2.putText(img, "Login Portal", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(img, (40, 80), (width - 40, 140), (255, 255, 255), -1)
    cv2.rectangle(img, (40, 80), (width - 40, 140), (180, 180, 180), 1)
    success, buf = cv2.imencode(".png", img)
    assert success
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@pytest.fixture
def test_db_session(tmp_path):
    """Provides an isolated SQLite database session and TestClient."""
    db_file = str(tmp_path / "e2e_validation.db")
    engine = create_db_engine(db_file)
    init_db(engine_override=engine)

    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    yield {
        "engine": engine,
        "session_factory": TestingSession,
        "client": client,
        "db_file": db_file,
    }

    app.dependency_overrides.clear()
    engine.dispose()


# ── PART 1: COMPONENT CONNECTIVITY ───────────────────────────────────

class TestPart1ComponentConnectivity:
    """Verify all components connect and interact without error."""

    def test_fastapi_starts_and_health_check(self, test_db_session):
        client = test_db_session["client"]
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data.get("status") in ("online", "ok") or "PhishGuard" in data.get("name", "")

    def test_all_three_ml_models_load_correctly(self):
        mgr = get_model_manager()
        url_model, explainer, url_mapping = mgr.get_url_model_and_explainer()
        visual_model, vis_mapping = mgr.get_visual_model()
        multi_model, multi_mapping = mgr.get_multimodal_model()

        assert url_model is not None, "URL model failed to load"
        assert visual_model is not None, "Visual model failed to load"
        assert multi_model is not None, "Multimodal model failed to load"

        # Feature count checks
        assert getattr(url_model, "n_features_in_", 20) == len(URL_FEATURE_NAMES)
        assert getattr(visual_model, "n_features_in_", 30) == len(VISUAL_FEATURE_NAMES)
        assert getattr(multi_model, "n_features_in_", 51) == len(FUSED_FEATURE_NAMES)

        # Class mapping semantics
        assert url_mapping == {0: "legitimate", 1: "phishing", 2: "malware_distribution"}
        assert vis_mapping == {0: "unverified_candidate", 1: "phishing"}
        assert multi_mapping == {0: "unverified_candidate", 1: "phishing"}

    def test_extension_api_contract(self, test_db_session):
        client = test_db_session["client"]
        payload = {"url": "https://www.google.com"}
        res = client.post("/analyze", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "id" in data
        assert "url" in data
        assert "prediction" in data
        assert "threat_category" in data
        assert "risk_score" in data
        assert "class_probabilities" in data
        assert "is_shadow_mode" in data
        assert data["is_shadow_mode"] is True

    def test_screenshot_failures_trigger_url_fallback(self, test_db_session):
        client = test_db_session["client"]
        # Explicit None screenshot
        res = client.post("/analyze", json={"url": "https://en.wikipedia.org", "screenshot_base64": None})
        assert res.status_code == 200
        data = res.json()
        assert data["model_routing"] == "url_model_fallback"
        assert data["visual_status"] == "not_provided"

    def test_records_saved_in_sqlite_and_retrieved_by_dashboard(self, test_db_session):
        client = test_db_session["client"]
        res = client.post("/analyze", json={"url": "https://python.org"})
        assert res.status_code == 200
        record_id = res.json()["id"]

        # Fetch directly from API listing
        list_res = client.get("/analyses")
        assert list_res.status_code == 200
        records = list_res.json()
        assert any(r["id"] == record_id for r in records)

        # Dashboard retrieval and processing
        stats = compute_statistics(records)
        assert stats["total"] >= 1
        df = records_to_dataframe(records)
        assert not df.empty
        assert "Model Routing" in df.columns


# ── PART 2: THE 10 END-TO-END SCENARIOS ───────────────────────────────

class TestPart2EndToEndScenarios:
    """Rigorous verification of the 10 core operational scenarios."""

    def test_scenario_1_legitimate_url_through_url_fallback(self, test_db_session):
        client = test_db_session["client"]
        url = "https://www.wikipedia.org"
        res = client.post("/analyze", json={"url": url, "screenshot_base64": None})
        assert res.status_code == 200
        data = res.json()
        assert data["threat_category"] == "legitimate"
        assert data["prediction"] == 0
        assert data["model_routing"] == "url_model_fallback"
        assert data["visual_status"] == "not_provided"
        assert "legitimate" in data["class_probabilities"]
        assert "malware_distribution" in data["class_probabilities"]

    def test_scenario_2_unverified_candidate_through_multimodal(self, test_db_session):
        client = test_db_session["client"]
        url = "https://www.wikipedia.org"
        screenshot = create_valid_test_screenshot_b64(300, 200, (180, 180, 180))
        res = client.post("/analyze", json={"url": url, "screenshot_base64": screenshot})
        assert res.status_code == 200
        data = res.json()
        # Benign site with screenshot must be routed to multimodal
        assert data["model_routing"] == "multimodal_full"
        assert data["visual_status"] == "extracted"
        assert data["threat_category"] == "unverified_candidate"
        assert data["prediction"] == 0
        # Check probability structure
        assert "unverified_candidate" in data["class_probabilities"]
        assert "phishing" in data["class_probabilities"]

    def test_scenario_3_phishing_url_with_valid_screenshot(self, test_db_session):
        client = test_db_session["client"]
        url = "http://secure-paypal-login-credential-verification.tk/signin.php?account=update"
        screenshot = create_valid_test_screenshot_b64(300, 200, (20, 50, 180))
        res = client.post("/analyze", json={"url": url, "screenshot_base64": screenshot})
        assert res.status_code == 200
        data = res.json()
        assert data["threat_category"] == "phishing"
        assert data["prediction"] == 1
        assert data["risk_score"] > 0.50
        assert data["model_routing"] == "multimodal_full"
        assert data["visual_status"] == "extracted"

    def test_scenario_4_malware_url_triggering_preemption(self, test_db_session):
        client = test_db_session["client"]
        url = "http://192.168.1.100:8080/setup.exe"
        screenshot = create_valid_test_screenshot_b64(300, 200, (100, 100, 100))
        res = client.post("/analyze", json={"url": url, "screenshot_base64": screenshot})
        assert res.status_code == 200
        data = res.json()
        # Malware preemption invariant
        assert data["threat_category"] == "malware_distribution"
        assert data["prediction"] == 2
        assert data["model_routing"] == "url_malware_preempted"
        assert data["class_probabilities"]["malware_distribution"] >= 0.50
        # Multimodal was overridden
        telem = data["explanation"]["multi_model_telemetry"]
        assert telem["multimodal_model_result"]["status"] == "overridden"
        assert "preempted by url malware detection" in telem["multimodal_model_result"]["reason"].lower()

    def test_scenario_5_restricted_chrome_page(self, test_db_session):
        client = test_db_session["client"]
        # Client validation / backend rejection of chrome:// URLs
        res = client.post("/analyze", json={"url": "chrome://settings"})
        assert res.status_code in (400, 422)

    def test_scenario_6_screenshot_capture_failure(self, test_db_session):
        client = test_db_session["client"]
        url = "https://www.github.com"
        # Extension sends null / omits screenshot when captureVisibleTab fails
        res = client.post("/analyze", json={"url": url, "screenshot_base64": None})
        assert res.status_code == 200
        data = res.json()
        assert data["model_routing"] == "url_model_fallback"
        assert data["visual_status"] == "not_provided"

    def test_scenario_7_invalid_screenshot_payload(self, test_db_session):
        client = test_db_session["client"]
        url = "https://www.github.com"
        # A) API layer validates schema and returns 422 on corrupted screenshot data
        corrupt_screenshot = "data:image/png;base64,VGhpcyBpcyBub3QgYSB2YWxpZCBpbWFnZSE="
        res = client.post("/analyze", json={"url": url, "screenshot_base64": corrupt_screenshot})
        assert res.status_code == 422

        # B) Service layer DecisionArbiter catches corrupt input and gracefully falls back
        arbiter = DecisionArbiter()
        url_feats = extract_url_features(url)
        mgr = get_model_manager()
        url_model, _, _ = mgr.get_url_model_and_explainer()
        multi_model, _ = mgr.get_multimodal_model()
        arb_res = arbiter.arbitrate(
            url=url,
            url_features=url_feats,
            url_model=url_model,
            multimodal_model=multi_model,
            screenshot_base64=corrupt_screenshot,
        )
        assert arb_res.model_routing == "url_model_fallback"
        assert arb_res.visual_status in ("invalid_image", "extraction_failed")
        assert arb_res.prediction is not None

    def test_scenario_8_backend_unavailable(self):
        # Simulating dashboard / extension behavior when server is offline
        from dashboard.app import analyze_new_url
        with patch("requests.post", side_effect=Exception("Connection refused")):
            record, message = analyze_new_url("https://www.test.com")
            assert record is None
            assert message is not None
            assert "Cannot connect" in message or "Unexpected error" in message

    def test_scenario_9_multimodal_model_unavailable_fallback(self, test_db_session):
        client = test_db_session["client"]
        url = "https://www.wikipedia.org"
        screenshot = create_valid_test_screenshot_b64(200, 150)
        # Patch multimodal model to None
        with patch("backend.services.analyzer.get_multimodal_model_path", return_value="nonexistent_path.joblib"):
            mgr = get_model_manager()
            mgr._multimodal_model = None
            mgr._multimodal_loaded_path = None
            res = client.post("/analyze", json={"url": url, "screenshot_base64": screenshot})
            assert res.status_code == 200
            data = res.json()
            assert data["model_routing"] == "url_model_fallback"
            assert data["threat_category"] == "legitimate"
            assert data["prediction"] == 0

    def test_scenario_10_historical_database_record_without_telemetry(self, test_db_session):
        # Insert a raw historical record mimicking Phase 1 schema
        SessionLocal = test_db_session["session_factory"]
        with SessionLocal() as db:
            legacy = Analysis(
                url="https://legacy-site.org",
                prediction=0,
                threat_category=None,
                risk_score=0.08,
                confidence=0.92,
                class_probabilities={"legitimate": 0.92, "phishing": 0.08, "malware_distribution": 0.0},
                model_routing=None,
                visual_status=None,
                is_shadow_mode=True,
                analysis_status="completed",
            )
            db.add(legacy)
            db.commit()
            db.refresh(legacy)
            legacy_dict = {
                "id": legacy.id,
                "url": legacy.url,
                "prediction": legacy.prediction,
                "threat_category": legacy.threat_category,
                "risk_score": legacy.risk_score,
                "analysis_status": legacy.analysis_status,
                "model_routing": legacy.model_routing,
                "visual_status": legacy.visual_status,
                "is_shadow_mode": legacy.is_shadow_mode,
            }

        # Dashboard functions must process legacy records smoothly
        stats = compute_statistics([legacy_dict])
        assert stats["legitimate"] == 1
        assert stats["unverified_candidate"] == 0
        assert stats["total"] == 1

        label = _prediction_label(legacy_dict["prediction"], threat_category=legacy_dict["threat_category"])
        assert "Legitimate" in label

        df = records_to_dataframe([legacy_dict])
        assert not df.empty
        assert df.iloc[0]["Threat Category"] == "Legitimate"


# ── PART 3: CLASSIFICATION CONSISTENCY ────────────────────────────────

class TestPart3ClassificationConsistency:
    """Verify isolation of 4 classes and cross-component consistency."""

    def test_four_threat_categories_distinct(self):
        categories = ["legitimate", "unverified_candidate", "phishing", "malware_distribution"]
        labels = [_threat_category_label(cat) for cat in categories]
        assert len(set(labels)) == 4
        assert labels[0] == "Legitimate"
        assert labels[1] == "Unverified Candidate"
        assert labels[2] == "Phishing"
        assert labels[3] == "Malware Distribution"

        # Check badges with emoji
        pred_badges = [
            _prediction_label(0, threat_category="legitimate"),
            _prediction_label(0, threat_category="unverified_candidate"),
            _prediction_label(1, threat_category="phishing"),
            _prediction_label(2, threat_category="malware_distribution"),
        ]
        assert "✅ Legitimate" in pred_badges[0]
        assert "🔍 Unverified Candidate" in pred_badges[1]
        assert "🚨 Phishing" in pred_badges[2]
        assert "☣️ Malware Distribution" in pred_badges[3]

    def test_unverified_candidate_never_mapped_to_legitimate(self):
        candidate_rec = {
            "prediction": 0,
            "threat_category": "unverified_candidate",
            "model_routing": "multimodal_full",
            "analysis_status": "completed",
        }
        stats = compute_statistics([candidate_rec])
        assert stats["unverified_candidate"] == 1
        assert stats["legitimate"] == 0

        formatted = _format_category(candidate_rec)
        assert formatted == "Unverified Candidate"
        assert formatted != "Legitimate"

    def test_malware_cannot_be_overridden_by_multimodal(self):
        arbiter = DecisionArbiter(malware_threshold=0.50)
        url_feats = {col: 1.0 if "ip" in col else 0.0 for col in URL_FEATURE_NAMES}
        
        # URL model predicts malware
        mock_url_model = MagicMock()
        mock_url_model.predict_proba.return_value = np.array([[0.01, 0.04, 0.95]])
        
        # Multimodal model predicts candidate (class 0)
        mock_multi_model = MagicMock()
        mock_multi_model.predict_proba.return_value = np.array([[0.99, 0.01]])

        vis_feats = {col: 0.5 for col in VISUAL_FEATURE_NAMES}
        result = arbiter.arbitrate(
            url="http://malware-drop.org/virus.exe",
            url_features=url_feats,
            url_model=mock_url_model,
            multimodal_model=mock_multi_model,
            pre_extracted_visual_features=vis_feats,
        )
        assert result.threat_category == "malware_distribution"
        assert result.prediction == 2
        assert result.model_routing == "url_malware_preempted"

    def test_probability_distributions_remain_separate(self, test_db_session):
        client = test_db_session["client"]
        screenshot = create_valid_test_screenshot_b64(200, 150)
        res = client.post("/analyze", json={"url": "https://www.paypal.com", "screenshot_base64": screenshot})
        assert res.status_code == 200
        data = res.json()
        telem = data["explanation"]["multi_model_telemetry"]

        # URL model has 3 classes
        url_res = telem["url_model_result"]
        url_probs = url_res.get("probabilities") or url_res.get("class_probabilities")
        assert len(url_probs) == 3
        assert "malware_distribution" in url_probs

        # Multimodal model has 2 classes
        multi_res = telem["multimodal_model_result"]
        multi_probs = multi_res.get("probabilities") or multi_res.get("class_probabilities")
        assert len(multi_probs) == 2
        assert "unverified_candidate" in multi_probs
        assert "malware_distribution" not in multi_probs


# ── PART 4: DATABASE AND DASHBOARD PARITY ─────────────────────────────

class TestPart4DatabaseAndDashboardParity:
    """Verify persistence, historical read, and metric parity."""

    def test_every_completed_analysis_persisted(self, test_db_session):
        client = test_db_session["client"]
        res = client.post("/analyze", json={"url": "https://test-persist.org"})
        assert res.status_code == 200
        created_id = res.json()["id"]

        SessionLocal = test_db_session["session_factory"]
        with SessionLocal() as db:
            row = get_analysis_by_id(db, created_id)
            assert row is not None
            assert row.url == "https://test-persist.org"
            assert row.analysis_status == "completed"

    def test_dashboard_statistics_match_database_records(self, test_db_session):
        client = test_db_session["client"]
        # Generate 3 distinct records
        client.post("/analyze", json={"url": "https://www.google.com"}) # Legit URL fallback
        client.post("/analyze", json={"url": "http://192.168.1.1/setup.exe"}) # Malware
        screenshot = create_valid_test_screenshot_b64(200, 150)
        client.post("/analyze", json={"url": "https://www.python.org", "screenshot_base64": screenshot}) # Candidate

        list_res = client.get("/analyses")
        records = list_res.json()
        assert len(records) == 3

        stats = compute_statistics(records)
        assert stats["total"] == 3
        assert stats["legitimate"] == 1
        assert stats["malware"] == 1
        assert stats["unverified_candidate"] == 1
        assert stats["phishing"] == 0

        # Chart counts
        chart_df = prepare_four_class_distribution(records)
        c_map = dict(zip(chart_df["Category"], chart_df["Count"]))
        assert c_map["Legitimate"] == 1
        assert c_map["Malware Distribution"] == 1
        assert c_map["Unverified Candidate"] == 1
        assert c_map["Phishing"] == 0

    def test_model_unavailable_does_not_corrupt_kpis(self):
        records = [
            {"analysis_status": "model_unavailable", "risk_score": None, "prediction": None},
            {"analysis_status": "completed", "risk_score": 0.20, "prediction": 0, "threat_category": "legitimate"},
        ]
        stats = compute_statistics(records)
        assert stats["total"] == 2
        assert stats["unavailable"] == 1
        assert stats["legitimate"] == 1
        # Average risk score should only include completed records with valid scores
        assert stats["avg_risk_score"] == 0.20

    def test_no_duplicate_records_on_single_analysis(self, test_db_session):
        client = test_db_session["client"]
        client.post("/analyze", json={"url": "https://unique-domain.com"})
        
        SessionLocal = test_db_session["session_factory"]
        with SessionLocal() as db:
            rows = db.query(Analysis).filter(Analysis.url == "https://unique-domain.com").all()
            assert len(rows) == 1


# ── PART 5: SHADOW-MODE SECURITY ──────────────────────────────────────

class TestPart5ShadowModeSecurity:
    """Verify absolute non-blocking, non-intercepting guarantees."""

    def test_manifest_has_no_blocking_permissions(self):
        manifest_path = Path("extension/manifest.json")
        assert manifest_path.is_file()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        permissions = manifest.get("permissions", [])
        assert "webRequestBlocking" not in permissions
        assert "declarativeNetRequest" not in permissions
        assert "declarativeNetRequestWithHostAccess" not in permissions

    def test_extension_scripts_contain_no_navigation_cancellation(self):
        for script_file in [Path("extension/popup.js"), Path("extension/background.js")]:
            if not script_file.is_file():
                continue
            content = script_file.read_text(encoding="utf-8")
            # Must not cancel or redirect requests
            assert "cancel: true" not in content
            assert "cancel:true" not in content
            assert "redirectUrl" not in content
            assert "tabs.remove" not in content

    def test_shadow_mode_indicators_in_api_and_dashboard(self, test_db_session):
        client = test_db_session["client"]
        res = client.post("/analyze", json={"url": "https://example.com"})
        data = res.json()
        assert data["is_shadow_mode"] is True

        # Check dashboard record table includes Shadow Mode
        df = records_to_dataframe([data])
        assert "Shadow Mode" in df.columns
        assert df.iloc[0]["Shadow Mode"] == "Active 🛡️"


# ── PART 6: PERFORMANCE & LATENCY TOLERANCES ─────────────────────────

class TestPart6PerformanceAndReliability:
    """Ensure subsystem response times are within expected operational bounds."""

    def test_feature_extraction_and_inference_latencies(self):
        mgr = get_model_manager()
        url_model, _, _ = mgr.get_url_model_and_explainer()
        multi_model, _ = mgr.get_multimodal_model()

        # URL extraction
        t0 = time.perf_counter()
        url_feats = extract_url_features("https://www.paypal.com/signin")
        t_url_feat = (time.perf_counter() - t0) * 1000.0
        assert t_url_feat < 100.0, f"URL feature extraction too slow: {t_url_feat:.2f}ms"

        # URL RF inference
        X_url = np.array([[url_feats[c] for c in URL_FEATURE_NAMES]])
        t0 = time.perf_counter()
        _ = url_model.predict_proba(X_url)
        t_url_inf = (time.perf_counter() - t0) * 1000.0
        assert t_url_inf < 150.0, f"URL RF inference too slow: {t_url_inf:.2f}ms"

        # Visual extraction
        test_img = np.full((240, 320, 3), 220, dtype=np.uint8)
        t0 = time.perf_counter()
        vis_feats = extract_visual_features(test_img)
        t_vis_feat = (time.perf_counter() - t0) * 1000.0
        assert t_vis_feat < 250.0, f"Visual feature extraction too slow: {t_vis_feat:.2f}ms"

        # Multimodal RF inference
        fused = fuse_features(url_feats, vis_feats)
        X_multi = np.array([[fused[c] for c in FUSED_FEATURE_NAMES]])
        t0 = time.perf_counter()
        _ = multi_model.predict_proba(X_multi)
        t_multi_inf = (time.perf_counter() - t0) * 1000.0
        assert t_multi_inf < 150.0, f"Multimodal RF inference too slow: {t_multi_inf:.2f}ms"

