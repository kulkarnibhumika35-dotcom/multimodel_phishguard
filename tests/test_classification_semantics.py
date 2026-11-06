"""
PhishGuard – Phase 3, Step 3.3.1: Classification Semantics & Popup Safety Regression Tests

Validates:
1. Multimodal class 0 strictly maps to 'unverified_candidate' (never 'legitimate').
2. Multimodal class 1 strictly maps to 'phishing'.
3. URL model class 0 strictly maps to 'legitimate' (verified URL baseline).
4. URL model class 1 strictly maps to 'phishing'.
5. URL model class 2 strictly maps to 'malware_distribution' (and preempts multimodal).
6. Screenshot failure fallback routes cleanly to standalone URL model (preserving 3 classes).
7. Probability and label consistency across backend and Chrome extension DOM rendering.
8. Unverified candidate is never presented as confirmed safe in UI or text explanations.
"""

import base64
import json
import os
import subprocess
from typing import Any
from unittest.mock import MagicMock
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.orm import sessionmaker

from backend.database.session import create_db_engine, get_db, init_db
from backend.main import app
from backend.services.decision_arbiter import DecisionArbiter
from ml.features.url_features import extract_url_features
from ml.features.feature_fusion import FUSED_FEATURE_NAMES
from ml.features.visual_features import VISUAL_FEATURE_NAMES

EXTENSION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extension"))
POPUP_JS_PATH = os.path.join(EXTENSION_DIR, "popup.js").replace("\\", "/")


def render_in_node(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute popup.renderResult in Node.js with mock DOM and return DOM state."""
    js_code = f"""
    const popup = require('{POPUP_JS_PATH}');

    function createMockElement(id, initialClasses = []) {{
      const classList = new Set(initialClasses);
      return {{
        id,
        textContent: '',
        innerHTML: '',
        style: {{}},
        classList: {{
          add: (...cls) => cls.forEach(c => classList.add(c)),
          remove: (...cls) => cls.forEach(c => classList.delete(c)),
          contains: c => classList.has(c),
          get list() {{ return Array.from(classList); }}
        }},
        appendChild(child) {{
          if (!this.children) this.children = [];
          this.children.push(child);
        }}
      }};
    }}

    global.document = {{
      createElement: (tag) => createMockElement(tag, []),
      getElementById: (id) => createMockElement(id, [])
    }};

    const els = {{
      urlBox: createMockElement('current-url'),
      analyzedUrlWrapper: createMockElement('analyzed-url-wrapper', ['hidden']),
      analyzedUrlEl: createMockElement('analyzed-url'),
      threatCategoryEl: createMockElement('threat-category'),
      predictionEl: createMockElement('prediction'),
      riskScoreEl: createMockElement('risk-score'),
      statusEl: createMockElement('status'),
      modelRoutingEl: createMockElement('model-routing'),
      visualStatusEl: createMockElement('visual-status'),
      infoNotice: createMockElement('info-notice', ['hidden']),
      shadowAdvisory: createMockElement('shadow-advisory'),
      shield: createMockElement('brand-shield'),
      resultSec: createMockElement('result-section', ['hidden']),
      probEls: {{
        section: createMockElement('probabilities-section', ['hidden']),
        legitLabel: createMockElement('prob-legit-label'),
        legitBar: createMockElement('prob-legit-bar'),
        legitVal: createMockElement('prob-legit-val'),
        phishBar: createMockElement('prob-phish-bar'),
        phishVal: createMockElement('prob-phish-val'),
        malwareBar: createMockElement('prob-malware-bar'),
        malwareVal: createMockElement('prob-malware-val')
      }},
      featEls: {{
        section: createMockElement('features-section', ['hidden']),
        list: createMockElement('features-list')
      }},
      multiModelDetails: createMockElement('multi-model-details', ['hidden']),
      subUrlStatus: createMockElement('sub-url-status'),
      subUrlPred: createMockElement('sub-url-pred'),
      subUrlProbs: createMockElement('sub-url-probs'),
      subMultiStatus: createMockElement('sub-multi-status'),
      subMultiPred: createMockElement('sub-multi-pred'),
      subMultiProbs: createMockElement('sub-multi-probs'),
      subVisStatus: createMockElement('sub-vis-status'),
      subVisPred: createMockElement('sub-vis-pred'),
      subVisProbs: createMockElement('sub-vis-probs')
    }};

    const data = {json.dumps(payload)};
    popup.renderResult(data, els);

    const result = {{
      threatCategoryText: els.threatCategoryEl.textContent,
      threatCategoryClasses: els.threatCategoryEl.classList.list,
      statusText: els.statusEl.textContent,
      statusClasses: els.statusEl.classList.list,
      riskScoreText: els.riskScoreEl.textContent,
      modelRoutingText: els.modelRoutingEl ? els.modelRoutingEl.textContent : '',
      visualStatusText: els.visualStatusEl ? els.visualStatusEl.textContent : '',
      probLegitLabelText: els.probEls.legitLabel ? els.probEls.legitLabel.textContent : '',
      probLegitBarClasses: els.probEls.legitBar ? els.probEls.legitBar.classList.list : [],
      shieldClasses: els.shield.classList.list,
      noticeText: els.infoNotice.textContent,
      noticeClasses: els.infoNotice.classList.list,
      subUrlPredText: els.subUrlPred ? els.subUrlPred.textContent : '',
      subMultiPredText: els.subMultiPred ? els.subMultiPred.textContent : '',
      subVisPredText: els.subVisPred ? els.subVisPred.textContent : '',
      features: (els.featEls.list.children || []).map(c => c.innerHTML || '')
    }};

    console.log(JSON.stringify(result));
    """
    res = subprocess.run(["node", "-e", js_code], capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip())


@pytest.fixture
def sample_png_b64():
    """Valid small PNG base64 Data URL."""
    import cv2
    img = np.full((64, 64, 3), 180, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    return f"data:image/png;base64,{base64.b64encode(encoded.tobytes()).decode('utf-8')}"


@pytest.fixture
def mock_url_features():
    return extract_url_features("https://example-test-site.org/page")


# ── Scenario 1: Multimodal Class 0 (Unverified Candidate) ──────────

class TestScenario1MultimodalClass0Candidate:
    """Multimodal model predicting class 0 MUST be classified as unverified_candidate."""

    def test_arbiter_outputs_unverified_candidate_for_multi_class_0(
        self, mock_url_features, sample_png_b64
    ):
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.85, 0.12, 0.03]])

        multi_model = MagicMock()
        multi_model.predict.return_value = np.array([0])
        multi_model.predict_proba.return_value = np.array([[0.93, 0.07]])

        res = arbiter.arbitrate(
            url="https://example-test-site.org/page",
            url_features=mock_url_features,
            url_model=url_model,
            multimodal_model=multi_model,
            screenshot_base64=sample_png_b64,
        )

        # Invariants
        assert res.model_routing == "multimodal_full"
        assert res.prediction == 0
        assert res.threat_category == "unverified_candidate"
        assert res.threat_category != "legitimate"  # Never map to legitimate!
        assert res.class_probabilities["unverified_candidate"] == 0.93
        assert res.class_probabilities["phishing"] == 0.07
        assert res.class_probabilities["legitimate"] == 0.93  # Additive alias preserved

    def test_extension_renders_candidate_semantics_not_safe(self):
        backend_data = {
            "url": "https://example-test-site.org/page",
            "prediction": 0,
            "threat_category": "unverified_candidate",
            "risk_score": 0.07,
            "confidence": 0.93,
            "analysis_status": "completed",
            "model_routing": "multimodal_full",
            "visual_status": "extracted",
            "is_shadow_mode": True,
            "class_probabilities": {
                "unverified_candidate": 0.93,
                "legitimate": 0.93,
                "phishing": 0.07,
                "malware_distribution": 0.0,
            },
            "top_features": [
                {
                    "feature_name": "is_https",
                    "feature_value": 1.0,
                    "contribution": 0.15,
                    "explanation": "Valid HTTPS connection detected.",
                }
            ],
            "multimodal_model_result": {
                "status": "completed",
                "prediction": 0,
                "threat_category": "unverified_candidate",
                "probabilities": {"unverified_candidate": 0.93, "phishing": 0.07},
            },
        }

        dom = render_in_node(backend_data)

        # Must display "Unverified Candidate", status "Candidate"
        assert dom["threatCategoryText"] == "Unverified Candidate"
        assert dom["statusText"] == "Candidate"
        assert "status-candidate" in dom["threatCategoryClasses"]
        assert "status-candidate" in dom["statusClasses"]
        assert "shield-candidate" in dom["shieldClasses"]
        assert "notice-candidate" in dom["noticeClasses"]

        # MUST NOT contain safe/green styling
        assert "status-safe" not in dom["threatCategoryClasses"]
        assert "status-safe" not in dom["statusClasses"]
        assert "shield-safe" not in dom["shieldClasses"]
        assert "notice-safe" not in dom["noticeClasses"]

        # Disclaimer must state candidate status and not verified safe
        assert "NOT been independently verified as safe" in dom["noticeText"]
        assert "Unverified Candidate" in dom["noticeText"]

        # Probabilities bar label
        assert dom["probLegitLabelText"] == "Candidate"

        # Diagnostic sub-model display
        assert dom["subMultiPredText"] == "Unverified Candidate"


# ── Scenario 2: Multimodal Class 1 (Phishing) ──────────────────────

class TestScenario2MultimodalClass1Phishing:
    """Multimodal model predicting class 1 MUST be classified as phishing."""

    def test_arbiter_outputs_phishing_for_multi_class_1(
        self, mock_url_features, sample_png_b64
    ):
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.70, 0.28, 0.02]])

        multi_model = MagicMock()
        multi_model.predict.return_value = np.array([1])
        multi_model.predict_proba.return_value = np.array([[0.15, 0.85]])

        res = arbiter.arbitrate(
            url="https://example-test-site.org/page",
            url_features=mock_url_features,
            url_model=url_model,
            multimodal_model=multi_model,
            screenshot_base64=sample_png_b64,
        )

        assert res.model_routing == "multimodal_full"
        assert res.prediction == 1
        assert res.threat_category == "phishing"
        assert res.risk_score == 0.85
        assert res.class_probabilities["phishing"] == 0.85

    def test_extension_renders_phishing_danger_for_multi_class_1(self):
        backend_data = {
            "url": "https://example-phish.net/login",
            "prediction": 1,
            "threat_category": "phishing",
            "risk_score": 0.85,
            "confidence": 0.85,
            "analysis_status": "completed",
            "model_routing": "multimodal_full",
            "visual_status": "extracted",
            "is_shadow_mode": True,
            "class_probabilities": {
                "unverified_candidate": 0.15,
                "legitimate": 0.15,
                "phishing": 0.85,
                "malware_distribution": 0.0,
            },
        }

        dom = render_in_node(backend_data)
        assert dom["threatCategoryText"] == "Phishing"
        assert dom["statusText"] == "Danger"
        assert "status-danger" in dom["threatCategoryClasses"]
        assert "shield-danger" in dom["shieldClasses"]
        assert "notice-danger" in dom["noticeClasses"]
        assert "High phishing threat detected" in dom["noticeText"]


# ── Scenario 3: Standalone URL Model Legitimate (Class 0) ───────────

class TestScenario3URLModelLegitimate:
    """URL model fallback predicting class 0 MUST be classified as legitimate."""

    def test_arbiter_outputs_legitimate_for_url_class_0(self, mock_url_features):
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.95, 0.04, 0.01]])

        res = arbiter.arbitrate(
            url="https://example-test-site.org/page",
            url_features=mock_url_features,
            url_model=url_model,
            screenshot_base64=None,
        )

        assert res.model_routing == "url_model_fallback"
        assert res.prediction == 0
        assert res.threat_category == "legitimate"
        assert res.class_probabilities["legitimate"] == 0.95
        assert res.class_probabilities["phishing"] == 0.04
        assert res.class_probabilities["malware_distribution"] == 0.01

    def test_extension_renders_safe_for_url_legitimate(self):
        backend_data = {
            "url": "https://www.google.com/",
            "prediction": 0,
            "threat_category": "legitimate",
            "risk_score": 0.05,
            "confidence": 0.95,
            "analysis_status": "completed",
            "model_routing": "url_model_fallback",
            "visual_status": "not_provided",
            "is_shadow_mode": True,
            "class_probabilities": {
                "legitimate": 0.95,
                "phishing": 0.04,
                "malware_distribution": 0.01,
            },
        }

        dom = render_in_node(backend_data)
        assert dom["threatCategoryText"] == "Legitimate"
        assert dom["statusText"] == "Safe"
        assert "status-safe" in dom["threatCategoryClasses"]
        assert "shield-safe" in dom["shieldClasses"]
        assert "notice-safe" in dom["noticeClasses"]
        assert "Verified legitimate website" in dom["noticeText"]
        assert dom["probLegitLabelText"] == "Legitimate"


# ── Scenario 4: Standalone URL Model Phishing (Class 1) ─────────────

class TestScenario4URLModelPhishing:
    """URL model fallback predicting class 1 MUST be classified as phishing."""

    def test_arbiter_outputs_phishing_for_url_class_1(self, mock_url_features):
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([1])
        url_model.predict_proba.return_value = np.array([[0.05, 0.92, 0.03]])

        res = arbiter.arbitrate(
            url="https://paypal-signin.secure.xyz/",
            url_features=mock_url_features,
            url_model=url_model,
            screenshot_base64=None,
        )

        assert res.model_routing == "url_model_fallback"
        assert res.prediction == 1
        assert res.threat_category == "phishing"
        assert res.class_probabilities["phishing"] == 0.92


# ── Scenario 5: URL Malware Preemption & Class 2 ────────────────────

class TestScenario5URLModelMalware:
    """URL model detecting malware MUST preempt multimodal and classify as malware_distribution."""

    def test_malware_preempts_multimodal_even_with_screenshot(
        self, mock_url_features, sample_png_b64
    ):
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([2])
        url_model.predict_proba.return_value = np.array([[0.01, 0.04, 0.95]])

        multi_model = MagicMock()  # Must NOT be evaluated

        res = arbiter.arbitrate(
            url="http://192.168.1.100:8080/payload.exe",
            url_features=mock_url_features,
            url_model=url_model,
            multimodal_model=multi_model,
            screenshot_base64=sample_png_b64,
        )

        assert res.model_routing == "url_malware_preempted"
        assert res.prediction == 2
        assert res.threat_category == "malware_distribution"
        assert res.class_probabilities["malware_distribution"] == 0.95
        multi_model.predict.assert_not_called()

    def test_extension_renders_critical_for_malware(self):
        backend_data = {
            "url": "http://malware-source.net/rat.exe",
            "prediction": 2,
            "threat_category": "malware_distribution",
            "risk_score": 0.98,
            "confidence": 0.95,
            "analysis_status": "completed",
            "model_routing": "url_malware_preempted",
            "visual_status": "extracted",
            "is_shadow_mode": True,
            "class_probabilities": {
                "legitimate": 0.02,
                "phishing": 0.03,
                "malware_distribution": 0.95,
            },
        }

        dom = render_in_node(backend_data)
        assert dom["threatCategoryText"] == "Malware"
        assert dom["statusText"] == "Critical"
        assert "status-malware" in dom["threatCategoryClasses"]
        assert "shield-malware" in dom["shieldClasses"]
        assert "notice-malware" in dom["noticeClasses"]
        assert "Malware distribution detected" in dom["noticeText"]


# ── Scenario 6: Screenshot Failure Fallback ────────────────────────

class TestScenario6ScreenshotFailureFallback:
    """Missing or invalid screenshot gracefully falls back to URL model preserving 3 classes."""

    def test_missing_screenshot_falls_back_to_url_model(self, mock_url_features):
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.90, 0.08, 0.02]])

        res = arbiter.arbitrate(
            url="https://site.org",
            url_features=mock_url_features,
            url_model=url_model,
            screenshot_base64=None,
        )

        assert res.model_routing == "url_model_fallback"
        assert res.visual_status == "not_provided"
        assert res.multimodal_model_result["status"] == "skipped"
        assert res.threat_category == "legitimate"

    def test_invalid_screenshot_falls_back_to_url_model(self, mock_url_features):
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([1])
        url_model.predict_proba.return_value = np.array([[0.10, 0.85, 0.05]])

        res = arbiter.arbitrate(
            url="https://phish.org",
            url_features=mock_url_features,
            url_model=url_model,
            screenshot_base64="data:image/png;base64,not_a_valid_image",
        )

        assert res.model_routing == "url_model_fallback"
        assert res.visual_status == "invalid_image"
        assert res.multimodal_model_result["status"] == "skipped"
        assert res.threat_category == "phishing"


# ── Scenario 7: Probability and Label Consistency ──────────────────

class TestScenario7ProbabilityAndLabelConsistency:
    """Validate consistency across probability keys and UI display labels."""

    def test_unverified_candidate_never_described_as_safe(self):
        backend_data = {
            "url": "https://unverified-domain.org",
            "prediction": 0,
            "threat_category": "unverified_candidate",
            "risk_score": 0.04,
            "confidence": 0.96,
            "analysis_status": "completed",
            "model_routing": "multimodal_full",
            "visual_status": "extracted",
            "class_probabilities": {
                "unverified_candidate": 0.96,
                "legitimate": 0.96,
                "phishing": 0.04,
                "malware_distribution": 0.0,
            },
        }

        dom = render_in_node(backend_data)
        assert dom["threatCategoryText"] != "Safe"
        assert dom["threatCategoryText"] != "Legitimate"
        assert dom["statusText"] != "Safe"
        assert "status-safe" not in dom["statusClasses"]
        assert "shield-safe" not in dom["shieldClasses"]
        assert "notice-safe" not in dom["noticeClasses"]

    def test_additive_alias_preserves_legacy_compatibility(
        self, mock_url_features, sample_png_b64
    ):
        arbiter = DecisionArbiter()
        url_model = MagicMock()
        url_model.predict.return_value = np.array([0])
        url_model.predict_proba.return_value = np.array([[0.90, 0.08, 0.02]])

        multi_model = MagicMock()
        multi_model.predict.return_value = np.array([0])
        multi_model.predict_proba.return_value = np.array([[0.94, 0.06]])

        res = arbiter.arbitrate(
            url="https://site.org",
            url_features=mock_url_features,
            url_model=url_model,
            multimodal_model=multi_model,
            screenshot_base64=sample_png_b64,
        )

        # Legacy consumers looking for 'legitimate' key must receive the probability
        assert "legitimate" in res.class_probabilities
        assert res.class_probabilities["legitimate"] == 0.94
        # New consumers looking for 'unverified_candidate' key must also receive it
        assert "unverified_candidate" in res.class_probabilities
        assert res.class_probabilities["unverified_candidate"] == 0.94
        # Primary threat category is unequivocally unverified_candidate
        assert res.threat_category == "unverified_candidate"

