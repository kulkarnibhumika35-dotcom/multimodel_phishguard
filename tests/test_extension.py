"""
PhishGuard – Chrome Extension Unit & Integration Tests (pytest)

Validates:
1. Manifest V3 specification and permissions.
2. DOM structure and CSS styling for 3-class visual states.
3. Execution of the JavaScript unit test suite (Node.js).
4. Alignment with FastAPI POST /analyze contract.
5. End-to-end integration with live FastAPI responses and simulated DOM rendering.
"""

import json
import os
import subprocess
from typing import Any
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.orm import sessionmaker

from backend.database.session import create_db_engine, get_db, init_db
from backend.main import app

EXTENSION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extension"))
MANIFEST_PATH = os.path.join(EXTENSION_DIR, "manifest.json")
POPUP_HTML_PATH = os.path.join(EXTENSION_DIR, "popup.html")
POPUP_CSS_PATH = os.path.join(EXTENSION_DIR, "popup.css")
POPUP_JS_PATH = os.path.join(EXTENSION_DIR, "popup.js").replace("\\", "/")
TEST_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_extension_script.js"))


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def app_with_temp_db(tmp_path, monkeypatch):
    """Point FastAPI database session dependency to an isolated SQLite file."""
    db_file = str(tmp_path / "extension_test.db")
    monkeypatch.setenv("PHISHGUARD_DB_PATH", db_file)

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
    yield db_file
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture
async def client(app_with_temp_db):
    """Async test client bound to the FastAPI application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def render_in_node(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute popup.renderResult in Node.js with mock DOM and return DOM state."""
    js_code = f"""
    const assert = require('assert');
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
      infoNotice: createMockElement('info-notice', ['hidden']),
      shield: createMockElement('brand-shield'),
      resultSec: createMockElement('result-section', ['hidden']),
      probEls: {{
        section: createMockElement('probabilities-section', ['hidden']),
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
      }}
    }};

    const data = {json.dumps(payload)};
    popup.renderResult(data, els);

    const result = {{
      threatCategoryText: els.threatCategoryEl.textContent,
      threatCategoryClasses: els.threatCategoryEl.classList.list,
      statusText: els.statusEl.textContent,
      statusClasses: els.statusEl.classList.list,
      riskScoreText: els.riskScoreEl.textContent,
      shieldClasses: els.shield.classList.list,
      noticeText: els.infoNotice.textContent,
      noticeClasses: els.infoNotice.classList.list,
      probSectionHidden: els.probEls.section.classList.contains('hidden'),
      probLegitVal: els.probEls.legitVal.textContent,
      probPhishVal: els.probEls.phishVal.textContent,
      probMalwareVal: els.probEls.malwareVal.textContent,
      featSectionHidden: els.featEls.section.classList.contains('hidden'),
      featCount: els.featEls.list.children ? els.featEls.list.children.length : 0
    }};

    console.log(JSON.stringify(result));
    """
    res = subprocess.run(["node", "-e", js_code], capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip())


# ── 1. Manifest V3 Architecture Tests ───────────────────────────

class TestExtensionManifest:
    def test_manifest_exists_and_is_valid_json(self):
        assert os.path.isfile(MANIFEST_PATH)
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["name"] == "PhishGuard"
        assert manifest["manifest_version"] == 3

    def test_manifest_permissions(self):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert "activeTab" in manifest.get("permissions", [])
        assert "http://127.0.0.1:8000/*" in manifest.get("host_permissions", [])

    def test_manifest_action_and_icons(self):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        action = manifest.get("action", {})
        assert action.get("default_popup") == "popup.html"
        assert action.get("default_icon") == "icon.png"
        assert os.path.isfile(os.path.join(EXTENSION_DIR, "icon.png"))


# ── 2. HTML & CSS Visual States ────────────────────────────────

class TestExtensionHTMLAndCSS:
    def test_popup_html_contains_required_elements(self):
        assert os.path.isfile(POPUP_HTML_PATH)
        with open(POPUP_HTML_PATH, "r", encoding="utf-8") as f:
            html = f.read()

        required_ids = [
            "current-url",
            "analyze-btn",
            "loading",
            "error-msg",
            "result-section",
            "analyzed-url-wrapper",
            "analyzed-url",
            "threat-category",
            "prediction",
            "risk-score",
            "status",
            "info-notice",
            "probabilities-section",
            "prob-legit-bar",
            "prob-legit-val",
            "prob-phish-bar",
            "prob-phish-val",
            "prob-malware-bar",
            "prob-malware-val",
            "features-section",
            "features-list",
        ]
        for el_id in required_ids:
            assert f'id="{el_id}"' in html, f"Missing element with id='{el_id}' in popup.html"

    def test_popup_css_contains_three_class_and_state_styles(self):
        assert os.path.isfile(POPUP_CSS_PATH)
        with open(POPUP_CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()

        required_classes = [
            ".status-safe",
            ".status-danger",
            ".status-malware",
            ".status-warn",
            ".shield-safe",
            ".shield-danger",
            ".shield-malware",
            ".shield-warn",
            ".notice-safe",
            ".notice-danger",
            ".notice-malware",
            ".notice-warn",
            ".legit-bar",
            ".phish-bar",
            ".malware-bar",
            ".badge-safe",
            ".badge-threat",
            ".badge-malware",
        ]
        for cls in required_classes:
            assert cls in css, f"Missing CSS class '{cls}' in popup.css"


# ── 3. JavaScript Execution Tests via Node.js ───────────────────

class TestExtensionJavaScriptExecution:
    def test_node_unit_test_suite_passes(self):
        assert os.path.isfile(TEST_SCRIPT_PATH)
        result = subprocess.run(
            ["node", TEST_SCRIPT_PATH],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"Node test suite failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert "All PhishGuard Extension Tests Passed Successfully!" in result.stdout


# ── 4. Backend Contract Alignment ──────────────────────────────

class TestExtensionBackendContractAlignment:
    def test_contract_alignment_with_fastapi_schema(self):
        from backend.schemas.analysis import AnalysisResponse

        schema_fields = set(AnalysisResponse.model_fields.keys())
        expected_extension_fields = {
            "url",
            "prediction",
            "threat_category",
            "risk_score",
            "class_probabilities",
            "probabilities",
            "top_features",
            "explanation",
            "analysis_status",
        }
        assert expected_extension_fields.issubset(schema_fields), (
            f"Missing fields in backend schema: {expected_extension_fields - schema_fields}"
        )


# ── 5. End-to-End FastAPI Integration Tests ───────────────────────

class TestExtensionE2EIntegration:
    EXTENSION_ORIGIN = "chrome-extension://knldjmfmopnppmplflldnbnekgfdnmfi"

    @pytest.mark.anyio
    async def test_e2e_cors_preflight_and_post(self, client):
        """Verify CORS preflight and actual POST requests from Chrome Extension origin succeed."""
        options_resp = await client.options(
            "/analyze",
            headers={
                "Origin": self.EXTENSION_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert options_resp.status_code == 200
        assert options_resp.headers.get("access-control-allow-origin") == self.EXTENSION_ORIGIN

        post_resp = await client.post(
            "/analyze",
            json={"url": "https://example.com"},
            headers={"Origin": self.EXTENSION_ORIGIN},
        )
        assert post_resp.status_code == 200
        assert post_resp.headers.get("access-control-allow-origin") == self.EXTENSION_ORIGIN

    @pytest.mark.anyio
    async def test_e2e_legitimate_url_pipeline(self, client):
        """End-to-end: Legitimate URL analyzed via backend and rendered in extension DOM."""
        resp = await client.post(
            "/analyze",
            json={"url": "https://www.google.com"},
            headers={"Origin": self.EXTENSION_ORIGIN},
        )
        assert resp.status_code == 200
        backend_data = resp.json()

        assert backend_data["prediction"] == 0
        assert backend_data["threat_category"] == "legitimate"
        assert backend_data["risk_score"] <= 0.5
        assert backend_data["class_probabilities"]["legitimate"] >= 0.9

        dom = render_in_node(backend_data)
        assert dom["threatCategoryText"] == "Legitimate"
        assert dom["statusText"] == "Safe"
        assert "status-safe" in dom["threatCategoryClasses"]
        assert "status-safe" in dom["statusClasses"]
        assert "shield-safe" in dom["shieldClasses"]
        assert "notice-safe" in dom["noticeClasses"]
        assert dom["probSectionHidden"] is False
        assert dom["featSectionHidden"] is False
        assert dom["featCount"] > 0

    @pytest.mark.anyio
    async def test_e2e_wikipedia_with_trailing_slash_pipeline(self, client):
        """End-to-end: Wikipedia with trailing slash correctly predicts legitimate in extension DOM."""
        resp = await client.post(
            "/analyze",
            json={"url": "https://www.wikipedia.org/"},
            headers={"Origin": self.EXTENSION_ORIGIN},
        )
        assert resp.status_code == 200
        backend_data = resp.json()

        assert backend_data["prediction"] == 0
        assert backend_data["threat_category"] == "legitimate"
        assert backend_data["risk_score"] < 0.15
        assert backend_data["class_probabilities"]["legitimate"] > 0.90

        dom = render_in_node(backend_data)
        assert dom["threatCategoryText"] == "Legitimate"
        assert dom["statusText"] == "Safe"
        assert "status-safe" in dom["threatCategoryClasses"]
        assert "status-safe" in dom["statusClasses"]
        assert "shield-safe" in dom["shieldClasses"]
        assert "notice-safe" in dom["noticeClasses"]
        assert dom["probSectionHidden"] is False
        assert dom["featSectionHidden"] is False
        assert dom["featCount"] > 0

    @pytest.mark.anyio
    async def test_e2e_phishing_url_pipeline(self, client):
        """End-to-end: Phishing URL analyzed via backend and rendered in extension DOM."""
        phish_url = "https://secure-paypal-login.account-verification.tk/login.php"
        resp = await client.post(
            "/analyze",
            json={"url": phish_url},
            headers={"Origin": self.EXTENSION_ORIGIN},
        )
        assert resp.status_code == 200
        backend_data = resp.json()

        assert backend_data["prediction"] == 1
        assert backend_data["threat_category"] == "phishing"
        assert backend_data["risk_score"] > 0.5
        assert backend_data["class_probabilities"]["phishing"] > 0.5

        dom = render_in_node(backend_data)
        assert dom["threatCategoryText"] == "Phishing"
        assert dom["statusText"] == "Danger"
        assert "status-danger" in dom["threatCategoryClasses"]
        assert "status-danger" in dom["statusClasses"]
        assert "shield-danger" in dom["shieldClasses"]
        assert "notice-danger" in dom["noticeClasses"]
        assert dom["probSectionHidden"] is False
        assert dom["featSectionHidden"] is False
        assert dom["featCount"] > 0

    @pytest.mark.anyio
    async def test_e2e_malware_url_pipeline(self, client):
        """End-to-end: Malware URL analyzed via backend and rendered in extension DOM."""
        malware_url = "http://192.168.1.1:8080/setup.exe"
        resp = await client.post(
            "/analyze",
            json={"url": malware_url},
            headers={"Origin": self.EXTENSION_ORIGIN},
        )
        assert resp.status_code == 200
        backend_data = resp.json()

        assert backend_data["prediction"] == 2
        assert backend_data["threat_category"] == "malware_distribution"
        assert backend_data["risk_score"] > 0.5
        assert backend_data["class_probabilities"]["malware_distribution"] > 0.5

        dom = render_in_node(backend_data)
        assert dom["threatCategoryText"] == "Malware"
        assert dom["statusText"] == "Critical"
        assert "status-malware" in dom["threatCategoryClasses"]
        assert "status-malware" in dom["statusClasses"]
        assert "shield-malware" in dom["shieldClasses"]
        assert "notice-malware" in dom["noticeClasses"]
        assert dom["probSectionHidden"] is False
        assert dom["featSectionHidden"] is False
        assert dom["featCount"] > 0

    @pytest.mark.anyio
    async def test_e2e_model_unavailable_pipeline(self, client, monkeypatch, tmp_path):
        """End-to-end: Model unavailable handled safely in extension DOM."""
        missing_model = str(tmp_path / "absent_model.joblib")
        monkeypatch.setenv("PHISHGUARD_MODEL_PATH", missing_model)

        resp = await client.post(
            "/analyze",
            json={"url": "https://example.com"},
            headers={"Origin": self.EXTENSION_ORIGIN},
        )
        assert resp.status_code == 200
        backend_data = resp.json()
        assert backend_data["analysis_status"] == "model_unavailable"

        dom = render_in_node(backend_data)
        assert dom["threatCategoryText"] == "Unavailable"
        assert dom["statusText"] == "Unavailable"
        assert "status-warn" in dom["threatCategoryClasses"]
        assert "status-warn" in dom["statusClasses"]
        assert "shield-warn" in dom["shieldClasses"]
        assert "notice-warn" in dom["noticeClasses"]
        assert dom["probSectionHidden"] is True
        assert dom["featSectionHidden"] is True

    @pytest.mark.anyio
    async def test_e2e_rejected_schemes(self, client):
        """Verify invalid schemes rejected by backend validation with HTTP 422."""
        for invalid_url in ["ftp://files.example.com", "chrome://extensions", "file:///etc/passwd"]:
            resp = await client.post(
                "/analyze",
                json={"url": invalid_url},
                headers={"Origin": self.EXTENSION_ORIGIN},
            )
            assert resp.status_code == 422
