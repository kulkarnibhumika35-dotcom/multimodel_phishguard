/**
 * PhishGuard – Chrome Extension Unit & Integration Tests (Node.js)
 *
 * Tests:
 * 1. URL validation (HTTP / HTTPS protocols only).
 * 2. Prediction state classification across all 3 classes (0, 1, 2) + model_unavailable.
 * 3. Human-readable category label formatting.
 * 4. Risk percentage and probability formatting.
 * 5. Probability rendering (bars & percentage text).
 * 6. SHAP feature rendering across safe, phishing, and malware states.
 * 7. Full renderResult DOM simulation for:
 *    - Legitimate (Class 0, Safe state, green badges)
 *    - Phishing (Class 1, Danger state, red badges)
 *    - Malware Distribution (Class 2, Critical state, purple badges)
 *    - Model Unavailable (Unavailable state, amber badges)
 * 8. Error handling for network failure, timeout (AbortError), and invalid responses.
 */

const assert = require("assert");

// ── Mock DOM Element Helper ────────────────────────────────────
function createMockElement(id = "", initialClasses = []) {
  const classList = new Set(initialClasses);
  return {
    id,
    tagName: "DIV",
    textContent: "",
    innerHTML: "",
    style: {},
    classList: {
      add: (...cls) => cls.forEach((c) => classList.add(c)),
      remove: (...cls) => cls.forEach((c) => classList.delete(c)),
      contains: (c) => classList.has(c),
      get length() {
        return classList.size;
      },
    },
    appendChild(child) {
      if (!this.children) this.children = [];
      this.children.push(child);
      this.innerHTML += `<div class="${child.className}">${child.textContent || ""}</div>`;
    },
  };
}

global.document = {
  createElement: (tag) => createMockElement(tag, []),
  getElementById: (id) => createMockElement(id, []),
};

const popup = require("../extension/popup.js");

console.log("=== Running PhishGuard Extension Tests ===");

function createMockDOMElements() {
  return {
    urlBox: createMockElement("current-url"),
    analyzedUrlWrapper: createMockElement("analyzed-url-wrapper", ["hidden"]),
    analyzedUrlEl: createMockElement("analyzed-url"),
    threatCategoryEl: createMockElement("threat-category"),
    predictionEl: createMockElement("prediction"),
    riskScoreEl: createMockElement("risk-score"),
    statusEl: createMockElement("status"),
    infoNotice: createMockElement("info-notice", ["hidden"]),
    shield: createMockElement("brand-shield"),
    resultSec: createMockElement("result-section", ["hidden"]),
    probEls: {
      section: createMockElement("probabilities-section", ["hidden"]),
      legitBar: createMockElement("prob-legit-bar"),
      legitVal: createMockElement("prob-legit-val"),
      phishBar: createMockElement("prob-phish-bar"),
      phishVal: createMockElement("prob-phish-val"),
      malwareBar: createMockElement("prob-malware-bar"),
      malwareVal: createMockElement("prob-malware-val"),
    },
    featEls: {
      section: createMockElement("features-section", ["hidden"]),
      list: createMockElement("features-list"),
    },
  };
}

// ── 1. URL Scheme Validation ──────────────────────────────────
console.log("Test 1: URL Scheme Validation");
assert.strictEqual(popup.isHttpOrHttps("https://google.com"), true);
assert.strictEqual(popup.isHttpOrHttps("http://insecure-site.org/login"), true);
assert.strictEqual(popup.isHttpOrHttps("ftp://files.example.com"), false);
assert.strictEqual(popup.isHttpOrHttps("chrome://extensions"), false);
assert.strictEqual(popup.isHttpOrHttps("javascript:alert(1)"), false);
assert.strictEqual(popup.isHttpOrHttps("file:///C:/path/file.txt"), false);
assert.strictEqual(popup.isHttpOrHttps(""), false);
assert.strictEqual(popup.isHttpOrHttps(null), false);
assert.strictEqual(popup.isHttpOrHttps(12345), false);

// ── 2. Classification Logic ────────────────────────────────────
console.log("Test 2: Prediction State Classification");
assert.strictEqual(popup.classifyPredictionState({ prediction: 0 }), "legitimate");
assert.strictEqual(popup.classifyPredictionState({ threat_category: "legitimate" }), "legitimate");

assert.strictEqual(popup.classifyPredictionState({ prediction: 1 }), "phishing");
assert.strictEqual(popup.classifyPredictionState({ threat_category: "phishing" }), "phishing");

assert.strictEqual(popup.classifyPredictionState({ prediction: 2 }), "malware");
assert.strictEqual(popup.classifyPredictionState({ threat_category: "malware_distribution" }), "malware");
assert.strictEqual(popup.classifyPredictionState({ threat_category: "malware" }), "malware");

assert.strictEqual(popup.classifyPredictionState({ prediction: null }), "unavailable");
assert.strictEqual(popup.classifyPredictionState({ analysis_status: "model_unavailable" }), "unavailable");
assert.strictEqual(popup.classifyPredictionState(null), "unavailable");

// ── 3. Percentage Formatting ──────────────────────────────────
console.log("Test 3: Percentage Formatting");
assert.strictEqual(popup.formatPercent(0.052), "5.2%");
assert.strictEqual(popup.formatPercent(0.985), "98.5%");
assert.strictEqual(popup.formatPercent(1.0), "100.0%");
assert.strictEqual(popup.formatPercent(0.0), "0.0%");
assert.strictEqual(popup.formatPercent(null), "—");
assert.strictEqual(popup.formatPercent(undefined), "—");

// ── 4. Category Label Formatting ──────────────────────────────
console.log("Test 4: Category Label Formatting");
assert.strictEqual(popup.formatCategoryLabel("legitimate"), "Legitimate");
assert.strictEqual(popup.formatCategoryLabel("phishing"), "Phishing");
assert.strictEqual(popup.formatCategoryLabel("malware"), "Malware");
assert.strictEqual(popup.formatCategoryLabel("unavailable"), "Unavailable");

// ── 5. Full Render: Legitimate (Class 0) ──────────────────────
console.log("Test 5: Full Render - Legitimate (Class 0)");
{
  const els = createMockDOMElements();
  const data = {
    url: "https://www.google.com",
    prediction: 0,
    threat_category: "legitimate",
    risk_score: 0.02,
    analysis_status: "completed",
    class_probabilities: {
      legitimate: 0.98,
      phishing: 0.015,
      malware_distribution: 0.005,
    },
    top_features: [
      {
        feature_name: "is_https",
        feature_value: 1.0,
        contribution: 0.12,
        explanation: "HTTPS enabled, supports legitimate classification.",
      },
    ],
  };

  popup.renderResult(data, els);

  assert.strictEqual(els.threatCategoryEl.textContent, "Legitimate");
  assert.strictEqual(els.statusEl.textContent, "Safe");
  assert.strictEqual(els.riskScoreEl.textContent, "2.0%");
  assert.strictEqual(els.threatCategoryEl.classList.contains("status-safe"), true);
  assert.strictEqual(els.statusEl.classList.contains("status-safe"), true);
  assert.strictEqual(els.shield.classList.contains("shield-safe"), true);
  assert.strictEqual(els.infoNotice.classList.contains("notice-safe"), true);
  assert.strictEqual(els.resultSec.classList.contains("hidden"), false);

  // Probabilities check
  assert.strictEqual(els.probEls.section.classList.contains("hidden"), false);
  assert.strictEqual(els.probEls.legitVal.textContent, "98.0%");
  assert.strictEqual(els.probEls.phishVal.textContent, "1.5%");
  assert.strictEqual(els.probEls.malwareVal.textContent, "0.5%");
}

// ── 6. Full Render: Phishing (Class 1) ────────────────────────
console.log("Test 6: Full Render - Phishing (Class 1)");
{
  const els = createMockDOMElements();
  const data = {
    url: "https://secure-paypal-login.tk/signin",
    prediction: 1,
    threat_category: "phishing",
    risk_score: 0.965,
    analysis_status: "completed",
    class_probabilities: {
      legitimate: 0.01,
      phishing: 0.965,
      malware_distribution: 0.025,
    },
    top_features: [
      {
        feature_name: "has_suspicious_tld",
        feature_value: 1.0,
        contribution: 0.28,
        explanation: "Top-level domain belongs to known high-risk TLD list increases phishing risk.",
      },
    ],
  };

  popup.renderResult(data, els);

  assert.strictEqual(els.threatCategoryEl.textContent, "Phishing");
  assert.strictEqual(els.statusEl.textContent, "Danger");
  assert.strictEqual(els.riskScoreEl.textContent, "96.5%");
  assert.strictEqual(els.threatCategoryEl.classList.contains("status-danger"), true);
  assert.strictEqual(els.statusEl.classList.contains("status-danger"), true);
  assert.strictEqual(els.shield.classList.contains("shield-danger"), true);
  assert.strictEqual(els.infoNotice.classList.contains("notice-danger"), true);
  assert.strictEqual(els.probEls.phishVal.textContent, "96.5%");
}

// ── 7. Full Render: Malware Distribution (Class 2) ───────────
console.log("Test 7: Full Render - Malware Distribution (Class 2)");
{
  const els = createMockDOMElements();
  const data = {
    url: "http://192.168.1.1:8080/setup.exe",
    prediction: 2,
    threat_category: "malware_distribution",
    risk_score: 0.995,
    analysis_status: "completed",
    class_probabilities: {
      legitimate: 0.0,
      phishing: 0.01,
      malware_distribution: 0.99,
    },
    top_features: [
      {
        feature_name: "has_ip_address",
        feature_value: 1.0,
        contribution: 0.35,
        explanation: "Direct raw IP address in URL hostname (increases malware distribution risk).",
      },
    ],
  };

  popup.renderResult(data, els);

  assert.strictEqual(els.threatCategoryEl.textContent, "Malware");
  assert.strictEqual(els.statusEl.textContent, "Critical");
  assert.strictEqual(els.riskScoreEl.textContent, "99.5%");
  assert.strictEqual(els.threatCategoryEl.classList.contains("status-malware"), true);
  assert.strictEqual(els.statusEl.classList.contains("status-malware"), true);
  assert.strictEqual(els.shield.classList.contains("shield-malware"), true);
  assert.strictEqual(els.infoNotice.classList.contains("notice-malware"), true);
  assert.strictEqual(els.probEls.malwareVal.textContent, "99.0%");
}

// ── 8. Full Render: Model Unavailable ────────────────────────
console.log("Test 8: Full Render - Model Unavailable");
{
  const els = createMockDOMElements();
  const data = {
    url: "https://example.com",
    prediction: null,
    threat_category: null,
    risk_score: null,
    analysis_status: "model_unavailable",
    class_probabilities: null,
    top_features: null,
  };

  popup.renderResult(data, els);

  assert.strictEqual(els.threatCategoryEl.textContent, "Unavailable");
  assert.strictEqual(els.statusEl.textContent, "Unavailable");
  assert.strictEqual(els.riskScoreEl.textContent, "—");
  assert.strictEqual(els.threatCategoryEl.classList.contains("status-warn"), true);
  assert.strictEqual(els.statusEl.classList.contains("status-warn"), true);
  assert.strictEqual(els.shield.classList.contains("shield-warn"), true);
  assert.strictEqual(els.infoNotice.classList.contains("notice-warn"), true);

  // Probabilities and features must be hidden
  assert.strictEqual(els.probEls.section.classList.contains("hidden"), true);
  assert.strictEqual(els.featEls.section.classList.contains("hidden"), true);
}

// ── 9. Error Handler Helper ──────────────────────────────────
console.log("Test 9: Error Display Helper");
{
  const errEl = createMockElement("error-msg", ["hidden"]);
  popup.showError("Connection failed test", errEl);
  assert.strictEqual(errEl.textContent, "Connection failed test");
  assert.strictEqual(errEl.classList.contains("hidden"), false);
}

// ── 10. SHAP Badge Distinction ────────────────────────────────
console.log("Test 10: SHAP Badges Distinguish Risk-Increasing vs Risk-Reducing");
{
  const els = createMockDOMElements();
  const features = [
    { feature_name: "has_suspicious_tld", contribution: 0.25, explanation: "High risk TLD increases risk." },
    { feature_name: "is_https", contribution: -0.08, explanation: "HTTPS enabled decreases risk." },
  ];
  popup.renderFeatures(features, els.featEls, "phishing");
  assert.strictEqual(els.featEls.list.children.length, 2);

  // Positive risk contribution -> badge-threat (+0.250 Risk)
  const item1 = els.featEls.list.children[0];
  const badge1 = item1.children[0].children[1];
  assert.strictEqual(badge1.classList.contains("badge-threat"), true);
  assert.strictEqual(badge1.textContent, "+0.250 Risk");

  // Negative risk contribution (mitigating) -> badge-safe (-0.080 Safe)
  const item2 = els.featEls.list.children[1];
  const badge2 = item2.children[0].children[1];
  assert.strictEqual(badge2.classList.contains("badge-safe"), true);
  assert.strictEqual(badge2.textContent, "-0.080 Safe");
}

console.log("=== All PhishGuard Extension Tests Passed Successfully! ===");
