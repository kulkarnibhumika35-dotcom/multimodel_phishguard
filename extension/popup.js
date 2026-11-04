// PhishGuard – Popup Script
// Connects to FastAPI backend (POST http://127.0.0.1:8000/analyze)
// Displays analyzed URL, 3-class prediction, risk score, category probabilities,
// and SHAP explainability indicators honestly and safely.

const API_ENDPOINT = "http://127.0.0.1:8000/analyze";
const REQUEST_TIMEOUT_MS = 12000;

// ── URL Validation (HTTP / HTTPS Only) ─────────────────────────
function isHttpOrHttps(url) {
  if (!url || typeof url !== "string") return false;
  try {
    const parsed = new URL(url.trim());
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

// ── Classify Prediction State ─────────────────────────────────
function classifyPredictionState(data) {
  if (!data) return "unavailable";

  if (data.analysis_status === "model_unavailable") {
    return "unavailable";
  }

  // If both prediction and threat_category are absent or null
  if (
    (data.prediction === null || data.prediction === undefined) &&
    !data.threat_category
  ) {
    return "unavailable";
  }

  const pred =
    data.prediction !== null && data.prediction !== undefined
      ? Number(data.prediction)
      : null;
  const cat = String(data.threat_category || "").toLowerCase();

  if (pred === 0 || cat === "legitimate") {
    return "legitimate";
  }
  if (pred === 1 || cat === "phishing") {
    return "phishing";
  }
  if (pred === 2 || cat === "malware_distribution" || cat === "malware") {
    return "malware";
  }

  if (data.prediction === null) {
    return "unavailable";
  }

  return "other";
}

// ── Format Percentage ─────────────────────────────────────────
function formatPercent(value) {
  if (value === null || value === undefined || isNaN(value)) {
    return "—";
  }
  const num = Number(value);
  const pct = num <= 1.0 ? num * 100 : num;
  return `${pct.toFixed(1)}%`;
}

// ── Format Human-Readable Category Label ──────────────────────
function formatCategoryLabel(state, rawCategory) {
  switch (state) {
    case "legitimate":
      return "Legitimate";
    case "phishing":
      return "Phishing";
    case "malware":
      return "Malware";
    case "unavailable":
      return "Unavailable";
    default:
      return rawCategory ? String(rawCategory) : "Unknown";
  }
}

// ── Render Category Probabilities ─────────────────────────────
function renderProbabilities(probs, els) {
  if (!els || !els.section) return;

  if (!probs || typeof probs !== "object") {
    hideElement(els.section);
    return;
  }

  const pLegit = Number(probs.legitimate ?? 0) * 100;
  const pPhish = Number(probs.phishing ?? 0) * 100;
  const pMalware = Number(probs.malware_distribution ?? probs.malware ?? 0) * 100;

  if (els.legitBar && els.legitVal) {
    els.legitBar.style.width = `${Math.min(100, Math.max(0, pLegit))}%`;
    els.legitVal.textContent = `${pLegit.toFixed(1)}%`;
  }

  if (els.phishBar && els.phishVal) {
    els.phishBar.style.width = `${Math.min(100, Math.max(0, pPhish))}%`;
    els.phishVal.textContent = `${pPhish.toFixed(1)}%`;
  }

  if (els.malwareBar && els.malwareVal) {
    els.malwareBar.style.width = `${Math.min(100, Math.max(0, pMalware))}%`;
    els.malwareVal.textContent = `${pMalware.toFixed(1)}%`;
  }

  showElement(els.section);
}

// ── Render Top SHAP Features ──────────────────────────────────
function renderFeatures(topFeatures, els, state) {
  if (!els || !els.section || !els.list) return;

  if (!Array.isArray(topFeatures) || topFeatures.length === 0) {
    hideElement(els.section);
    return;
  }

  const doc =
    typeof document !== "undefined"
      ? document
      : els.list && els.list.ownerDocument
        ? els.list.ownerDocument
        : null;

  if (!doc || !doc.createElement) {
    hideElement(els.section);
    return;
  }

  els.list.innerHTML = "";

  topFeatures.forEach((feat) => {
    const item = doc.createElement("div");
    item.className = "feature-item";

    const header = doc.createElement("div");
    header.className = "feature-header";

    const nameEl = doc.createElement("span");
    nameEl.className = "feature-name";
    nameEl.textContent = String(feat.feature_name || "indicator");

    const badgeEl = doc.createElement("span");
    badgeEl.className = "feature-badge";

    const contrib = typeof feat.contribution === "number" ? feat.contribution : 0;
    const sign = contrib >= 0 ? "+" : "";
    const direction = feat.direction || (contrib > 0 ? "positive" : contrib < 0 ? "negative" : "neutral");

    if (state === "legitimate") {
      if (contrib >= 0) {
        badgeEl.classList.add("badge-safe");
        badgeEl.textContent = `${sign}${contrib.toFixed(3)} Safe`;
      } else {
        badgeEl.classList.add("badge-threat");
        badgeEl.textContent = `${contrib.toFixed(3)} Risk`;
      }
    } else if (state === "malware") {
      if (contrib > 0) {
        badgeEl.classList.add("badge-malware");
        badgeEl.textContent = `${sign}${contrib.toFixed(3)} Malware`;
      } else {
        badgeEl.classList.add("badge-safe");
        badgeEl.textContent = `${contrib.toFixed(3)} Safe`;
      }
    } else {
      // Phishing or other threat state
      if (contrib > 0) {
        badgeEl.classList.add("badge-threat");
        badgeEl.textContent = `${sign}${contrib.toFixed(3)} Risk`;
      } else {
        badgeEl.classList.add("badge-safe");
        badgeEl.textContent = `${contrib.toFixed(3)} Safe`;
      }
    }

    header.appendChild(nameEl);
    header.appendChild(badgeEl);

    const textEl = doc.createElement("p");
    textEl.className = "feature-text";
    textEl.textContent = String(feat.explanation || "Feature influenced prediction.");

    item.appendChild(header);
    item.appendChild(textEl);
    els.list.appendChild(item);
  });

  showElement(els.section);
}

// ── Core Render Function ──────────────────────────────────────
function renderResult(data, elements = {}) {
  const els = {
    urlBox: elements.urlBox || document.getElementById("current-url"),
    analyzedUrlWrapper: elements.analyzedUrlWrapper || document.getElementById("analyzed-url-wrapper"),
    analyzedUrlEl: elements.analyzedUrlEl || document.getElementById("analyzed-url"),
    threatCategoryEl: elements.threatCategoryEl || document.getElementById("threat-category"),
    predictionEl: elements.predictionEl || document.getElementById("prediction"),
    riskScoreEl: elements.riskScoreEl || document.getElementById("risk-score"),
    statusEl: elements.statusEl || document.getElementById("status"),
    infoNotice: elements.infoNotice || document.getElementById("info-notice"),
    shield: elements.shield || document.getElementById("brand-shield"),
    resultSec: elements.resultSec || document.getElementById("result-section"),
    probEls: elements.probEls || {
      section: document.getElementById("probabilities-section"),
      legitBar: document.getElementById("prob-legit-bar"),
      legitVal: document.getElementById("prob-legit-val"),
      phishBar: document.getElementById("prob-phish-bar"),
      phishVal: document.getElementById("prob-phish-val"),
      malwareBar: document.getElementById("prob-malware-bar"),
      malwareVal: document.getElementById("prob-malware-val"),
    },
    featEls: elements.featEls || {
      section: document.getElementById("features-section"),
      list: document.getElementById("features-list"),
    },
  };

  const targetUrl = data.url || (els.urlBox ? els.urlBox.textContent : "");

  // Display analyzed URL in both header box and result section
  if (els.urlBox) els.urlBox.textContent = targetUrl;
  if (els.analyzedUrlEl && els.analyzedUrlWrapper) {
    els.analyzedUrlEl.textContent = targetUrl;
    showElement(els.analyzedUrlWrapper);
  }

  // Clear previous color classes
  const statusClasses = ["status-safe", "status-danger", "status-malware", "status-warn"];
  const shieldClasses = ["shield-safe", "shield-danger", "shield-malware", "shield-warn"];
  const noticeClasses = ["notice-safe", "notice-danger", "notice-malware", "notice-warn"];

  [els.threatCategoryEl, els.predictionEl, els.riskScoreEl, els.statusEl].forEach((el) => {
    if (el) el.classList.remove(...statusClasses);
  });
  if (els.shield) els.shield.classList.remove(...shieldClasses);
  if (els.infoNotice) els.infoNotice.classList.remove(...noticeClasses);

  const state = classifyPredictionState(data);
  const categoryLabel = formatCategoryLabel(state, data.threat_category);

  if (els.threatCategoryEl) {
    els.threatCategoryEl.textContent = categoryLabel;
  }
  if (els.predictionEl) {
    els.predictionEl.textContent = categoryLabel;
  }

  switch (state) {
    case "unavailable": {
      if (els.threatCategoryEl) els.threatCategoryEl.classList.add("status-warn");
      if (els.statusEl) {
        els.statusEl.textContent = "Unavailable";
        els.statusEl.classList.add("status-warn");
      }
      if (els.riskScoreEl) {
        els.riskScoreEl.textContent = "—";
        els.riskScoreEl.classList.add("status-warn");
      }
      if (els.shield) els.shield.classList.add("shield-warn");
      if (els.infoNotice) {
        els.infoNotice.textContent =
          "The ML model is currently unavailable on the backend. No prediction was made.";
        els.infoNotice.classList.add("notice-warn");
        showElement(els.infoNotice);
      }
      hideElement(els.probEls.section);
      hideElement(els.featEls.section);
      break;
    }

    case "legitimate": {
      if (els.threatCategoryEl) els.threatCategoryEl.classList.add("status-safe");
      if (els.statusEl) {
        els.statusEl.textContent = "Safe";
        els.statusEl.classList.add("status-safe");
      }
      if (els.riskScoreEl) {
        els.riskScoreEl.textContent = formatPercent(data.risk_score);
        els.riskScoreEl.classList.add("status-safe");
      }
      if (els.shield) els.shield.classList.add("shield-safe");
      if (els.infoNotice) {
        els.infoNotice.textContent =
          "Verified legitimate website. Low threat indicators detected.";
        els.infoNotice.classList.add("notice-safe");
        showElement(els.infoNotice);
      }
      renderProbabilities(data.class_probabilities || data.probabilities, els.probEls);
      renderFeatures(data.top_features, els.featEls, state);
      break;
    }

    case "phishing": {
      if (els.threatCategoryEl) els.threatCategoryEl.classList.add("status-danger");
      if (els.statusEl) {
        els.statusEl.textContent = "Danger";
        els.statusEl.classList.add("status-danger");
      }
      if (els.riskScoreEl) {
        els.riskScoreEl.textContent = formatPercent(data.risk_score);
        els.riskScoreEl.classList.add("status-danger");
      }
      if (els.shield) els.shield.classList.add("shield-danger");
      if (els.infoNotice) {
        els.infoNotice.textContent =
          "Warning: High phishing threat detected. Avoid entering passwords or personal details.";
        els.infoNotice.classList.add("notice-danger");
        showElement(els.infoNotice);
      }
      renderProbabilities(data.class_probabilities || data.probabilities, els.probEls);
      renderFeatures(data.top_features, els.featEls, state);
      break;
    }

    case "malware": {
      if (els.threatCategoryEl) els.threatCategoryEl.classList.add("status-malware");
      if (els.statusEl) {
        els.statusEl.textContent = "Critical";
        els.statusEl.classList.add("status-malware");
      }
      if (els.riskScoreEl) {
        els.riskScoreEl.textContent = formatPercent(data.risk_score);
        els.riskScoreEl.classList.add("status-malware");
      }
      if (els.shield) els.shield.classList.add("shield-malware");
      if (els.infoNotice) {
        els.infoNotice.textContent =
          "Critical: Malware distribution or C2 threat detected. Do not download or execute files from this host.";
        els.infoNotice.classList.add("notice-malware");
        showElement(els.infoNotice);
      }
      renderProbabilities(data.class_probabilities || data.probabilities, els.probEls);
      renderFeatures(data.top_features, els.featEls, state);
      break;
    }

    default: {
      if (els.threatCategoryEl) {
        els.threatCategoryEl.textContent = String(data.prediction ?? "Unknown");
        els.threatCategoryEl.classList.add("status-warn");
      }
      if (els.statusEl) {
        els.statusEl.textContent = data.analysis_status || "Completed";
        els.statusEl.classList.add("status-warn");
      }
      if (els.riskScoreEl) {
        els.riskScoreEl.textContent = formatPercent(data.risk_score);
        els.riskScoreEl.classList.add("status-warn");
      }
      if (els.infoNotice) hideElement(els.infoNotice);
      renderProbabilities(data.class_probabilities || data.probabilities, els.probEls);
      renderFeatures(data.top_features, els.featEls, state);
      break;
    }
  }

  showElement(els.resultSec);
}

// ── UI Visibility Helpers ─────────────────────────────────────
function showElement(el) {
  if (el) el.classList.remove("hidden");
}

function hideElement(el) {
  if (el) el.classList.add("hidden");
}

function showError(msg, errorEl) {
  const el = errorEl || document.getElementById("error-msg");
  if (el) {
    el.textContent = msg;
    showElement(el);
  }
}

// ── DOM Event Binding ─────────────────────────────────────────
if (typeof document !== "undefined" && typeof document.addEventListener === "function") {
  document.addEventListener("DOMContentLoaded", () => {
    const urlBox = document.getElementById("current-url");
    const analyzeBtn = document.getElementById("analyze-btn");
    const loading = document.getElementById("loading");
    const errorMsg = document.getElementById("error-msg");
    const resultSec = document.getElementById("result-section");

    let activeUrl = "";

    // 1. Retrieve Active Tab URL via Chrome API
    if (typeof chrome !== "undefined" && chrome.tabs && chrome.tabs.query) {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs && tabs.length > 0 && tabs[0].url) {
          activeUrl = tabs[0].url.trim();
          urlBox.textContent = activeUrl;

          if (isHttpOrHttps(activeUrl)) {
            analyzeBtn.disabled = false;
            hideElement(errorMsg);
          } else {
            analyzeBtn.disabled = true;
            showError("Only HTTP and HTTPS URLs can be analyzed.", errorMsg);
          }
        } else {
          urlBox.textContent = "Unable to retrieve active tab URL";
          analyzeBtn.disabled = true;
          showError("Could not retrieve the active tab URL.", errorMsg);
        }
      });
    } else {
      // Fallback for standalone/dev browser preview
      urlBox.textContent = "https://example.com";
      activeUrl = "https://example.com";
      analyzeBtn.disabled = false;
    }

    // 2. Handle Analyze Button Click
    analyzeBtn.addEventListener("click", async () => {
      if (!isHttpOrHttps(activeUrl)) {
        showError("Only HTTP and HTTPS URLs can be analyzed.", errorMsg);
        return;
      }

      hideElement(errorMsg);
      hideElement(resultSec);
      showElement(loading);
      analyzeBtn.disabled = true;

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

      try {
        const response = await fetch(API_ENDPOINT, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ url: activeUrl }),
          signal: controller.signal,
        });

        if (!response.ok) {
          let errorDetail = `Backend returned HTTP ${response.status}`;
          try {
            const errData = await response.json();
            if (errData && errData.detail) {
              errorDetail = Array.isArray(errData.detail)
                ? errData.detail.map((d) => d.msg || JSON.stringify(d)).join(", ")
                : String(errData.detail);
            }
          } catch {
            // Non-JSON response body
          }
          throw new Error(errorDetail);
        }

        const data = await response.json();
        renderResult(data);
      } catch (err) {
        let message = err.message;
        if (err.name === "AbortError") {
          message = "Analysis timed out. The PhishGuard backend took too long to respond.";
        } else if (
          err.message.includes("Failed to fetch") ||
          err.message.includes("NetworkError") ||
          err.message.includes("net::ERR_CONNECTION_REFUSED")
        ) {
          message = "Unable to reach PhishGuard API at http://127.0.0.1:8000. Ensure the server is running.";
        }
        showError(message, errorMsg);
      } finally {
        clearTimeout(timeoutId);
        hideElement(loading);
        analyzeBtn.disabled = !isHttpOrHttps(activeUrl);
      }
    });
  });
}

// ── Node.js Export for Automated Testing ───────────────────────
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    API_ENDPOINT,
    REQUEST_TIMEOUT_MS,
    isHttpOrHttps,
    classifyPredictionState,
    formatPercent,
    formatCategoryLabel,
    renderProbabilities,
    renderFeatures,
    renderResult,
    showElement,
    hideElement,
    showError,
  };
}
