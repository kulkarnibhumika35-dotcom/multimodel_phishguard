// PhishGuard – Popup Script
// Connects to FastAPI backend (POST http://127.0.0.1:8000/analyze)
// Displays analyzed URL, unified 3-class prediction, risk score, category probabilities,
// model routing telemetry, visual status, shadow-mode advisory, and tri-model diagnostic inspection.

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

// ── Webpage Screenshot Capture (User-Triggered) ───────────────
function captureVisibleTabScreenshot() {
  return new Promise((resolve) => {
    if (
      typeof chrome === "undefined" ||
      !chrome.tabs ||
      typeof chrome.tabs.captureVisibleTab !== "function"
    ) {
      resolve(null);
      return;
    }

    // 1. Query the active tab in the current window to obtain its actual windowId
    if (typeof chrome.tabs.query === "function") {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.lastError) {
          console.warn(
            "[PhishGuard] Failed to query active tab for screenshot:",
            chrome.runtime.lastError.message
          );
          resolve(null);
          return;
        }

        const windowId =
          tabs && tabs.length > 0 && typeof tabs[0].windowId === "number"
            ? tabs[0].windowId
            : null;

        try {
          chrome.tabs.captureVisibleTab(windowId, { format: "png" }, (dataUrl) => {
            if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.lastError) {
              console.warn(
                `[PhishGuard] Screenshot capture failed for window ${windowId}:`,
                chrome.runtime.lastError.message
              );
              resolve(null);
            } else if (dataUrl && typeof dataUrl === "string") {
              resolve(dataUrl);
            } else {
              console.warn("[PhishGuard] Screenshot capture returned empty or invalid data.");
              resolve(null);
            }
          });
        } catch (err) {
          console.error("[PhishGuard] Unexpected exception during captureVisibleTab:", err);
          resolve(null);
        }
      });
    } else {
      // Fallback if tabs.query is unavailable in current context
      try {
        chrome.tabs.captureVisibleTab(null, { format: "png" }, (dataUrl) => {
          if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.lastError) {
            console.warn(
              "[PhishGuard] Screenshot capture failed:",
              chrome.runtime.lastError.message
            );
            resolve(null);
          } else if (dataUrl && typeof dataUrl === "string") {
            resolve(dataUrl);
          } else {
            console.warn("[PhishGuard] Screenshot capture returned empty or invalid data.");
            resolve(null);
          }
        });
      } catch (err) {
        console.error("[PhishGuard] Unexpected exception during captureVisibleTab:", err);
        resolve(null);
      }
    }
  });
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
  const routing = String(data.model_routing || "").toLowerCase();

  // Malware safeguard takes absolute precedence
  if (
    pred === 2 ||
    cat === "malware_distribution" ||
    cat === "malware" ||
    routing === "url_malware_preempted"
  ) {
    return "malware";
  }

  // Phishing threat detection
  if (pred === 1 || cat === "phishing") {
    return "phishing";
  }

  // Unverified candidate: Multimodal class 0 or candidate threat category
  // CRITICAL: Never map unverified candidate directly to verified legitimate
  if (
    cat === "unverified_candidate" ||
    cat === "candidate" ||
    (pred === 0 && routing === "multimodal_full")
  ) {
    return "unverified_candidate";
  }

  // Verified legitimate: URL-only baseline or explicit legitimate category
  if (pred === 0 || cat === "legitimate") {
    return "legitimate";
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
    case "unverified_candidate":
      return "Unverified Candidate";
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

// ── Format Routing & Visual Labels ─────────────────────────────
function formatRoutingLabel(routing) {
  switch (routing) {
    case "multimodal_full":
      return "Multimodal Full (51 Feat)";
    case "url_model_fallback":
      return "URL Fallback (20 Feat)";
    case "url_malware_preempted":
      return "Malware Preempted";
    case "model_unavailable":
      return "Model Unavailable";
    default:
      return routing ? String(routing).replace(/_/g, " ") : "Standard";
  }
}

function formatVisualStatusLabel(status) {
  switch (status) {
    case "extracted":
      return "Captured (30 Feat)";
    case "not_provided":
      return "Not Provided";
    case "capture_failed":
      return "Capture Failed";
    case "invalid_image":
      return "Invalid Image";
    case "extraction_failed":
      return "Extraction Failed";
    default:
      return status ? String(status).replace(/_/g, " ") : "None";
  }
}

// ── Format Supplementary Category Label ─────────────────────────
function formatSupplementaryCategory(pred, cat) {
  const lowerCat = String(cat || "").toLowerCase();
  if (lowerCat === "unverified_candidate" || (pred === 0 && lowerCat !== "legitimate")) {
    return "Unverified Candidate";
  }
  if (lowerCat === "legitimate" || pred === 0) {
    return "Legitimate";
  }
  if (lowerCat === "phishing" || pred === 1) {
    return "Phishing";
  }
  if (lowerCat === "malware_distribution" || lowerCat === "malware" || pred === 2) {
    return "Malware Distribution";
  }
  if (pred === null || pred === undefined) {
    return "Unavailable";
  }
  return String(cat || pred);
}

// ── Render Category Probabilities ─────────────────────────────
function renderProbabilities(probs, els, state) {
  if (!els || !els.section) return;

  if (!probs || typeof probs !== "object") {
    hideElement(els.section);
    return;
  }

  const isCandidate =
    state === "unverified_candidate" ||
    (probs.unverified_candidate !== undefined && state !== "legitimate");

  if (els.legitLabel) {
    els.legitLabel.textContent = isCandidate ? "Candidate" : "Legitimate";
  }

  const pLegit = Number(
    (isCandidate && probs.unverified_candidate !== undefined)
      ? probs.unverified_candidate
      : (probs.legitimate ?? 0)
  ) * 100;
  const pPhish = Number(probs.phishing ?? 0) * 100;
  const pMalware = Number(probs.malware_distribution ?? probs.malware ?? 0) * 100;

  if (els.legitBar && els.legitVal) {
    els.legitBar.style.width = `${Math.min(100, Math.max(0, pLegit))}%`;
    els.legitVal.textContent = `${pLegit.toFixed(1)}%`;
    if (isCandidate) {
      els.legitBar.classList.remove("legit-bar");
      els.legitBar.classList.add("candidate-bar");
    } else {
      els.legitBar.classList.remove("candidate-bar");
      els.legitBar.classList.add("legit-bar");
    }
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

    if (state === "unverified_candidate") {
      if (contrib >= 0) {
        badgeEl.classList.add("badge-candidate");
        badgeEl.textContent = `${sign}${contrib.toFixed(3)} Candidate`;
      } else {
        badgeEl.classList.add("badge-threat");
        badgeEl.textContent = `${contrib.toFixed(3)} Risk`;
      }
    } else if (state === "legitimate") {
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

// ── Render Multi-Model Breakdown (Diagnostic) ─────────────────
function renderMultiModelBreakdown(data, els) {
  if (!els || !els.multiModelDetails) return;

  const telemetry =
    (data && data.explanation && data.explanation.multi_model_telemetry) || {};

  const urlRes = data.url_model_result || telemetry.url_model_result;
  const multiRes = data.multimodal_model_result || telemetry.multimodal_model_result;
  const visRes = data.visual_model_result || telemetry.visual_model_result;

  if (!urlRes && !multiRes && !visRes) {
    hideElement(els.multiModelDetails);
    return;
  }

  // 1. URL Model Card
  if (els.subUrlStatus && els.subUrlPred && els.subUrlProbs) {
    if (urlRes && urlRes.status === "completed") {
      els.subUrlStatus.textContent = "Evaluated";
      els.subUrlStatus.className = "sub-badge sub-badge-active";
      els.subUrlPred.textContent = formatSupplementaryCategory(
        urlRes.prediction,
        urlRes.threat_category
      );
      if (urlRes.probabilities) {
        const pLeg = ((urlRes.probabilities.legitimate ?? 0) * 100).toFixed(1);
        const pPhi = ((urlRes.probabilities.phishing ?? 0) * 100).toFixed(1);
        const pMal = ((urlRes.probabilities.malware_distribution ?? 0) * 100).toFixed(1);
        els.subUrlProbs.textContent = `L: ${pLeg}% | P: ${pPhi}% | M: ${pMal}%`;
      } else {
        els.subUrlProbs.textContent = "";
      }
    } else {
      const rawStatus = urlRes && urlRes.status ? String(urlRes.status) : "Unavailable";
      els.subUrlStatus.textContent = rawStatus.charAt(0).toUpperCase() + rawStatus.slice(1);
      els.subUrlStatus.className = "sub-badge sub-badge-warn";
      els.subUrlPred.textContent = "—";
      els.subUrlProbs.textContent = "";
    }
  }

  // 2. Multimodal Model Card
  if (els.subMultiStatus && els.subMultiPred && els.subMultiProbs) {
    if (multiRes && multiRes.status === "completed") {
      els.subMultiStatus.textContent = "Primary Arbiter";
      els.subMultiStatus.className = "sub-badge sub-badge-active";
      els.subMultiPred.textContent = formatSupplementaryCategory(
        multiRes.prediction,
        multiRes.threat_category
      );
      if (multiRes.probabilities) {
        const pCand = ((multiRes.probabilities.unverified_candidate ?? 0) * 100).toFixed(1);
        const pPhi = ((multiRes.probabilities.phishing ?? 0) * 100).toFixed(1);
        els.subMultiProbs.textContent = `Candidate: ${pCand}% | Phishing: ${pPhi}%`;
      } else {
        els.subMultiProbs.textContent = "";
      }
    } else if (multiRes && multiRes.status === "overridden") {
      els.subMultiStatus.textContent = "Overridden";
      els.subMultiStatus.className = "sub-badge sub-badge-warn";
      els.subMultiPred.textContent = "Preempted by Malware Guard";
      els.subMultiProbs.textContent = multiRes.reason || "";
    } else {
      const rawStatus = multiRes && multiRes.status ? String(multiRes.status) : "Skipped";
      els.subMultiStatus.textContent = rawStatus.charAt(0).toUpperCase() + rawStatus.slice(1);
      els.subMultiStatus.className = "sub-badge sub-badge-neutral";
      els.subMultiPred.textContent = "—";
      els.subMultiProbs.textContent = multiRes && multiRes.reason ? multiRes.reason : "";
    }
  }

  // 3. Visual Model Card
  if (els.subVisStatus && els.subVisPred && els.subVisProbs) {
    if (visRes && visRes.status === "completed") {
      els.subVisStatus.textContent = "Auxiliary";
      els.subVisStatus.className = "sub-badge sub-badge-active";
      els.subVisPred.textContent = formatSupplementaryCategory(
        visRes.prediction,
        visRes.threat_category
      );
      if (visRes.probabilities) {
        const pCand = ((visRes.probabilities.unverified_candidate ?? 0) * 100).toFixed(1);
        const pPhi = ((visRes.probabilities.phishing ?? 0) * 100).toFixed(1);
        els.subVisProbs.textContent = `Candidate: ${pCand}% | Phishing: ${pPhi}%`;
      } else {
        els.subVisProbs.textContent = "";
      }
    } else {
      const rawStatus = visRes && visRes.status ? String(visRes.status) : "Skipped";
      els.subVisStatus.textContent = rawStatus.charAt(0).toUpperCase() + rawStatus.slice(1);
      els.subVisStatus.className = "sub-badge sub-badge-neutral";
      els.subVisPred.textContent = "—";
      els.subVisProbs.textContent = visRes && visRes.reason ? visRes.reason : "";
    }
  }

  showElement(els.multiModelDetails);
}

// ── Core Render Function ──────────────────────────────────────
function renderResult(data, elements = {}) {
  const doc = typeof document !== "undefined" ? document : null;

  const els = {
    urlBox: elements.urlBox || (doc ? doc.getElementById("current-url") : null),
    analyzedUrlWrapper: elements.analyzedUrlWrapper || (doc ? doc.getElementById("analyzed-url-wrapper") : null),
    analyzedUrlEl: elements.analyzedUrlEl || (doc ? doc.getElementById("analyzed-url") : null),
    threatCategoryEl: elements.threatCategoryEl || (doc ? doc.getElementById("threat-category") : null),
    predictionEl: elements.predictionEl || (doc ? doc.getElementById("prediction") : null),
    riskScoreEl: elements.riskScoreEl || (doc ? doc.getElementById("risk-score") : null),
    statusEl: elements.statusEl || (doc ? doc.getElementById("status") : null),
    modelRoutingEl: elements.modelRoutingEl || (doc ? doc.getElementById("model-routing") : null),
    visualStatusEl: elements.visualStatusEl || (doc ? doc.getElementById("visual-status") : null),
    infoNotice: elements.infoNotice || (doc ? doc.getElementById("info-notice") : null),
    shadowAdvisory: elements.shadowAdvisory || (doc ? doc.getElementById("shadow-advisory") : null),
    shield: elements.shield || (doc ? doc.getElementById("brand-shield") : null),
    resultSec: elements.resultSec || (doc ? doc.getElementById("result-section") : null),
    probEls: elements.probEls || {
      section: doc ? doc.getElementById("probabilities-section") : null,
      legitLabel: doc ? doc.getElementById("prob-legit-label") : null,
      legitBar: doc ? doc.getElementById("prob-legit-bar") : null,
      legitVal: doc ? doc.getElementById("prob-legit-val") : null,
      phishBar: doc ? doc.getElementById("prob-phish-bar") : null,
      phishVal: doc ? doc.getElementById("prob-phish-val") : null,
      malwareBar: doc ? doc.getElementById("prob-malware-bar") : null,
      malwareVal: doc ? doc.getElementById("prob-malware-val") : null,
    },
    featEls: elements.featEls || {
      section: doc ? doc.getElementById("features-section") : null,
      list: doc ? doc.getElementById("features-list") : null,
    },
    multiModelDetails: elements.multiModelDetails || (doc ? doc.getElementById("multi-model-details") : null),
    subUrlStatus: elements.subUrlStatus || (doc ? doc.getElementById("sub-url-status") : null),
    subUrlPred: elements.subUrlPred || (doc ? doc.getElementById("sub-url-pred") : null),
    subUrlProbs: elements.subUrlProbs || (doc ? doc.getElementById("sub-url-probs") : null),
    subMultiStatus: elements.subMultiStatus || (doc ? doc.getElementById("sub-multi-status") : null),
    subMultiPred: elements.subMultiPred || (doc ? doc.getElementById("sub-multi-pred") : null),
    subMultiProbs: elements.subMultiProbs || (doc ? doc.getElementById("sub-multi-probs") : null),
    subVisStatus: elements.subVisStatus || (doc ? doc.getElementById("sub-vis-status") : null),
    subVisPred: elements.subVisPred || (doc ? doc.getElementById("sub-vis-pred") : null),
    subVisProbs: elements.subVisProbs || (doc ? doc.getElementById("sub-vis-probs") : null),
  };

  const targetUrl = data.url || (els.urlBox ? els.urlBox.textContent : "");

  // Display analyzed URL in both header box and result section
  if (els.urlBox) els.urlBox.textContent = targetUrl;
  if (els.analyzedUrlEl && els.analyzedUrlWrapper) {
    els.analyzedUrlEl.textContent = targetUrl;
    showElement(els.analyzedUrlWrapper);
  }

  // Clear previous color classes
  const statusClasses = [
    "status-safe",
    "status-candidate",
    "status-danger",
    "status-malware",
    "status-warn",
  ];
  const shieldClasses = [
    "shield-safe",
    "shield-candidate",
    "shield-danger",
    "shield-malware",
    "shield-warn",
  ];
  const noticeClasses = [
    "notice-safe",
    "notice-candidate",
    "notice-danger",
    "notice-malware",
    "notice-warn",
  ];

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

  // Display pipeline routing and visual status
  if (els.modelRoutingEl) {
    els.modelRoutingEl.textContent = formatRoutingLabel(data.model_routing);
  }
  if (els.visualStatusEl) {
    els.visualStatusEl.textContent = formatVisualStatusLabel(data.visual_status);
  }

  // Ensure shadow mode advisory is displayed
  if (els.shadowAdvisory) {
    showElement(els.shadowAdvisory);
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

    case "unverified_candidate": {
      if (els.threatCategoryEl) {
        els.threatCategoryEl.textContent = "Unverified Candidate";
        els.threatCategoryEl.classList.add("status-candidate");
      }
      if (els.statusEl) {
        els.statusEl.textContent = "Candidate";
        els.statusEl.classList.add("status-candidate");
      }
      if (els.riskScoreEl) {
        els.riskScoreEl.textContent = formatPercent(data.risk_score);
        els.riskScoreEl.classList.add("status-candidate");
      }
      if (els.shield) els.shield.classList.add("shield-candidate");
      if (els.infoNotice) {
        els.infoNotice.textContent =
          "Unverified Candidate: Low phishing indicators detected by multimodal AI. Note: This domain is a candidate from unverified data and has NOT been independently verified as safe.";
        els.infoNotice.classList.add("notice-candidate");
        showElement(els.infoNotice);
      }
      renderProbabilities(data.class_probabilities || data.probabilities, els.probEls, state);
      renderFeatures(data.top_features, els.featEls, state);
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
      renderProbabilities(data.class_probabilities || data.probabilities, els.probEls, state);
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
      renderProbabilities(data.class_probabilities || data.probabilities, els.probEls, state);
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
        const extraNote =
          data.model_routing === "url_malware_preempted"
            ? " Critical: Malware distribution detected by URL classifier. Multimodal classification preempted for safety."
            : " Critical: Malware distribution or C2 threat detected. Do not download or execute files from this host.";
        els.infoNotice.textContent = extraNote.trim();
        els.infoNotice.classList.add("notice-malware");
        showElement(els.infoNotice);
      }
      renderProbabilities(data.class_probabilities || data.probabilities, els.probEls, state);
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
      renderProbabilities(data.class_probabilities || data.probabilities, els.probEls, state);
      renderFeatures(data.top_features, els.featEls, state);
      break;
    }
  }

  // Render diagnostic tri-model inspection if available
  renderMultiModelBreakdown(data, els);

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
  const el = errorEl || (typeof document !== "undefined" ? document.getElementById("error-msg") : null);
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

      // Capture visible tab screenshot on user action (PNG base64 Data URL)
      let screenshotBase64 = null;
      try {
        screenshotBase64 = await captureVisibleTabScreenshot();
      } catch (err) {
        console.warn("[PhishGuard] Tab screenshot capture failed:", err);
        screenshotBase64 = null;
      }

      const requestPayload = { url: activeUrl };
      if (screenshotBase64) {
        requestPayload.screenshot_base64 = screenshotBase64;
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

      try {
        const response = await fetch(API_ENDPOINT, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestPayload),
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
          err.message &&
          (err.message.includes("Failed to fetch") ||
            err.message.includes("NetworkError") ||
            err.message.includes("net::ERR_CONNECTION_REFUSED"))
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
    captureVisibleTabScreenshot,
    classifyPredictionState,
    formatPercent,
    formatCategoryLabel,
    formatRoutingLabel,
    formatVisualStatusLabel,
    formatSupplementaryCategory,
    renderProbabilities,
    renderFeatures,
    renderMultiModelBreakdown,
    renderResult,
    showElement,
    hideElement,
    showError,
  };
}
