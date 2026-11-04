# PhishGuard – Comprehensive Final System Audit Report

**Audit Date**: November 2026  
**Audited Architecture**: Chrome Extension (MV3) → FastAPI (REST) → Canonical URL Normalization → 20 Lexical Features → Three-Class Random Forest → TreeSHAP Explainability → SQLite Database & Automated Migrations → Streamlit Analytics Dashboard.

---

## Executive Summary

A comprehensive, end-to-end system audit was performed across all six architectural tiers of PhishGuard. The active production pipeline successfully classifies URLs into three discrete threat categories (**0 = Legitimate**, **1 = Phishing**, **2 = Malware Distribution**), provides real-time TreeSHAP local feature attributions with polarity badges, guarantees zero domain leakage between training and evaluation partitions, and persists all analysis details to SQLite for historical inspection in the Streamlit dashboard and Chrome Extension.

Importantly, this audit definitively verified the architectural boundary between the active **URL-based detection pipeline** and the **experimental visual feature extraction branch**:
* The production classifier (`ml/models/url_model.joblib`), FastAPI analyzer, TreeSHAP explainer, Chrome Extension, and Streamlit dashboard operate **strictly on the 20 canonical URL lexical features**.
* Visual screenshot feature extraction (`ml/features/visual_features.py`) and feature fusion (`ml/features/feature_fusion.py`) exist as **standalone, experimental unit-tested modules** and are **not** currently integrated into the production training or inference pipeline.

---

## 1. Audit Findings by Category

### 1. Dataset Integrity, Provenance, Labels & Class Mapping
* **Raw Sources**:
  * Legitimate: PhiUSIIL dataset (`datasets/raw/phiusiil.csv`), UCI ID 967. Label mapping correctly inverted: PhiUSIIL 1 → PhishGuard 0 (`legitimate`), PhiUSIIL 0 → PhishGuard 1 (`phishing`).
  * Phishing: PhishTank verified online feeds (`datasets/raw/phishing.csv`) + Kaggle threat intelligence feed (`datasets/raw/kaggle_urls.csv`).
  * Malware Distribution: Kaggle malware feeds (`datasets/raw/kaggle_urls.csv`) + URLhaus.
* **Leakage Prevention**: Registered apex domain grouping (`split_by_registered_domain`) guarantees **0.00% domain overlap** between train (245,673 samples, 152,703 domains) and test (61,417 samples, 36,756 domains).
* **Provenance**: SHA-256 fingerprinting recorded in `ml/evaluation/url_model_eval.json` (`dataset_sha256 = 990c5d52260110a3ccf935fccce0480261c7ef5fc7075b232437cc3bdf1d04a0`).
* **Class Mapping**: Strict consistency across `ml/models/class_mapping.json`, backend schemas, and database:
  * `0: "legitimate"`
  * `1: "phishing"`
  * `2: "malware_distribution"`

### 2. Training-Time vs. Inference-Time Feature Extraction Consistency
* **Canonical Feature Ordering**: Both training (`ml/training/train_url_model.py`) and inference (`backend/services/analyzer.py`) evaluate the identical 20 canonical features in `FEATURE_NAMES`.
* **Normalization Consistency**:
  * Training normalized all incoming candidate URLs via `canonical_normalize_url(url)`, which strips root trailing slashes (`https://example.com/` → `https://example.com`).
  * Inference-time feature extraction now executes `canonical_normalize_url(url)` prior to calling `extract_url_features(url)`. This ensures browser tab URLs containing trailing root slashes do not cause false-positive slash count inflation (`num_slashes = 2` instead of 3; `path_length = 0` instead of 1).

### 3. Three-Class Model Loading & Prediction
* **Model Artefact**: `ml/models/url_model.joblib` (RandomForestClassifier, `n_estimators=100`, `n_features_in_=20`, `classes_=[0, 1, 2]`).
* **Performance**: Test accuracy = 93.15%, Macro F1 = 93.32%, Weighted F1 = 93.14%.
* **Lifecycle**: `ModelManager` safely pre-loads the model and explainer at FastAPI startup, checks `n_features_in_ == 20`, and handles absent/corrupt model files gracefully by returning `analysis_status: "model_unavailable"` without synthetic predictions.

### 4. Probability Normalization & Risk Score Calculation
* **Probabilities**: Extracted from `clf.predict_proba()` and normalized to sum to $1.0 \pm 0.02$.
* **Risk Score**: Calculated as overall malicious probability:
  $$\text{Risk Score} = 1.0 - P(\text{legitimate}) = P(\text{phishing}) + P(\text{malware\_distribution})$$
* **Integrity**: When model is unavailable, `prediction`, `threat_category`, `risk_score`, and `class_probabilities` are set to `None`; no random or synthetic risk scores are ever invented.

### 5. SHAP Contribution Correctness & Explanations
* **Engine**: TreeSHAP via `shap.TreeExplainer(model)`.
* **Dynamic Target Class**: Explains the predicted class (or explicit target class).
* **Category Distinctions**:
  * Class 0 (Legitimate): Never uses threat language; highlights HTTPS protocol, standard TLDs, and lack of credential lures.
  * Class 1 (Phishing): Explains brand lures, deceptive keywords, and subdomain depth.
  * Class 2 (Malware Distribution): Highlights raw IP hosting, non-standard ports, and executable patterns.
* **Badge Polarity**:
  * Risk-increasing contributions ($> 0$ for threats) are labeled `+Risk` / `+Malware`.
  * Risk-reducing / mitigating contributions ($< 0$ for threats, e.g. HTTPS present) are labeled `-Safe`.

### 6. URL Normalization & False-Positive Regression Tests
* **Resolved Regression**: Verified that root URLs with trailing slashes (`https://www.wikipedia.org/`, `https://www.google.com/`, `https://www.microsoft.com/`, `https://www.github.com/`) evaluate as Legitimate (Class 0) with risk score $< 10\%$.
* **Automated Tests**: Explicit test suite in `tests/test_analysis_api.py` (`TestWikipediaAndLegitimateURLRegression`) and `tests/test_extension.py` validates benign URLs with trailing slashes and confirms threat fixture URLs remain accurately flagged.

### 7. FastAPI Endpoints, Schemas & Error Handling
* **Endpoints**:
  * `POST /analyze`: Extracts features, classifies threat category, computes SHAP attributions, persists to SQLite, and returns `AnalysisResponse`.
  * `GET /analyses`: Paginated historical list of analyses ordered newest first.
  * `GET /analyses/{id}`: Single record lookup returning complete probability and explainability JSON.
  * `GET /health` & `GET /`: Health and version status probes.
* **Validation**: Pydantic `AnalyzeRequest` strictly validates HTTP/HTTPS schemes and valid hostnames, rejecting `ftp://`, `chrome://`, `file://`, and `javascript:`.

### 8. SQLite Persistence, Migrations & Historical Records
* **Database**: `data/phishguard.db` (SQLite).
* **Automated Migration**: `backend/database/session.py` inspects the table schema at startup and dynamically applies missing columns (`threat_category`, `class_probabilities`, `top_features`, `explanation`).
* **Compatibility View**: Automatically creates/refreshes the SQL view `analyses AS SELECT * FROM analysis`.
* **Historical Compatibility**: Legacy records with null explanation or probability fields are retrieved without schema errors and hydrated with backward-compatible defaults.

### 9. Chrome Manifest V3 Permissions & API Integration
* **Manifest**: `extension/manifest.json` specifies `"manifest_version": 3`.
* **Permissions**: Least-privilege configuration: `"permissions": ["activeTab"]` and `"host_permissions": ["http://127.0.0.1:8000/*"]`.
* **API Integration**: Connects directly to `POST http://127.0.0.1:8000/analyze` using `fetch` with a 12-second `AbortController` timeout.

### 10. Extension UI States for All Three Prediction Classes
* **Legitimate (Class 0)**: Displays `Safe` badge, `.status-safe` green styling, risk score, and green indicator badges.
* **Phishing (Class 1)**: Displays `Danger` badge, `.status-danger` red styling, high risk score, credential warning, and threat indicators.
* **Malware Distribution (Class 2)**: Displays `Critical` badge, `.status-malware` purple styling, critical risk score, execution warning, and malware indicators.
* **Unavailable / Offline**: Displays `Unavailable` warning notice, `.status-warn` amber styling, and diagnostic instructions to start the backend.

### 11. Dashboard Statistics, Charts, Filters & Historical Inspection
* **Metrics**: Separate counters for Total, Legitimate, Phishing, Malware Distribution, and Total Threats.
* **Visuals**: Three-Class Distribution bar chart and daily activity timeline line chart.
* **Filtering**: Substring URL search, Threat Category dropdown, and Analysis Status dropdown.
* **Inspection Station**: Interactive selector displaying 3-class probability progress bars, risk score, and SHAP top indicators with polarity badges.
* **Persistence**: Connects directly to FastAPI/SQLite without duplicating or inventing data.

### 12. CORS, Security & Sensitive Data Handling
* **CORS**: Configured with `allow_origin_regex=r"^chrome-extension://[a-z0-9]+$"` to properly match browser-generated extension IDs, plus explicit origins for localhost dev (`3000`, `5173`, `8000`, `8501`).
* **Data Privacy**: No credentials, passwords, or session cookies are recorded or inspected; only URL strings and analytical features are processed and stored.

### 13. Dependency Versions & Requirements Consistency
* **Environment**: Python 3.14.3, Node.js v24.21.0.
* **Requirements Alignment**: `requirements.txt` has been updated to include `anyio>=4.0.0` for asynchronous test execution. Uninstalled legacy dependencies (`alembic`, `xgboost`, `tldextract`, `plotly`) documented.

### 14. Project Structure & Documentation
* **Directories**:
  * `backend/`: FastAPI app, SQLite ORM models, Pydantic schemas, and analyzer service.
  * `extension/`: Chrome MV3 extension assets, icons, HTML, CSS, and vanilla JS.
  * `ml/`: Feature engineering (`url_features.py`, `visual_features.py`, `feature_fusion.py`), training pipeline, model artefacts, and SHAP explainer.
  * `dashboard/`: Streamlit security analytics application.
  * `datasets/`: Raw feeds (`phiusiil.csv`, `phishing.csv`, `kaggle_urls.csv`, `tranco.csv`) and processed dataset (`processed_urls.csv`).
  * `docs/`: Provenance metadata, research documents, and audit logs.
  * `tests/`: 337 pytest tests and 10 Node.js extension tests.

---

## 2. Investigation of the Visual Analysis Branch

**Audit Objective**: Verify whether visual screenshot analysis is actively contributing to predictions.

### Architectural Inspection Findings
1. **Module Implementation**:
   * `ml/features/visual_features.py` extracts 30 numerical visual features (dimensions, brightness, BGR color distributions, edge density, image entropy, dominant color clusters).
   * `ml/features/feature_fusion.py` provides `fuse_features()` to concatenate 20 URL features + `has_visual_features` indicator + 30 visual features into a 51-element vector.
2. **Training Pipeline Integration**:
   * In `ml/training/dataset_pipeline.py`, feature extraction is invoked as:
     ```python
     feats = extract_url_features(rec["url"])
     ```
     Visual features are **not** extracted or included in `datasets/processed/processed_urls.csv`.
   * In `ml/training/train_url_model.py`, the model is fitted strictly on the 20 URL features:
     ```python
     clf.fit(X_train, y_train)  # X_train.shape[1] == 20
     ```
3. **Model Artefact**:
   * `ml/models/url_model.joblib` has `n_features_in_ = 20`. Passing a 51-feature fused vector raises a `ValueError: X has 51 features, but RandomForestClassifier is expecting 20 features`.
4. **SHAP Explainer Constraints**:
   * `ml/explainability/explainer.py` explicitly validates:
     ```python
     if len(features) > expected_len:
         raise ValueError(
             "Input has features > 20. Multimodal explainability is not supported "
             "until a compatible multimodal model exists."
         )
     ```
5. **Inference & Clients**:
   * Neither `backend/services/analyzer.py`, `extension/popup.js`, nor `dashboard/app.py` take screenshot inputs or invoke visual extraction.

### Conclusion on Visual Branch
* **Status**: **Standalone Experimental Prototype**.
* Visual features are **NOT** contributing to live model predictions.
* PhishGuard is currently a **20-feature Lexical URL Three-Class Detection System**. To become truly multimodal, future iterations would require:
  1. A dataset containing paired URLs and webpage screenshots.
  2. Training a multimodal Random Forest / XGBoost model on the 51 fused features.
  3. Chrome extension screenshot capture via `chrome.tabs.captureVisibleTab` transmitting base64 image data to `POST /analyze`.

---

## 3. Issue Registry & Classification

| Priority | Issue Description | Component | Status | Remediation / Mitigation |
|---|---|---|---|---|
| **Low** | Missing `anyio` in `requirements.txt` | Environment | **FIXED** | Added `anyio>=4.0.0` to `requirements.txt` under `# Testing`. |
| **Low** | Unused dependencies in `requirements.txt` (`alembic`, `xgboost`, `tldextract`, `plotly`) | Dependencies | Documented | Keep as optional or remove in production container lockfiles. |
| **Low** | Temporary `test_fail.db` in repository root | Repository | **FIXED** | Removed temporary SQLite file; verified `.gitignore` contains `*.db`. |
| **Medium** | PhiUSIIL legitimate dataset contains only apex/subdomains without paths | Dataset | Documented | Canonical normalization mitigates root trailing slashes; future model retraining should incorporate benign deep paths. |
| **Medium** | Multimodal feature fusion not integrated into active model | Architecture | Clarified | Documented project accurately as a 20-feature URL classifier with an experimental vision branch. |

---

## 4. Full Automated Test Results

* **Pytest Suite (`python -m pytest`)**:
  * **337 passed** in 48.96s (14 test modules, 0 failures, 0 warnings).
* **Node.js Extension Suite (`node tests/test_extension_script.js`)**:
  * **10 passed** (0 failures).
* **Total Automated Tests**: **347 passed**.

---

## 5. Deployment Readiness Assessment

* **Overall Status**: **PRODUCTION READY for Lexical URL Analysis (Three-Class)**.
* **Backend**: FastAPI runs cleanly, serves CORS preflight and POST requests to Chrome extensions, and provides full JSON persistence in SQLite.
* **Extension**: Manifest V3 compliant, displays all 3 threat classes with distinct color states, handles timeouts/errors gracefully, and correctly categorizes SHAP risk-increasing vs. risk-reducing indicators.
* **Dashboard**: Full 3-class analytics, category/status filtering, and interactive SHAP inspection station verified against live API responses.
* **Recommendation**: When publishing marketing or academic summaries, describe PhishGuard as an **"AI-Powered Three-Class URL Threat Detection System with Experimental Multimodal Vision Capabilities"** to maintain scientific and technical accuracy.

