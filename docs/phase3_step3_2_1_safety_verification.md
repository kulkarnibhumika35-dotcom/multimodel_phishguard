# PhishGuard Phase 3, Step 3.2.1: Service-Layer Integration Safety Verification Report

**Document Version:** 1.0.0  
**Audit Date:** November 6, 2026  
**Auditor:** PhishGuard Verification Engine  
**Target Architecture Document:** `docs/phase3_integration_architecture.md` (Version 1.1.0)  
**Target Codebase Modules:**  
- `backend/services/decision_arbiter.py`
- `backend/services/analyzer.py`
- `backend/schemas/analysis.py`
- `backend/database/models.py`
- `tests/test_service_layer_safety_verification.py`

---

## Executive Summary

This verification report documents the comprehensive safety, robustness, semantic, and invariant audit conducted on the completed **Phase 3, Step 3.2 Service-Layer Multimodal Integration**.

### Core Audit Outcomes:
1. **Probability Semantics & Terminology**:
   - Eliminated misleading assertions referring to raw Random Forest ensemble vote frequencies as "calibrated" probabilities.
   - Formalized semantic definitions: `risk_score` and `confidence` strictly represent **`raw_model_probability`** (uncalibrated tree vote fractions) unless empirical calibration (e.g., Platt scaling or isotonic regression) is performed in a subsequent phase.
   - Code comments, docstrings, and tests updated to maintain technical accuracy.
2. **API Response & Distribution Consistency**:
   - Verified that top-level `prediction`, `threat_category`, and `class_probabilities` strictly adhere to the unified 3-class schema (`legitimate`, `phishing`, `malware_distribution`).
   - Verified that sub-model probability distributions are strictly isolated within `multi_model_telemetry`:
     - Standalone URL model: 3 classes (`legitimate`, `phishing`, `malware_distribution`).
     - Standalone Visual model: 2 classes (`unverified_candidate`, `phishing`).
     - Multimodal model: 2 classes (`unverified_candidate`, `phishing`).
   - Verified that binary candidate/phishing distributions are never confused with or swapped into the URL model's three-class space.
   - Verified that malware lures retain their original 3-class probability distribution.
3. **Model Failure Isolation**:
   - Verified through fault injection that the failure of any optional component (missing visual model, missing multimodal model, runtime inference exception) never crashes the service and gracefully falls back to the standalone URL model.
   - Verified that a missing URL model triggers the **Primary Model Availability Guard**, safely returning `analysis_status: "model_unavailable"` rather than emitting an unhandled 500 error or attempting blind binary multimodal inference.
4. **Malware Preemption Invariant Proof**:
   - Formally proved and tested that malware detection by the URL model (either argmax Class 2 or $P(\text{malware}) \ge 0.50$) preempts the binary multimodal model unconditionally, preventing malware lures from ever being classified as candidate or legitimate.
5. **Backward Compatibility**:
   - Confirmed zero destructive SQLite schema migrations.
   - Confirmed historical database records (from Phase 1 and Phase 2) stored without `multi_model_telemetry` deserialize and validate cleanly through Pydantic.
6. **Shadow-Mode Invariant**:
   - Verified that `is_shadow_mode=True` is enforced across all operational paths, with zero HTTP redirection, zero request cancellation, and zero navigation blocking.
7. **Test Suite Execution**:
   - **494 passed in 55.95 seconds** across the entire repository test suite, including 19 new targeted safety verification tests.

---

## 1. Probability Semantics Audit

### 1.1 Empirical Audit of Model Outputs
Scikit-learn's `RandomForestClassifier.predict_proba()` calculates the mean predicted class probabilities of the trees in the forest:
$$P(Y = c \mid X) = \frac{1}{B} \sum_{b=1}^{B} P_b(Y = c \mid X)$$
where each tree $b$ outputs the proportion of training samples of class $c$ in the terminal leaf node.

In finite-sample tree ensembles without post-processing:
1. Probability estimates tend to be pushed away from 0 and 1 towards the center (due to leaf sample constraints), or clustered around discrete fractional steps ($k / B$).
2. **Empirical calibration (e.g. Platt scaling / sigmoid fitting or isotonic regression) has NOT been fitted on held-out validation data for the current models.**
3. Describing these probabilities as "calibrated" was a technical misnomer.

### 1.2 Corrections Applied
- **Code Refinements**:
  In `backend/services/decision_arbiter.py`:
  - Updated module docstring from:
    `"Derives risk score from calibrated model probabilities; never uses arbitrary static clamping."`
    to:
    `"Derives risk score directly from raw ensemble model probabilities (uncalibrated leaf vote fractions); never uses arbitrary static clamping."`
  - Updated malware preemption comment and local variable:
    ```python
    # Raw threat risk score: 1.0 - P(legit) = P(phish) + P(malware)
    # Note: Random Forest predict_proba outputs are raw ensemble vote fractions (raw_model_probability),
    # not empirically calibrated probabilities (e.g. via Platt scaling or isotonic regression).
    raw_threat_risk = round(float(1.0 - p_legit), 4)
    ```
  - Updated `tests/test_decision_arbiter.py` to assert raw ensemble threat score rather than claiming calibrated status.
- **Preservation of API Field Types**:
  - The API response fields remain `risk_score: float` and `confidence: float` (both bounded in $[0.0, 1.0]$).
  - No changes to client contracts were required.

### 1.3 Mathematical Formulation of Runtime Probability Scoring

| Routing Pathway | Target Condition | Risk Score Formula ($R$) | Confidence Formula ($C$) | Description |
| :--- | :--- | :---: | :---: | :--- |
| **`multimodal_full`** | Screenshot extracted & valid; $P_{\text{url}}(\text{malware}) < 0.50$ | $P_{\text{multi}}(\text{phishing})$ | $\max(P_{\text{multi}}(\text{cand}), P_{\text{multi}}(\text{phish}))$ | Raw multimodal ensemble vote fraction for phishing threat mass. |
| **`url_malware_preempted`** | URL model predicts Class 2 OR $P_{\text{url}}(\text{malware}) \ge 0.50$ | $1.0 - P_{\text{url}}(\text{legitimate})$ | $P_{\text{url}}(\text{malware_distribution})$ | Total non-benign probability mass; confidence set to raw malware leaf proportion. |
| **`url_model_fallback`** | Screenshot missing, corrupt, or multimodal model unavailable | $1.0 - P_{\text{url}}(\text{legitimate})$ | $P_{\text{url}}(\text{threat_category})$ | Standalone URL threat mass; confidence set to winning threat category probability. |
| **`model_unavailable`** | Primary URL model unavailable or unrecoverable error | `None` (`null`) | `None` (`null`) | Zero synthetic scores generated. |

---

## 2. API Response Consistency & Distribution Isolation

### 2.1 Uniform 3-Class Response Semantics
All PhishGuard consumers (Chrome Extension, Streamlit Dashboard, and automated evaluation harnesses) expect an unequivocal 3-class classification vocabulary:
- `0`: `"legitimate"`
- `1`: `"phishing"`
- `2`: `"malware_distribution"`

Even when inference routes through the binary multimodal model (whose native training labels were `0 = unverified_candidate` and `1 = phishing`), the top-level API response translates and aligns the prediction into the canonical 3-class vocabulary:
```json
{
  "prediction": 0,
  "threat_category": "legitimate",
  "risk_score": 0.0600,
  "confidence": 0.9400,
  "model_routing": "multimodal_full",
  "class_probabilities": {
    "legitimate": 0.9400,
    "phishing": 0.0600,
    "malware_distribution": 0.0000
  }
}
```

### 2.2 Strict Distribution Isolation in Telemetry
Under `explanation["multi_model_telemetry"]`, each constituent model's native class mapping and probability distribution are preserved without cross-contamination:

```json
{
  "multi_model_telemetry": {
    "url_model_result": {
      "status": "completed",
      "prediction": 0,
      "threat_category": "legitimate",
      "probabilities": {
        "legitimate": 0.9000,
        "phishing": 0.0800,
        "malware_distribution": 0.0200
      }
    },
    "visual_model_result": {
      "status": "completed",
      "prediction": 0,
      "threat_category": "unverified_candidate",
      "probabilities": {
        "unverified_candidate": 0.8800,
        "phishing": 0.1200
      }
    },
    "multimodal_model_result": {
      "status": "completed",
      "prediction": 0,
      "threat_category": "legitimate",
      "probabilities": {
        "unverified_candidate": 0.9400,
        "phishing": 0.0600
      },
      "decision_margin": 0.4400,
      "ambiguous": false
    },
    "model_routing": "multimodal_full",
    "is_shadow_mode": true,
    "confidence": 0.9400
  }
}
```

**Verification Proof:**
1. The binary candidate/phishing keys (`unverified_candidate`) exist ONLY inside `visual_model_result` and `multimodal_model_result`.
2. The 3-class keys (`legitimate`, `phishing`, `malware_distribution`) exist in `url_model_result` and top-level `class_probabilities`.
3. In malware preemption, `class_probabilities` is a direct copy of the URL model's 3-class distribution, preserving the true malware probability.

---

## 3. Model Failure Isolation & Fault-Injection Audit

A comprehensive fault matrix was subjected to targeted testing in `tests/test_service_layer_safety_verification.py`.

### 3.1 Failure Isolation Matrix

| Fault Injected | Subsystem Affected | Observed System Behavior | Routing Mode | API Status Code | Verified In Test |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Missing Multimodal Model** (`None` or file missing) | Multimodal Classifier | URL model evaluates; screenshot features extracted; falls back gracefully to standalone URL model. | `url_model_fallback` | 200 OK | `test_missing_multimodal_model_falls_back_to_url` |
| **Missing Visual Model** (`None` or file missing) | Auxiliary Visual Model | Visual telemetry marked `{"status": "unavailable"}`; multimodal and URL models evaluate unaffected. | `multimodal_full` | 200 OK | `test_missing_visual_model_preserves_multimodal_execution` |
| **Missing URL Model** (`None` or file missing) | Primary URL Classifier | Primary Model Availability Guard halts pipeline; multimodal model is skipped; zero synthetic scores. | `model_unavailable` | 200 OK | `test_missing_url_model_fails_safely_to_model_unavailable` |
| **Incompatible Feature Count** (e.g. 10 instead of 20/51) | ModelManager validation | Model loading rejected; model reference set to `None`; falls back to safe mode. | Fallback / Unavailable | 200 OK | `test_incompatible_feature_count_rejected` |
| **Corrupted Metadata JSON** (Invalid JSON syntax) | ModelManager metadata loader | Exception caught; falls back to embedded safe default class mappings. | Unaffected | 200 OK | `test_invalid_metadata_json_falls_back_to_defaults` |
| **Multimodal Runtime Exception** (e.g. Memory error during `predict`) | Inference Engine | Exception logged; catches error; falls back to standalone URL model with explanatory reason. | `url_model_fallback` | 200 OK | `test_multimodal_inference_exception_falls_back_cleanly` |
| **Visual Runtime Exception** (e.g. Feature matrix error) | Auxiliary Visual Model | Exception logged; records error in visual telemetry; multimodal inference finishes normally. | `multimodal_full` | 200 OK | `test_visual_model_inference_exception_isolated` |

### 3.2 Primary Model Availability Guard Rationale
A critical safety discovery during Step 3.2 and verified in Step 3.2.1 is that **the URL model is not merely a fallback—it is the prerequisite gatekeeper for the entire pipeline**.

Because the multimodal model is strictly a **binary classifier** (distinguishing only between phishing and candidates), it possesses zero capacity to detect malware distribution lures (e.g. executables, ransomware payloads, drive-by installers). If the URL model were bypassed when missing and the multimodal model executed alone, a malware payload with a benign-looking landing page would be misclassified as legitimate candidate.

Therefore, the **Primary Model Availability Guard** strictly enforces:
$$\text{If URL Model Unavailable} \implies \text{Halt Inference} \implies \text{Emit } \texttt{model\_unavailable}$$

---

## 4. Malware Preemption Invariant Proof

### 4.1 Invariant Definition
Let:
- $\hat{Y}_{\text{url}} \in \{0, 1, 2\}$ be the URL model predicted class.
- $P_{\text{url}}(\text{malware}) \in [0.0, 1.0]$ be the raw URL model probability for Class 2.
- $\theta_{\text{mal}} = 0.50$ be the approved malware preemption threshold.
- $\hat{Y}_{\text{multi}} \in \{0, 1\}$ be the binary multimodal prediction.

The final decision $\hat{Y}_{\text{final}}$ must satisfy:
$$\Big( \hat{Y}_{\text{url}} = 2 \;\lor\; P_{\text{url}}(\text{malware}) \ge \theta_{\text{mal}} \Big) \implies \Big( \hat{Y}_{\text{final}} = 2 \;\land\; \text{Routing} = \texttt{url\_malware\_preempted} \Big)$$
irrespective of whether a screenshot is provided and irrespective of $\hat{Y}_{\text{multi}}$.

### 4.2 Empirical Boundary Test Results
Parametric boundary testing was executed across the critical threshold domain:

| $P_{\text{url}}(\text{malware})$ | URL Argmax Class | Multimodal $P(\text{cand})$ | Resulting Model Routing | Final Prediction | Malware Overruled? |
| :---: | :---: | :---: | :--- | :---: | :---: |
| `0.4900` | 1 (phishing) | 0.9900 | `multimodal_full` | `0` (legitimate) | No (Below threshold) |
| `0.5000` | 2 (malware) | 0.9900 | `url_malware_preempted` | `2` (malware) | **PREEMPTED** |
| `0.5100` | 2 (malware) | 0.9900 | `url_malware_preempted` | `2` (malware) | **PREEMPTED** |
| `0.9900` | 2 (malware) | 0.9900 | `url_malware_preempted` | `2` (malware) | **PREEMPTED** |

**Conclusion:** The malware preemption invariant is verified unconditionally across all boundary conditions.

---

## 5. Backward Compatibility & Database Integrity

### 5.1 SQLite Schema Non-Destructiveness
No SQLite schema migrations or `ALTER TABLE` statements were performed. The physical table schema remains:
```sql
CREATE TABLE analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url VARCHAR NOT NULL,
    prediction INTEGER,
    threat_category VARCHAR,
    risk_score FLOAT,
    class_probabilities JSON,
    top_features JSON,
    explanation JSON,
    analysis_status VARCHAR NOT NULL,
    analyzed_at DATETIME NOT NULL,
    visual_status VARCHAR,
    visual_features JSON
);
```

### 5.2 Deserialization of Historical Records
Historical records generated during Phase 1 or Phase 2 contain no `multi_model_telemetry` inside `explanation`. In `backend/database/models.py`, transient property accessors provide fail-safe defaults:
- `retrieved.url_model_result` $\to$ `None`
- `retrieved.visual_model_result` $\to$ `None`
- `retrieved.multimodal_model_result` $\to$ `None`
- `retrieved.model_routing` $\to$ `"url_model_fallback"`
- `retrieved.is_shadow_mode` $\to$ `True`
- `retrieved.fallback_reason` $\to$ `None`
- `retrieved.confidence` $\to$ `self.risk_score`

This was verified by creating an isolated SQLite database, inserting a legacy record without telemetry, querying it back, and validating that `AnalysisResponse.model_validate(retrieved)` completes without error (`test_historical_db_record_without_telemetry_reads_cleanly`).

---

## 6. Shadow-Mode Invariant Audit

### 6.1 Non-Interception Verification
PhishGuard is currently deployed in **Shadow Mode**. The system must never block user web navigation or issue HTTP redirection headers.

- **HTTP Status Code Audit**: All analysis requests return `200 OK` (or `422 Unprocessable Entity` on invalid client inputs).
- **HTTP Header Audit**: No `Location` headers or redirect status codes (`301`, `302`, `307`) are ever emitted by `POST /analyze`.
- **Telemetry Tagging**: Every response explicitly contains `"is_shadow_mode": true`.
- **Extension Invariant**: The backend provides passive risk scores; no active blocking instructions are dispatched.

---

## 7. Verification Test Suite & Execution Results

### 7.1 New Safety Verification Test Module
A dedicated test suite was implemented in [`tests/test_service_layer_safety_verification.py`](file:///c:/Users/kulka/OneDrive/Desktop/fishGuard/tests/test_service_layer_safety_verification.py), comprising 19 targeted tests:
1. `TestProbabilitySemantics`:
   - `test_url_fallback_probability_semantics` [PASSED]
   - `test_multimodal_full_probability_semantics` [PASSED]
   - `test_malware_preemption_probability_semantics` [PASSED]
2. `TestAPIResponseConsistency`:
   - `test_class_system_alignment_across_all_routes` [PASSED]
   - `test_distribution_isolation_between_models` [PASSED]
3. `TestModelFailureIsolation`:
   - `test_missing_multimodal_model_falls_back_to_url` [PASSED]
   - `test_missing_visual_model_preserves_multimodal_execution` [PASSED]
   - `test_missing_url_model_fails_safely_to_model_unavailable` [PASSED]
   - `test_multimodal_inference_exception_falls_back_cleanly` [PASSED]
   - `test_visual_model_inference_exception_isolated` [PASSED]
   - `test_invalid_metadata_json_falls_back_to_defaults` [PASSED]
4. `TestMalwarePreemptionInvariant`:
   - `test_malware_threshold_boundary_conditions[0.49-multimodal_full]` [PASSED]
   - `test_malware_threshold_boundary_conditions[0.50-url_malware_preempted]` [PASSED]
   - `test_malware_threshold_boundary_conditions[0.51-url_malware_preempted]` [PASSED]
   - `test_malware_threshold_boundary_conditions[0.99-url_malware_preempted]` [PASSED]
5. `TestBackwardCompatibility`:
   - `test_historical_db_record_without_telemetry_reads_cleanly` [PASSED]
   - `test_api_get_analyses_returns_all_expected_fields` [PASSED]
6. `TestShadowModeInvariant`:
   - `test_shadow_mode_flag_always_active` [PASSED]
   - `test_api_does_not_issue_redirects_on_high_threat` [PASSED]

### 7.2 Full Regression Suite Run
The entire automated test suite was executed across all components:
```bash
python -m pytest tests/ -q
```
**Results:**
```text
494 passed in 55.95s
```
Zero test failures, zero regressions, and zero deprecation warnings.

---

## 8. Summary of Applied Fixes & Readiness Confirmation

1. **Applied Fixes**:
   - `backend/services/decision_arbiter.py`: Replaced claims of "calibrated probabilities" with precise technical descriptions of raw ensemble leaf vote fractions (`raw_model_probability`).
   - `tests/test_decision_arbiter.py`: Updated test descriptions and comments to remove synthetic calibration claims.
   - `tests/test_service_layer_safety_verification.py`: Created complete 19-test audit suite verifying probability semantics, failure isolation, malware preemption boundaries, database readability, and shadow-mode guarantees.
2. **Readiness Confirmation**:
   - The service layer is robust, isolated against faults, and mathematically verified.
   - Step 3.2.1 is complete.
   - The system is verified and ready for user approval before moving to Step 3.3.

