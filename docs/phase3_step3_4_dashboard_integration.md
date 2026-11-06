# PhishGuard Phase 3 — Step 3.4: Streamlit Dashboard Multimodal Integration

**Document Version:** 1.0.0  
**Phase:** Phase 3 — Multimodal Production Integration  
**Step:** Step 3.4 — Streamlit Dashboard Multimodal Integration  
**Status:** Completed & Fully Verified  
**Date:** 2026-11-06  

---

## 1. Executive Summary

In Phase 3, Step 3.4, the PhishGuard Streamlit web dashboard (`dashboard/app.py`) was successfully integrated with the Decision Arbiter and FastAPI multi-model analysis system. The dashboard provides end-to-end observability, auditability, and deep inspection for security analysts without compromising safety or data integrity.

### Key Milestones Achieved:
1. **Four-Class Semantics Enforcement**: Fully recognizes four distinct threat classifications:
   - `Legitimate` (Class 0 from 3-class URL model)
   - `Unverified Candidate` (Class 0 from 2-class binary Multimodal/Visual model)
   - `Phishing` (Class 1 across all models)
   - `Malware Distribution` (Class 2 from URL model)
   - `Unverified Candidate` is **never** counted or presented as confirmed `Legitimate`.
2. **Authoritative `threat_category` Ground Truth**: Adheres to the Decision Arbiter's `threat_category` as the authoritative classification.
3. **Strict Probability Distribution Separation**: Preserves separate displays for:
   - URL Model (3-class distribution: Legitimate, Phishing, Malware Distribution)
   - Visual Model (2-class distribution: Unverified Candidate, Phishing)
   - Multimodal Model (2-class distribution: Unverified Candidate, Phishing)
4. **Enhanced KPI Overview**: Surfaces 8 comprehensive high-level metrics including total analyses, 4-class counts, average raw risk score, screenshot capture rate, and multimodal vs. fallback routing volumes.
5. **Interactive Visual Analytics (6 Charts)**:
   - 4-Class Classification Distribution (Bar chart with semantic color coding)
   - Risk Score Tier Distribution (Histogram across Low, Moderate, Medium, High, Critical)
   - Daily Analysis Activity Timeline (Chronological trend line)
   - Model Routing Pipeline Distribution (Multimodal Full vs. URL Fallback vs. Preemption)
   - Screenshot Availability Breakdown (Extracted vs. Not Provided vs. Failed)
   - Sub-Model Disagreement Statistics (URL model vs. Multimodal model alignment)
6. **Detailed Multi-Model Inspection Station**:
   - URL, Visual, and Multimodal sub-model verdict badges and raw probabilities
   - Routing decision reason and fallback explanation
   - Visual features inspection table (30 canonical dimensions)
   - Direction-aware TreeSHAP feature contributions (Risk Factor vs. Candidate Factor vs. Legitimate Factor)
7. **Persistent Shadow-Mode Banners**:
   - Prominent top-level banner: `"🛡️ PhishGuard Shadow Mode: Predictions are informational and do not block navigation."`
   - Explicit experimental model disclaimer regarding missing-modality generalization.
8. **100% Backward Compatibility**: Seamlessly loads historical database records lacking multimodal telemetry or `threat_category` fields, safely defaulting to `url_model_fallback`.
9. **Rigorous Verification**: 82 dashboard tests passing (100%), 535 repository pytest tests passing (100%), and 20 Chrome extension node tests passing (100%).

---

## 2. Four-Class Semantics & Candidate Isolation

### 2.1 The Semantic Challenge
The URL model operates in a 3-class domain (`legitimate`, `phishing`, `malware_distribution`), whereas the visual and multimodal models operate in a 2-class domain (`unverified_candidate`, `phishing`). In previous iterations, non-phishing predictions were informally grouped as "safe" or "legitimate".

In Step 3.4, `unverified_candidate` is strictly isolated as a fourth independent category.

| Semantic Category | Originating Pipeline | Raw Model Class | Display Badge | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| **Legitimate** | URL Model Fallback | Class `0` | `🟢 Legitimate` | Verified benign domain analyzed via 20 URL features. |
| **Unverified Candidate** | Multimodal Full Pipeline | Class `0` | `🔵 Unverified Candidate` | Benign candidate under multimodal evaluation; unverified registered domain. |
| **Phishing** | Multimodal or URL Model | Class `1` | `🔴 Phishing` | Confirmed credential theft or phishing site. |
| **Malware Distribution** | URL Safeguard Preemption | Class `2` | `🟣 Malware Distribution` | Malicious file hosting, raw IP executable, or dropper. |
| **Unavailable** | Error / Timeout | `None` | `⚪ Model Unavailable` | Backend service failure or model missing; non-blocking. |

### 2.2 Mathematical Isolation in `compute_statistics`
`compute_statistics()` in `dashboard/app.py` enforces absolute separation between `unverified_candidate` and `legitimate`:
```python
if cat in ("unverified_candidate", "candidate") or (pred in (0, "0") and routing == "multimodal_full"):
    unverified_candidate += 1
elif cat in ("legitimate", "0") or pred in (0, "0"):
    legitimate += 1
```
An unverified candidate never increments the legitimate counter under any circumstance.

---

## 3. High-Level KPI Overview & Model Routing

The top of the dashboard displays 8 critical operational metrics:

```
┌─────────────────┬─────────────────┬─────────────────────────┬──────────────────────┐
│ Total Analyses  │ Legitimate      │ Unverified Candidates   │ Phishing Detected    │
│      1,248      │      812        │           240           │         156          │
├─────────────────┼─────────────────┼─────────────────────────┼──────────────────────┤
│ Malware Preempt │ Avg Risk Score  │ Screenshot Success Rate │ Multimodal Routing   │
│       40        │     0.1824      │         84.2%           │   920 Full / 328 Fall│
└─────────────────┴─────────────────┴─────────────────────────┴──────────────────────┘
```

- **Threat Total**: Calculated strictly as `phishing + malware_distribution`.
- **Average Raw Risk Score**: Arithmetic mean of `risk_score` for all completed analyses, explicitly annotated as raw model probability (not calibrated).
- **Screenshot Success Rate**: Percentage of analyses where screenshot visual features were successfully extracted (`visual_status == "extracted"`).
- **Pipeline Utilization**: Direct count and ratio of `multimodal_full` (51 fused features) vs. `url_model_fallback` (20 URL features).

---

## 4. Visual Analytics Suite (6 Charts)

All 6 charts are backed by pure data-preparation helpers that safely handle empty database states:

### 4.1 Threat Category Distribution
- **Helper**: `prepare_four_class_distribution(records)`
- **Display**: Bar chart illustrating the volume of Legitimate, Unverified Candidate, Phishing, and Malware Distribution.
- **Safety**: Empty records yield a 4-row zero-count DataFrame to prevent Altair/Streamlit rendering exceptions.

### 4.2 Raw Model Risk Score Tiers
- **Helper**: `prepare_risk_score_distribution(records)`
- **Tiers**:
  - `0.0 - 0.2 (Low)`
  - `0.2 - 0.4 (Moderate)`
  - `0.4 - 0.6 (Medium)`
  - `0.6 - 0.8 (High)`
  - `0.8 - 1.0 (Critical)`
- **Semantics**: Ignores `model_unavailable` records and missing risk scores.

### 4.3 Daily Analysis Activity Timeline
- **Helper**: `prepare_daily_activity(records)`
- **Display**: Line chart tracking analysis volume over time, parsing ISO timestamps with UTC/Z support.

### 4.4 Model Routing Pipeline Distribution
- **Helper**: `prepare_model_routing_distribution(records)`
- **Breakdown**:
  - `Multimodal Full (51 Feat)`
  - `URL Fallback (20 Feat)`
  - `Malware Preempted`
  - `Model Unavailable`

### 4.5 Screenshot Availability Breakdown
- **Helper**: `prepare_screenshot_availability(records)`
- **Breakdown**:
  - `Captured (Extracted)`
  - `Not Provided`
  - `Capture Failed`
  - `Invalid Image`

### 4.6 Sub-Model Disagreement Statistics
- **Helper**: `prepare_model_disagreement_statistics(records)`
- **Metrics**:
  - `Agreement Rate`: Proportion of multimodal analyses where URL model and Multimodal model agree on threat status (both class 0 or both class 1).
  - `Disagreement Rate`: Proportion of conflicting verdicts.
  - `Disagreement Breakdown`:
    - `URL Legit vs Multi Phish`: URL model predicted Legitimate, but Multimodal model detected Phishing.
    - `URL Phish vs Multi Candidate`: URL model predicted Phishing, but Multimodal model predicted Candidate.
    - `Malware Preemption`: URL model detected Malware Distribution and preempted the Multimodal model.

---

## 5. Recent Analyses Table (10 Columns)

The historical log table was expanded to 10 structured columns:

| Column Header | Format / Example | Description |
| :--- | :--- | :--- |
| **ID** | `42` | Unique SQLite record ID. |
| **URL** | `https://example.com/login` | Full target URL analyzed. |
| **Prediction** | `🔵 Unverified Candidate` | Human-readable badge with semantic icon. |
| **Threat Category** | `unverified_candidate` | Machine-readable canonical category name. |
| **Risk Score** | `0.0600` | Raw uncalibrated model probability (4 decimal places). |
| **Model Routing** | `⚡ Multimodal (51 Feat)` | Routing pipeline executed (`Multimodal`, `URL Fallback`, `Malware Preemption`). |
| **Screenshot Status** | `📸 Extracted` | Screenshot ingestion and feature extraction outcome. |
| **Shadow Mode** | `Active 🛡️` | Explicit indicator confirming passive, non-blocking telemetry. |
| **Status** | `completed` | Pipeline execution status (`completed`, `model_unavailable`, `failed`). |
| **Analyzed At** | `2026-11-06 11:20:00 UTC` | Formatted UTC timestamp. |

---

## 6. Detailed Multi-Model Inspection Station

When a user selects a specific record from the dashboard, a comprehensive inspection station is rendered:

### 6.1 Sub-Model Probabilities & Decision Trace
Displays separate, non-confused probability distributions side-by-side:

```
┌─────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
│ URL Model (3-Class)             │ Visual Model (2-Class)          │ Multimodal Model (2-Class)      │
│ Verdict: 🟢 Legitimate          │ Verdict: 🔵 Unverified Candidate│ Verdict: 🔵 Unverified Candidate│
├─────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Legitimate: 94.0%               │ Unverified Candidate: 89.2%     │ Unverified Candidate: 93.5%     │
│ Phishing: 5.8%                  │ Phishing: 10.8%                 │ Phishing: 6.5%                  │
│ Malware: 0.2%                   │ (Binary Classification)         │ (51 Fused Features)             │
└─────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
```

### 6.2 Visual Features Expander
For multimodal analyses with `visual_status == "extracted"`, the dashboard renders all 30 extracted visual features in a clean tabular view:
- Geometry: `width`, `height`, `aspect_ratio`, `file_size_bytes`
- Color & Tone: `mean_brightness`, `std_brightness`, `rms_contrast`, `grayscale_entropy`
- Edges & Complexity: `canny_edge_density`, `laplacian_variance`, `sobel_edge_density`
- Structure: `dominant_colors_count`, `color_palette_entropy`, `hsv_saturation_mean`

### 6.3 Explainable AI (TreeSHAP) Polarity Badges
Feature importance badges strictly reflect model semantics:
- **Red Badges (`▲ Risk Factor`)**: Features with positive contributions increasing phishing or malware risk.
- **Sky-Blue Badges (`▼ Candidate Factor`)**: Features reducing risk in multimodal analyses (never labeled "Safe").
- **Green Badges (`▼ Legitimate Factor`)**: Features supporting legitimate classification in URL-only analyses.

---

## 7. Safety, Shadow-Mode Banners & Disclaimers

### 7.1 Persistent Shadow-Mode Banner
Positioned immediately beneath the main page title:
> **🛡️ PhishGuard Shadow Mode Active**: All predictions and risk scores are informational only. The extension and dashboard operate in passive audit mode and **do not block browser navigation or intercept network requests**.

### 7.2 Experimental Multimodal Model Notice
Displayed within the Visual Analytics expander:
> **⚠️ Experimental Multimodal Model Notice**: The multimodal classifier was trained exclusively on pairs where screenshots were successfully captured (`has_visual_features = 1`). In production, missing screenshots fall back to the standalone 20-feature URL classifier (`url_model.joblib`), which preserves 3-class malware preemption. Zero-masked visual feature ablation is treated as experimental research, not a production fallback.

---

## 8. Backward Compatibility Verification

The dashboard was tested against legacy and partial records:
1. **Pre-Multimodal Historical Records**: Records with only URL features, lacking `model_routing`, `visual_status`, or `multi_model_telemetry`.
   - Handled gracefully: `model_routing` defaults to `"url_model_fallback"`, and `visual_status` defaults to `"not_provided"`.
2. **Missing `threat_category` Field**: Historical records where only `prediction` exists (0, 1, 2).
   - Correctly maps 0 to `Legitimate`, 1 to `Phishing`, and 2 to `Malware Distribution`.
3. **`model_unavailable` Isolation**: Records where analysis failed or backend was offline.
   - Preserves `Unavailable` label, `None` risk scores render as a neutral dash (`-`), and counts are excluded from average risk calculations.

---

## 9. Comprehensive Verification & Test Results

### 9.1 Dashboard Test Suite (`tests/test_dashboard.py`)
**82 Tests Passed / 0 Failed**:
- `TestComputeStatistics` (16 tests): 4-class counts, candidate isolation, threat totals, average risk score, screenshot success rate, fallback counts.
- `TestPredictionLabel` (9 tests): Formatting across 4 classes, string/int inputs, model unavailable states.
- `TestThreatCategoryLabel` (1 test): Label parity across all valid categories.
- `TestRiskDisplay` (4 tests): Normal formatting, zero risk, model unavailable dash representation.
- `TestFormatTimestamp` (4 tests): UTC formatting, ISO string parsing, null handling.
- `TestFilterRecords` (10 tests): URL query search, threat category filtering, status filtering, routing filtering.
- `TestRecordsToDataFrame` (7 tests): 10-column schema verification, 4-class rows, shadow-mode column.
- `TestValidateURLInput` (6 tests): Client-side URL format validation.
- `TestAnalyzeNewURL` (9 tests): HTTP client interaction with FastAPI backend across legitimate, phishing, malware, and error payloads.
- `TestVisualAnalyticsCharts` (10 tests): All 6 chart data preparation functions with populated and empty states.
- `TestDetailedAnalysisAndRoutingLabels` (2 tests): Routing and visual status badge formatters.
- `TestBackwardCompatibility` (3 tests): Historical records lacking telemetry, missing threat categories, and model unavailable isolation.

### 9.2 Full Repository Regression Test Suite
```
pytest tests/ -q
535 passed in 83.92s
```
Every test in the repository continues to pass without error.

### 9.3 Chrome Extension Node Test Suite
```
node tests/test_extension_script.js
=== All PhishGuard Extension Tests Passed Successfully! (20/20) ===
```

---

## 10. Conclusion and Next Steps

Step 3.4 is 100% complete, fully tested, and verified against all safety guidelines:
- The Streamlit dashboard now provides rich multimodal observability with strict 4-class semantics.
- All three models maintain separate probability displays.
- Unverified candidates are never conflated with confirmed legitimate sites.
- Historical database records are preserved and handled seamlessly.
- Model files, weights, and extension logic were left untouched.

PhishGuard Phase 3, Step 3.4 is ready for production sign-off.

