# PhishGuard – Multimodal Feature Compatibility Audit (Phase 2, Step 2.5)

**Audit Date**: 2026-11-05T17:25:30.440764+00:00  
**Dataset Path**: `C:/Users/kulka/OneDrive/Desktop/fishGuard/datasets/processed/multimodal_features.csv`  
**Dataset SHA-256**: `4a212d3d0902336373e1431a81284c006dce899752a9f99ab3db2b5c725c8ed0`  
**Total Valid Paired Samples**: **1486** (100% paired URL + screenshot)  

> [!WARNING]
> **Data Governance Notice & Provenance Limitation**:  
> CRITICAL PROVENANCE & LABEL COMPATIBILITY NOTICE: 1. The visual dataset legitimate candidate partition originates from Tranco popularity rankings    ('tranco_candidate_unverified', threat_category='unverified_candidate') and is NOT independently    verified benign ground truth. 2. The URL model was trained on PhiUSIIL (with 3 classes: 0=legitimate, 1=phishing, 2=malware_distribution).    The visual model was trained on Zenodo V2.0 (with 2 classes: 0=unverified_candidate, 1=phishing). 3. These classes are NOT interchangeable: treating Tranco candidates as verified legitimate leads to label    incompatibility, as evidenced by lexical domain/path divergence. 4. All multimodal records preserve original label provenance for rigorous auditability.

---

## 1. Feature Order & Schema Verification

| Component | Expected | Observed | Status |
| :--- | :---: | :---: | :---: |
| **Total Features** | **51** | **51** | PASS |
| **URL Features (Idx 0–19)** | 20 features | 51 (first 20) | PASS (Exact Match) |
| **Availability Flag (Idx 20)** | `has_visual_features` | `has_visual_features` | PASS (Index 20) |
| **Visual Features (Idx 21–50)** | 30 features | 51 (last 30) | PASS (Exact Match) |
| **NaN / Inf Values** | 0 | 0 NaN / 0 Inf | PASS (Clean) |
| **Metadata Columns Preserved** | 9 columns | 9 columns | PASS |
| **Total CSV Field Count** | 60 columns | 60 columns | PASS |

### Canonical Feature Ordering Specification (51 Features)

```text
[00..19] URL Lexical Features (20):
         01. url_length               02. hostname_length          03. num_dots
         04. num_hyphens              05. num_underscores          06. num_slashes
         07. num_at                   08. num_question_marks       09. num_ampersands
         10. num_equals               11. num_percent              12. num_digits
         13. num_subdomains           14. is_https                 15. has_ip_address
         16. has_suspicious_tld       17. has_suspicious_keyword   18. path_length
         19. query_length             20. url_entropy

[20]     Modality Availability Indicator (1):
         21. has_visual_features (1 = present & verified, 0 = missing/corrupt)

[21..50] Visual Screenshot Features (30):
         22. width                    23. height                   24. aspect_ratio
         25. mean_brightness          26. std_brightness           27. mean_blue
         28. mean_green               29. mean_red                 30. std_blue
         31. std_green                32. std_red                  33. color_variance
         34. edge_density             35. image_entropy            36. dominant_color_b
         37. dominant_color_g         38. dominant_color_r         39. dominant_color_proportion
         40. dominant_color_1_b       41. dominant_color_1_g       42. dominant_color_1_r
         43. dominant_color_1_proportion                           44. dominant_color_2_b
         45. dominant_color_2_g       46. dominant_color_2_r       47. dominant_color_2_proportion
         48. dominant_color_3_b       49. dominant_color_3_g       50. dominant_color_3_r
         51. dominant_color_3_proportion
```

---

## 2. Model Prediction Interface & Compatibility Audit

### A. URL Model (`ml/models/url_model.joblib`)
- **Input Dimension**: Expected **20** features; received 20 features from multimodal slice `[0:20]`.
- **Classes**: `[0, 1, 2]` (`0: legitimate`, `1: phishing`, `2: malware_distribution`).
- **Prediction Interface**: `clf.predict(X_url)` and `clf.predict_proba(X_url)` executed successfully on all 1486 samples.
- **Probability Normalization**: Sum of predicted probabilities equals 1.0 across all samples.
- **Prediction Distribution on Dataset**:
  - `phishing (1)`: 1283 (86.3%)
  - `legitimate (0)`: 200 (13.5%)
  - `malware_distribution (2)`: 3 (0.2%)

### B. Visual Model (`ml/models/visual_model.joblib`)
- **Input Dimension**: Expected **30** features; received 30 features from multimodal slice `[21:51]`.
- **Classes**: `[0, 1]` (`0: unverified_candidate`, `1: phishing`).
- **Prediction Interface**: `clf.predict(X_vis)` and `clf.predict_proba(X_vis)` executed successfully on all 1486 samples.
- **Probability Normalization**: Sum of predicted probabilities equals 1.0 across all samples.
- **Prediction Distribution on Dataset**:
  - `unverified_candidate (0)`: 810 (54.5%)
  - `phishing (1)`: 676 (45.5%)

---

## 3. Label Compatibility & Non-Interchangeability Analysis

| Property | URL Detection Model | Visual Screenshot Model | Dataset Ground Truth |
| :--- | :--- | :--- | :--- |
| **Number of Classes** | 3 Classes | 2 Classes | 2 Categories |
| **Class Definitions** | 0: `legitimate`<br>1: `phishing`<br>2: `malware_distribution` | 0: `unverified_candidate`<br>1: `phishing` | `unverified_candidate` (N=799)<br>`phishing` (N=687) |
| **Training Corpus** | PhiUSIIL (Lexical) | Zenodo V2.0 (Visual Screenshots) | Zenodo V2.0 Clean Subset |
| **Provenance** | Verified Benign / PhishTank | Tranco Popularity / PhishTank | Tranco (`tranco_candidate_unverified`) & PhishTank (`phishtank_verified`) |

### Key Compatibility Findings:
1. **Tranco Popularity != Benign Ground Truth**:
   The URL model predicts `phishing` for **605 / 799 (75.7%)** of the Tranco candidates. This occurs because popular domains frequently employ dynamic redirect paths, complex query parameters, and non-HTTPS entrypoints that mimic phishing lexical heuristics.
2. **Visual Model Candidate Stability**:
   In stark contrast to the URL model, the visual model flags only **24 / 799 (3.0%)** of Tranco candidates as phishing. Page rendering features (e.g. brightness, entropy, multi-color palettes) capture the structural complexity of genuine websites that lexical analysis misidentifies.
3. **Target Mapping Policy for Multimodal Fusion**:
   To prevent target leakage and unverified label conflation:
   - All dataset records explicitly retain `label_provenance`.
   - Binary multimodal target: `1 = phishing` (PhishTank verified), `0 = unverified_candidate` (Tranco popularity).
   - Any evaluation or documentation must note that class 0 represents popularity candidates rather than human-verified benign URLs.

---

## 4. Cross-Modality Concordance Analysis

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **Total Agreement (Both Agree)** | **849** | **57.13%** |
| Both Predict Phishing (Corroborated Threat) | 661 | 44.48% |
| Both Predict Candidate / Non-Phishing | 188 | 12.65% |
| Disagreement: URL says Phish, Visual says Candidate | 622 | 41.86% |
| Disagreement: URL says Candidate, Visual says Phish | 15 | 1.01% |

### Threat Detection Complementarity (PhishTank Phishing Samples: N = 687)

- **URL Model Phishing Recall**: **98.69%** (678/687)
- **Visual Model Phishing Recall**: **94.91%** (652/687)
- **Both Models Detect Phishing**: **643/687 (93.60%)**
- **Ensemble Union Detection (Either Detects)**: **687/687 (100.00%)**

> [!TIP]
> **Multimodal Advantage**: Combining lexical and visual signals captures 687 out of 687 verified phishing threats (100.00% coverage), demonstrating substantial complementary detection power.

---

## 5. Artifact Manifest

- **Multimodal Dataset**: `datasets/processed/multimodal_features.csv` (1486 rows, 60 columns)
- **Train Split (Domain-Separated)**: `datasets/processed/multimodal_train.csv` (1,190 rows, 0 domain leakage)
- **Test Split (Domain-Separated)**: `datasets/processed/multimodal_test.csv` (296 rows, 0 domain leakage)
- **Compatibility Audit JSON**: `ml/evaluation/multimodal_compatibility_audit.json`
- **Compatibility Report**: `docs/multimodal_compatibility_report.md`
