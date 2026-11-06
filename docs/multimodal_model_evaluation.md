# PhishGuard – Multimodal Classifier Evaluation Report (Phase 2, Step 2.6)

**Evaluation Date**: 2026-11-05T17:41:52.213807+00:00  
**Model Architecture**: `RandomForestClassifier` (Tuned parameters: `{'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2'}`)  
**Input Features**: Exactly 51 canonical fused features (0 metadata fields)  
**Partitioning**: Registered-domain grouped splitting (0% domain and 0% hash leakage)  

> [!WARNING]
> **Data Governance Notice & Provenance Limitation**:  
> CRITICAL LIMITATION: Legitimate class consists of Tranco candidates ('unverified_candidate') rather than manually verified benign ground truth. The multimodal classifier cannot be claimed as fully validated or deployed as a standalone production oracle without human/hybrid corroboration.  
> All metrics reported below represent classification performance against the candidate dataset baseline.

---

## 1. Executive Performance Summary (Held-Out Test Set: N = 296)

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **98.65%** | Overall classification accuracy on unseen registered domains |
| **Macro F1-Score** | **0.9864** | Unweighted mean F1 across both classes |
| **Phishing F1-Score** | **0.9852** | Primary threat detection harmonic mean |
| **Phishing Precision** | **100.00%** | Proportion of visual/URL phishing alarms that are true threats |
| **Phishing Recall** | **97.08%** | Proportion of verified phishing threats successfully caught |
| **ROC-AUC Score** | **0.9987** | Area under the Receiver Operating Characteristic curve |
| **5-Fold CV F1 (Train)** | **0.9880 (±0.0075)** | 5-fold stratified cross-validation on training partition |

---

## 2. Confusion Matrix (Held-Out Test Set: N = 296)

| Ground Truth \ Prediction | Predicted Candidate (0) | Predicted Phishing (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Candidate (Tranco)** | **159** (TN) | 0 (FP) | 159 |
| **Actual Phishing (PhishTank)** | 4 (FN) | **133** (TP) | 137 |

---

## 3. Detailed Classification Report

```text
                      precision    recall  f1-score   support

unverified_candidate       0.98      1.00      0.99       159
            phishing       1.00      0.97      0.99       137

            accuracy                           0.99       296
           macro avg       0.99      0.99      0.99       296
        weighted avg       0.99      0.99      0.99       296

```

---

## 4. Cross-Model Empirical Comparison (Part C)

All three models were evaluated against the identical 296 held-out test records:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Macro F1 | ROC-AUC | Phishing Caught | False Alarms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **URL-Only Model** (`url_model.joblib`) | 59.46% | 53.31% | 100.00% | 0.6954 | 0.5447 | 0.4664 | 137/137 | 120/159 |
| **Visual-Only Model** (`visual_model.joblib`) | 80.07% | 80.95% | 74.45% | 0.7757 | 0.7982 | 0.8491 | 102/137 | 24/159 |
| **Multimodal Model** (`multimodal_model.joblib`) | **98.65%** | **100.00%** | **97.08%** | **0.9852** | **0.9864** | **0.9987** | **133/137** | **0/159** |

### Triangulation & Complementarity Analysis:
1. **False Alarm Elimination on Popular Candidates**:
   - The URL-only model suffered a false alarm rate of 75.5% (120/159) on Tranco popularity candidates because complex deep URLs and marketing parameters trigger lexical threat heuristics.
   - The visual-only model had 24 false alarms (15.1%).
   - The multimodal model reduced false alarms to **0/159 (0.0%)**, yielding **100.00% precision**.
2. **Threat Detection Recall Boost**:
   - The visual-only model missed 35 phishing sites (74.45% recall).
   - The multimodal model caught **133/137 threats (97.08% recall)**, recovering 31 threats missed by visual analysis alone.
3. **Overall F1-Score Gains**:
   - Multimodal F1 (0.9852) shows an improvement of **+0.2095** over visual-only and **+0.2898** over URL-only.

---

## 5. Top 15 Feature Importances (Multimodal Synergy)

| Rank | Feature Name | Modality Domain | Gini Importance |
| :---: | :--- | :---: | :---: |
| 1 | `num_slashes` | URL Lexical | 0.19582 |
| 2 | `path_length` | URL Lexical | 0.18645 |
| 3 | `url_entropy` | URL Lexical | 0.10422 |
| 4 | `url_length` | URL Lexical | 0.10329 |
| 5 | `num_digits` | URL Lexical | 0.04179 |
| 6 | `std_brightness` | Screenshot Visual | 0.03041 |
| 7 | `dominant_color_3_proportion` | Screenshot Visual | 0.02847 |
| 8 | `std_green` | Screenshot Visual | 0.02768 |
| 9 | `color_variance` | Screenshot Visual | 0.02762 |
| 10 | `std_blue` | Screenshot Visual | 0.02152 |
| 11 | `hostname_length` | URL Lexical | 0.01676 |
| 12 | `image_entropy` | Screenshot Visual | 0.01480 |
| 13 | `std_red` | Screenshot Visual | 0.01379 |
| 14 | `dominant_color_2_proportion` | Screenshot Visual | 0.01334 |
| 15 | `num_dots` | URL Lexical | 0.01132 |

---

## 6. Artifact Manifest

- Trained Model: `ml/models/multimodal_model.joblib`
- Feature Names: `ml/models/multimodal_feature_names.json`
- Class Mapping: `ml/models/multimodal_class_mapping.json`
- Evaluation JSON: `ml/evaluation/multimodal_model_eval.json`
- Evaluation Markdown: `docs/multimodal_model_evaluation.md`
